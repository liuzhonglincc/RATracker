import argparse
import datetime
import time
import torch
import copy
import numpy as np
import tools.utils.misc as utils

from torch.utils.data import DataLoader
from pathlib import Path
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from info_nce import InfoNCE
from tools.utils import util
from tools.models.tram import build as build_model
from tools.utils.util import DatasetTrain
from tools.utils.util import linear_assignment

def get_args_parser():
    parser = argparse.ArgumentParser('train tram', add_help=False)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=10, type=int)
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume_model', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--merger_dropout', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='MOT17', help='MOT17 or MOT20 ')
    parser.add_argument('--half', action='store_true', help='Whether to use half training set for training')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=128)
    return parser

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)
    device = torch.device(args.device)

    model = build_model(args)
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        print(f"args.weight_decay:{args.weight_decay}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.resume_model:
        checkpoint = torch.load(args.resume_model, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
            
        p_groups = copy.deepcopy(optimizer.param_groups)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for pg, pg_old in zip(optimizer.param_groups, p_groups):
            pg['lr'] = pg_old['lr']
            pg['initial_lr'] = pg_old['initial_lr']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.override_resumed_lr_drop = True
        if args.override_resumed_lr_drop:
            print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
            lr_scheduler.step_size = args.lr_drop
            lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        lr_scheduler.step(lr_scheduler.last_epoch)
        args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()

    writer_loss = SummaryWriter(f'{args.output_dir}/loss_log')
    if args.dataset == "MOT17":
        train_seqs = ["MOT17-02-FRCNN",
                      "MOT17-04-FRCNN",
                      "MOT17-05-FRCNN",
                      "MOT17-09-FRCNN",
                      "MOT17-10-FRCNN",
                      "MOT17-11-FRCNN",
                      "MOT17-13-FRCNN"]
        objects = {
            "MOT17-02-FRCNN":{},
            "MOT17-04-FRCNN":{},
            "MOT17-05-FRCNN":{},
            "MOT17-09-FRCNN":{},
            "MOT17-10-FRCNN":{},
            "MOT17-11-FRCNN":{},
            "MOT17-13-FRCNN":{}
        }
    elif args.dataset == "MOT20":
        train_seqs = ["MOT20-01",
                    "MOT20-02",
                    "MOT20-03",
                    "MOT20-05"]
        objects = {
            "MOT20-01":{},
            "MOT20-02":{},
            "MOT20-03":{},
            "MOT20-05":{}
        }
    
    for train_seq in train_seqs:
        if args.half:
            f_r = open(f"datasets/{args.dataset}/train/{train_seq}/gt/gt_train_half.txt", mode="r")
        else:
            f_r = open(f"datasets/{args.dataset}/train/{train_seq}/gt/gt.txt", mode="r")
        for line in f_r:
            lines = line.split(",")
            if int(lines[0]) not in objects[train_seq].keys():
                objects[train_seq][int(lines[0])] = [[int(lines[1]), float(lines[2]), float(lines[3]), float(lines[2])+float(lines[4]), float(lines[3])+float(lines[5])]]
            else:
                objects[train_seq][int(lines[0])].append([int(lines[1]), float(lines[2]), float(lines[3]), float(lines[2])+float(lines[4]), float(lines[3])+float(lines[5])])

    train_data = []
    if args.dataset == "MOT17":
        max_num = 110
    elif args.dataset == "MOT20":
        max_num = 260
    for train_seq, seq_data in objects.items():
        for frame_id, frame_data in seq_data.items():
            if frame_id+1 not in seq_data.keys():
                continue
            trajectories = torch.tensor(frame_data)
            humans = torch.tensor(seq_data[frame_id+1])

            trajectories_box = list(np.array(trajectories)[:, 1:5])
            humans_box = list(np.array(humans)[:, 1:5])
            
            _ious = util.ious(trajectories_box, humans_box)
            cost_matrix = 1 - _ious
            matches, utrack, udet = linear_assignment(cost_matrix, thresh=0.9)

            triplets = [] # humans as anchors
            triplets_rev = [] # trajectories as anchors
            for trajectory_index, human_index in matches:
                negs = [j for j in range(trajectories.shape[0]) if j != trajectory_index]
                while len(negs) < max_num-1:
                    negs.append(-1)
                triplet = [human_index, trajectory_index]
                triplet.extend(negs)
                triplets.append(triplet)

                negs_rev = [j for j in range(humans.shape[0]) if j != human_index]
                while len(negs_rev) < max_num-1:
                    negs_rev.append(-1)
                triplet_rev = [trajectory_index, human_index]
                triplet_rev.extend(negs_rev)
                triplets_rev.append(triplet_rev)
            for udet_index in udet:
                negs = [j for j in range(trajectories.shape[0])]
                while len(negs) < max_num-1:
                    negs.append(-1)
                triplet = [udet_index, -1]
                triplet.extend(negs)
                triplets.append(triplet)
            for utrack_index in utrack:
                negs = [j for j in range(humans.shape[0])]
                while len(negs) < max_num-1:
                    negs.append(-1)
                triplet_rev = [utrack_index, -1]
                triplet_rev.extend(negs)
                triplets_rev.append(triplet_rev)

            trajectories_feature = trajectories[:, 1:]
            humans_feature = humans[:, 1:]
            trajectories_feature_padding = torch.zeros([max_num - trajectories_feature.shape[0], trajectories_feature.shape[1]],dtype=torch.float32)
            trajectories_feature = torch.cat((trajectories_feature, trajectories_feature_padding), dim=0)
            humans_feature_padding = torch.zeros([max_num - humans_feature.shape[0],humans_feature.shape[1]],dtype=torch.float32)
            humans_feature = torch.cat((humans_feature, humans_feature_padding), dim=0)

            while len(triplets) < max_num:
                triplets.append([-1 for __ in range(max_num+1)])
            triplets = torch.tensor(triplets)
            while len(triplets_rev) < max_num:
                triplets_rev.append([-1 for __ in range(max_num+1)])
            triplets_rev = torch.tensor(triplets_rev)

            train_data.append([trajectories_feature, humans_feature, triplets, triplets_rev])

    dataset = DatasetTrain(train_data)

    losses_min = None
    loss = InfoNCE(negative_mode='paired')
    for epoch in range(args.start_epoch, args.epochs):
        if epoch != 0 and epoch % args.lr_drop == 0:
            optimizer.param_groups[0]["lr"] *= 0.1
        loss_epoch = 0
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        for _, (trajectories_feature, humans_feature, triplets, triplets_rev) in enumerate(dataloader):
            trajectories_aligned_feature, humans_aligned_feature = model(trajectories_feature.to(device), humans_feature.to(device))
            anchor_features = None
            positive_features = None
            negtive_features = None

            for taf, haf, triplet, triplet_rev in zip(trajectories_aligned_feature, humans_aligned_feature, triplets, triplets_rev):
                # human as anchor
                # delete padding
                not_padding = triplet[:, 0] != -1
                triplet = triplet[not_padding]
                positive_exist = triplet[:, 1] != -1
                positive_no_exist = triplet[:, 1] == -1
                triplet_pe = triplet[positive_exist]
                triplet_pne = triplet[positive_no_exist]
                if triplet_pe.shape[0] > 0 :
                    triplet_pe_anchor = triplet_pe[:, 0]
                    triplet_pe_positive = triplet_pe[:, 1]
                    triplet_pe_negative = triplet_pe[:, 2:]
                    if anchor_features is None:
                        anchor_features = haf[triplet_pe_anchor]
                    else:
                        anchor_features = torch.cat((anchor_features, haf[triplet_pe_anchor]), dim=0)
                    if positive_features is None:
                        positive_features = taf[triplet_pe_positive]
                    else:
                        positive_features = torch.cat((positive_features, taf[triplet_pe_positive]), dim=0)
                    if negtive_features is None:
                        negtive_features = taf[triplet_pe_negative]
                    else:
                        negtive_features = torch.cat((negtive_features, taf[triplet_pe_negative]), dim=0)
                if triplet_pne.shape[0] > 0 :
                    triplet_pne_anchor = triplet_pne[:, 0]
                    triplet_pne_negative = triplet_pne[:, 2:]
                    if anchor_features is None:
                        anchor_features = haf[triplet_pne_anchor]
                    else:
                        anchor_features = torch.cat((anchor_features, haf[triplet_pne_anchor]), dim=0)
                    # take self-feature as positive
                    if positive_features is None:
                        positive_features = haf[triplet_pne_anchor]
                    else:
                        positive_features = torch.cat((positive_features, haf[triplet_pne_anchor]), dim=0)
                    if negtive_features is None:
                        negtive_features = taf[triplet_pne_negative]
                    else:
                        negtive_features = torch.cat((negtive_features, taf[triplet_pne_negative]), dim=0)

                # trajectory as anchor
                not_padding = triplet_rev[:, 0] != -1
                triplet_rev = triplet_rev[not_padding]
                positive_exist = triplet_rev[:, 1] != -1
                positive_no_exist = triplet_rev[:, 1] == -1
                triplet_rev_pe = triplet_rev[positive_exist]
                triplet_rev_pne = triplet_rev[positive_no_exist]
                if triplet_rev_pe.shape[0] > 0 :
                    triplet_rev_pe_anchor = triplet_rev_pe[:, 0]
                    triplet_rev_pe_positive = triplet_rev_pe[:, 1]
                    triplet_rev_pe_negative = triplet_rev_pe[:, 2:]
                    if anchor_features is None:
                        anchor_features = taf[triplet_rev_pe_anchor]
                    else:
                        anchor_features = torch.cat((anchor_features, taf[triplet_rev_pe_anchor]), dim=0)
                    if positive_features is None:
                        positive_features = haf[triplet_rev_pe_positive]
                    else:
                        positive_features = torch.cat((positive_features, haf[triplet_rev_pe_positive]), dim=0)
                    if negtive_features is None:
                        negtive_features = haf[triplet_rev_pe_negative]
                    else:
                        negtive_features = torch.cat((negtive_features, haf[triplet_rev_pe_negative]), dim=0)
                if triplet_rev_pne.shape[0] > 0 :
                    triplet_rev_pne_anchor = triplet_rev_pne[:, 0]
                    triplet_rev_pne_negative = triplet_rev_pne[:, 2:]
                    if anchor_features is None:
                        anchor_features = taf[triplet_rev_pne_anchor]
                    else:
                        anchor_features = torch.cat((anchor_features, taf[triplet_rev_pne_anchor]), dim=0)
                    # take self-feature as positive
                    if positive_features is None:
                        positive_features = taf[triplet_rev_pne_anchor]
                    else:
                        positive_features = torch.cat((positive_features, taf[triplet_rev_pne_anchor]), dim=0)
                    if negtive_features is None:
                        negtive_features = haf[triplet_rev_pne_negative]
                    else:
                        negtive_features = torch.cat((negtive_features, haf[triplet_rev_pne_negative]), dim=0)

            infonce_loss = loss(anchor_features, positive_features, negtive_features)

            if not torch.isnan(infonce_loss).any():
                loss_epoch += infonce_loss
                optimizer.zero_grad()
                infonce_loss.backward()
                optimizer.step()

        f_1 = open(f"{args.output_dir}/train.txt", mode="a+")
        f_1.write(f"epoch:{epoch},   info-nce loss for tram:{loss_epoch}\n")
        f_1.close()

        if not losses_min or loss_epoch < losses_min:
            losses_min = loss_epoch

            filename=f'{args.output_dir}/model_best.pth'
            utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, filename)
        writer_loss.add_scalar('epoch_loss', loss_epoch , epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train tram', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)