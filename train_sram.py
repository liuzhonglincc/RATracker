import argparse
import datetime
import time
import torch
import copy
import numpy as np
import tools.utils.misc as utils

from info_nce import InfoNCE
from numpy import mat
from pathlib import Path
from tensorboardX import SummaryWriter
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tools.utils.util import DatasetTrain
from tools.utils.util import Giou_np
from tools.models.sram import build as build_model

def get_args_parser():
    parser = argparse.ArgumentParser('train sram', add_help=False)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
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
    parser.add_argument('--mark_ration', type=float, default=0.2)
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
        objects_human = {
            "MOT17-02-FRCNN":{},
            "MOT17-04-FRCNN":{},
            "MOT17-05-FRCNN":{},
            "MOT17-09-FRCNN":{},
            "MOT17-10-FRCNN":{},
            "MOT17-11-FRCNN":{},
            "MOT17-13-FRCNN":{}
        }
        objects_mark = {
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
        objects_human = {
            "MOT20-01":{},
            "MOT20-02":{},
            "MOT20-03":{},
            "MOT20-05":{}
        }
        objects_mark = {
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
            tlx = float(lines[2])
            tly = float(lines[3])
            w = float(lines[4])
            h = float(lines[5])
            mark_tlx = tlx + args.mark_ration*w
            mark_tly = tly + args.mark_ration*h
            mark_w = (1-2*args.mark_ration)*w
            mark_h = (1-2*args.mark_ration)*h
            if int(lines[0]) not in objects_human[train_seq].keys():
                objects_human[train_seq][int(lines[0])] = [[int(lines[1]), float(lines[2]), float(lines[3]), float(lines[2])+float(lines[4]), float(lines[3])+float(lines[5])]]
                objects_mark[train_seq][int(lines[0])] = [[int(lines[1]), mark_tlx, mark_tly, mark_tlx+mark_w, mark_tly+mark_h]]
            else:
                objects_human[train_seq][int(lines[0])].append([int(lines[1]), float(lines[2]), float(lines[3]), float(lines[2])+float(lines[4]), float(lines[3])+float(lines[5])])
                objects_mark[train_seq][int(lines[0])].append([int(lines[1]), mark_tlx, mark_tly, mark_tlx+mark_w, mark_tly+mark_h])

    train_data = []
    if args.dataset == "MOT17":
        max_num = 110
    elif args.dataset == "MOT20":
        max_num = 260
    for train_seq, seq_data in objects_human.items():
        for frame_id, frame_person_data in seq_data.items():
            frame_mark_data = objects_mark[train_seq][frame_id]

            humans = torch.tensor(frame_person_data)
            marks = torch.tensor(frame_mark_data)

            iou_list = []
            for mark in frame_mark_data:
                item = []
                for person in frame_person_data:
                    iou = Giou_np(np.array([mark[1], mark[2], mark[3], mark[4]]), np.array([person[1], person[2], person[3], person[4]]))
                    if np.isnan(iou):
                        iou = 1.0
                    item.append(iou)
                iou_list.append(item)
            iou_mat = mat(iou_list)
            cost_matrix = 1 - iou_mat
            match_index_list = linear_sum_assignment(cost_matrix)

            triplets = [] # humans as anchors
            triplets_rev = [] # marks as anchors
            for mark_index, human_index in zip(match_index_list[0], match_index_list[1]):
                negs = [j for j in range(marks.shape[0]) if j != mark_index]
                while len(negs) < max_num-1:
                    negs.append(-1)
                triplet = [human_index, mark_index]
                triplet.extend(negs)
                triplets.append(triplet)

                negs_rev = [j for j in range(humans.shape[0]) if j != human_index]
                while len(negs_rev) < max_num-1:
                    negs_rev.append(-1)
                triplet_rev = [mark_index, human_index]
                triplet_rev.extend(negs_rev)
                triplets_rev.append(triplet_rev)

            humans_feature = humans[:, 1:]
            marks_feature = marks[:, 1:]
            humans_feature_padding = torch.zeros([max_num - humans_feature.shape[0],humans_feature.shape[1]],dtype=torch.float32)
            humans_feature = torch.cat((humans_feature, humans_feature_padding), dim=0)
            marks_feature_padding = torch.zeros([max_num - marks_feature.shape[0],marks_feature.shape[1]],dtype=torch.float32)
            marks_feature = torch.cat((marks_feature, marks_feature_padding), dim=0)

            while len(triplets) < max_num:
                triplets.append([-1 for __ in range(max_num+1)])
            triplets = torch.tensor(triplets)
            while len(triplets_rev) < max_num:
                triplets_rev.append([-1 for __ in range(max_num+1)])
            triplets_rev = torch.tensor(triplets_rev)

            train_data.append([humans_feature, marks_feature, triplets, triplets_rev])

    dataset = DatasetTrain(train_data)

    losses_min = None
    loss = InfoNCE(negative_mode='paired')
    for epoch in range(args.start_epoch, args.epochs):
        if epoch != 0 and epoch % args.lr_drop == 0:
            optimizer.param_groups[0]["lr"] *= 0.1
        loss_epoch = 0
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        for _, (humans_feature, marks_feature, triplets, triplets_rev) in enumerate(dataloader):
            humans_aligned_feature, marks_aligned_feature = model(humans_feature.to(device), marks_feature.to(device))
            anchor_features = None
            positive_features = None
            negtive_features = None

            for haf, maf, triplet, triplet_rev in zip(humans_aligned_feature, marks_aligned_feature, triplets, triplets_rev):
                # human as anchor
                # delete padding
                not_padding = triplet[:, 0] != -1
                triplet = triplet[not_padding]
                triplet_pe = triplet
                if triplet_pe.shape[0] > 0 :
                    triplet_pe_anchor = triplet_pe[:, 0]
                    triplet_pe_positive = triplet_pe[:, 1]
                    triplet_pe_negative = triplet_pe[:, 2:]
                    if anchor_features is None:
                        anchor_features = haf[triplet_pe_anchor]
                    else:
                        anchor_features = torch.cat((anchor_features, haf[triplet_pe_anchor]), dim=0)
                    if positive_features is None:
                        positive_features = maf[triplet_pe_positive]
                    else:
                        positive_features = torch.cat((positive_features, maf[triplet_pe_positive]), dim=0)
                    if negtive_features is None:
                        negtive_features = maf[triplet_pe_negative]
                    else:
                        negtive_features = torch.cat((negtive_features, maf[triplet_pe_negative]), dim=0)

                # mark as anchor
                not_padding = triplet_rev[:, 0] != -1
                triplet_rev = triplet_rev[not_padding]
                triplet_rev_pe = triplet_rev
                if triplet_rev_pe.shape[0] > 0 :
                    triplet_rev_pe_anchor = triplet_rev_pe[:, 0]
                    triplet_rev_pe_positive = triplet_rev_pe[:, 1]
                    triplet_rev_pe_negative = triplet_rev_pe[:, 2:]
                    if anchor_features is None:
                        anchor_features = maf[triplet_rev_pe_anchor]
                    else:
                        anchor_features = torch.cat((anchor_features, maf[triplet_rev_pe_anchor]), dim=0)
                    if positive_features is None:
                        positive_features = haf[triplet_rev_pe_positive]
                    else:
                        positive_features = torch.cat((positive_features, haf[triplet_rev_pe_positive]), dim=0)
                    if negtive_features is None:
                        negtive_features = haf[triplet_rev_pe_negative]
                    else:
                        negtive_features = torch.cat((negtive_features, haf[triplet_rev_pe_negative]), dim=0)

            infonce_loss = loss(anchor_features, positive_features, negtive_features)

            if not torch.isnan(infonce_loss).any():
                loss_epoch += infonce_loss
                optimizer.zero_grad()
                infonce_loss.backward()
                optimizer.step()

        f_1 = open(f"{args.output_dir}/train.txt", mode="a+")
        f_1.write(f"epoch:{epoch},   nfo-nce loss for sram:{loss_epoch}\n")
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
    parser = argparse.ArgumentParser('train sram', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)