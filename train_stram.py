import argparse
import datetime
import time
import torch
import copy
import numpy as np
import tools.utils.misc as utils

from numpy import mat
from pathlib import Path
from info_nce import InfoNCE
from tensorboardX import SummaryWriter
from scipy.optimize import linear_sum_assignment
from tools.utils import util
from tools.models.stram import build as build_model
from tools.utils.util import Giou_np
from tools.utils.util import linear_assignment

def get_args_parser():
    parser = argparse.ArgumentParser('train stram', add_help=False)
    parser.add_argument('--lr', default=1e-5, type=float)
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

    losses_min = None
    loss = InfoNCE(negative_mode='paired')
    for epoch in range(args.start_epoch, args.epochs):
        if epoch != 0 and epoch % args.lr_drop == 0:
            optimizer.param_groups[0]["lr"] *= 0.1
        loss_epoch_temporal = 0
        loss_epoch_spatial = 0

        for train_seq, seq_data in objects_human.items():
            for frame_id in range(1, len(seq_data.keys())):
                trajectories = objects_human[train_seq][frame_id]
                humans = objects_human[train_seq][frame_id+1]
                marks = objects_mark[train_seq][frame_id+1]

                # temporal
                trajectories = torch.tensor(trajectories)
                humans = torch.tensor(humans)
                trajectories_box = list(np.array(trajectories)[:, 1:5])
                humans_box = list(np.array(humans)[:, 1:5])
                _ious = util.ious(trajectories_box, humans_box)
                cost_matrix = 1 - _ious
                matches, utrack, udet = linear_assignment(cost_matrix, thresh=0.8)
                triplets_human_trajectory = [] # humans as anchors
                triplets_trajectory_human = [] # trajectories as anchors
                for trajectory_index, human_index in matches:
                    negs = [j for j in range(trajectories.shape[0]) if j != trajectory_index]
                    triplet = [human_index, trajectory_index]
                    triplet.append(negs)
                    triplets_human_trajectory.append(triplet)
                    negs_rev = [j for j in range(humans.shape[0]) if j != human_index]
                    triplet_rev = [trajectory_index, human_index]
                    triplet_rev.append(negs_rev)
                    triplets_trajectory_human.append(triplet_rev)
                for udet_index in udet:
                    negs = [j for j in range(trajectories.shape[0])]
                    triplet = [udet_index, -1]
                    triplet.append(negs)
                    triplets_human_trajectory.append(triplet)
                for utrack_index in utrack:
                    negs = [j for j in range(humans.shape[0])]
                    triplet_rev = [utrack_index, -1]
                    triplet_rev.append(negs)
                    triplets_trajectory_human.append(triplet_rev)

                # spatial
                iou_list = []
                for mark in marks:
                    item = []
                    for person in humans_box:
                        iou = Giou_np(np.array([mark[1], mark[2], mark[3], mark[4]]), np.array([person[0], person[1], person[2], person[3]]))
                        if np.isnan(iou):
                            iou = 1.0
                        item.append(iou)
                    iou_list.append(item)
                iou_mat = mat(iou_list)
                match_index_list = linear_sum_assignment(1 - iou_mat)
                triplets_human_mark = [] # humans as anchors
                triplets_mark_human = [] # marks as anchors
                for mark_index, human_index in zip(match_index_list[0], match_index_list[1]):
                    negs = [j for j in range(len(marks)) if j != mark_index]
                    triplet = [human_index, mark_index]
                    triplet.append(negs)
                    triplets_human_mark.append(triplet)
                    
                    negs_rev = [j for j in range(humans.shape[0]) if j != human_index]
                    triplet_rev = [mark_index, human_index]
                    triplet_rev.append(negs_rev)
                    triplets_mark_human.append(triplet_rev)

                trajectories = trajectories[:, 1:].unsqueeze(0).to(device)
                humans = humans[:, 1:].unsqueeze(0).to(device)
                marks = torch.tensor(marks)[:, 1:].unsqueeze(0).to(device)
                trajectories_aligned_feature, marks_aligned_feature, humans_aligned_feature = model(trajectories, marks, humans)
                trajectories_aligned_feature = trajectories_aligned_feature.squeeze(0)
                humans_aligned_feature = humans_aligned_feature.squeeze(0)
                marks_aligned_feature = marks_aligned_feature.squeeze(0)

                # temporal
                anchor_features_t = None
                positive_features_t = None
                negtive_features_t = None
                for sample in triplets_human_trajectory:
                    if anchor_features_t == None:
                        anchor_features_t = humans_aligned_feature[sample[0]].unsqueeze(0)
                    else:
                        anchor_features_t = torch.cat((anchor_features_t, humans_aligned_feature[sample[0]].unsqueeze(0)), dim=0)
                    if positive_features_t == None:
                        if sample[1] == -1:
                            positive_features_t = humans_aligned_feature[sample[0]].unsqueeze(0)
                        else:
                            positive_features_t = trajectories_aligned_feature[sample[1]].unsqueeze(0)
                    else:
                        if sample[1] == -1:
                            positive_features_t = torch.cat((positive_features_t, humans_aligned_feature[sample[0]].unsqueeze(0)), dim=0)
                        else:
                            positive_features_t = torch.cat((positive_features_t, trajectories_aligned_feature[sample[1]].unsqueeze(0)), dim=0)
                    if negtive_features_t == None:
                        if sample[1] == -1:
                            negtive_features_t = trajectories_aligned_feature[1:].unsqueeze(0)
                        else:
                            negtive_features_t = trajectories_aligned_feature[torch.tensor(sample[2])].unsqueeze(0)
                    else:
                        if sample[1] == -1:
                            negtive_features_t = torch.cat((negtive_features_t, trajectories_aligned_feature[1:].unsqueeze(0)), dim=0)
                        else:
                            negtive_features_t = torch.cat((negtive_features_t, trajectories_aligned_feature[torch.tensor(sample[2])].unsqueeze(0)), dim=0)
                loss_t_h_t = loss(anchor_features_t, positive_features_t, negtive_features_t)
                anchor_features_t = None
                positive_features_t = None
                negtive_features_t = None
                for sample in triplets_trajectory_human:
                    if anchor_features_t == None:
                        anchor_features_t = trajectories_aligned_feature[sample[0]].unsqueeze(0)
                    else:
                        anchor_features_t = torch.cat((anchor_features_t, trajectories_aligned_feature[sample[0]].unsqueeze(0)), dim=0)
                    if positive_features_t == None:
                        if sample[1] == -1:
                            positive_features_t = trajectories_aligned_feature[sample[0]].unsqueeze(0)
                        else:
                            positive_features_t = humans_aligned_feature[sample[1]].unsqueeze(0)
                    else:
                        if sample[1] == -1:
                            positive_features_t = torch.cat((positive_features_t, trajectories_aligned_feature[sample[0]].unsqueeze(0)), dim=0)
                        else:
                            positive_features_t = torch.cat((positive_features_t, humans_aligned_feature[sample[1]].unsqueeze(0)), dim=0)
                    if negtive_features_t == None:
                        if sample[1] == -1:
                            negtive_features_t = humans_aligned_feature[1:].unsqueeze(0)
                        else:
                            negtive_features_t = humans_aligned_feature[torch.tensor(sample[2])].unsqueeze(0)
                    else:
                        if sample[1] == -1:
                            negtive_features_t = torch.cat((negtive_features_t, humans_aligned_feature[1:].unsqueeze(0)), dim=0)
                        else:
                            negtive_features_t = torch.cat((negtive_features_t, humans_aligned_feature[torch.tensor(sample[2])].unsqueeze(0)), dim=0)
                loss_t_t_h = loss(anchor_features_t, positive_features_t, negtive_features_t)
                
                # spatial
                anchor_features_s = None
                positive_features_s = None
                negtive_features_s = None
                for sample in triplets_human_mark:
                    if anchor_features_s == None:
                        anchor_features_s = humans_aligned_feature[sample[0]].unsqueeze(0)
                    else:
                        anchor_features_s = torch.cat((anchor_features_s, humans_aligned_feature[sample[0]].unsqueeze(0)), dim=0)
                    if positive_features_s == None:
                        positive_features_s = marks_aligned_feature[sample[1]].unsqueeze(0)
                    else:
                        positive_features_s = torch.cat((positive_features_s, marks_aligned_feature[sample[1]].unsqueeze(0)), dim=0)
                    if negtive_features_s == None:
                        negtive_features_s = marks_aligned_feature[torch.tensor(sample[2])].unsqueeze(0)
                    else:
                        negtive_features_s = torch.cat((negtive_features_s, marks_aligned_feature[torch.tensor(sample[2])].unsqueeze(0)), dim=0)
                loss_s_h_m = loss(anchor_features_s, positive_features_s, negtive_features_s)
                anchor_features_s = None
                positive_features_s = None
                negtive_features_s = None
                for sample in triplets_mark_human:
                    if anchor_features_s == None:
                        anchor_features_s = marks_aligned_feature[sample[0]].unsqueeze(0)
                    else:
                        anchor_features_s = torch.cat((anchor_features_s, marks_aligned_feature[sample[0]].unsqueeze(0)), dim=0)
                    if positive_features_s == None:
                        positive_features_s = humans_aligned_feature[sample[1]].unsqueeze(0)
                    else:
                        positive_features_s = torch.cat((positive_features_s, humans_aligned_feature[sample[1]].unsqueeze(0)), dim=0)
                    if negtive_features_s == None:
                        negtive_features_s = humans_aligned_feature[torch.tensor(sample[2])].unsqueeze(0)
                    else:
                        negtive_features_s = torch.cat((negtive_features_s, humans_aligned_feature[torch.tensor(sample[2])].unsqueeze(0)), dim=0)
                loss_s_m_h = loss(anchor_features_s, positive_features_s, negtive_features_s)
                            
                loss_t = loss_t_h_t + loss_t_t_h
                loss_s = loss_s_h_m + loss_s_m_h
                loss_all = loss_t + loss_s

                if not torch.isnan(loss_all).any():
                    optimizer.zero_grad()
                    loss_all.backward()
                    optimizer.step()
                    
                if not torch.isnan(loss_t).any():
                    loss_epoch_temporal += loss_t
                if not torch.isnan(loss_s).any():
                    loss_epoch_spatial += loss_s
                    
        f_1 = open(f"{args.output_dir}/train.txt", mode="a+")
        f_1.write(f"epoch:{epoch},   temporal loss:{loss_epoch_temporal} ,  spatial loss:{loss_epoch_spatial}\n")
        f_1.close()

        if not losses_min or loss_epoch_temporal+loss_epoch_spatial  < losses_min:
            losses_min = loss_epoch_temporal +loss_epoch_spatial

            filename=f'{args.output_dir}/model_best.pth'
            utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, filename)
        writer_loss.add_scalar('epoch_loss', loss_epoch_temporal+loss_epoch_spatial, epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train stram', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)