import argparse
import os
import torch
import motmetrics as mm
from loguru import logger
import sys
sys.path.append(".")
from tracker.baseline import Baseline
from tracker.evaluation import Evaluator

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("--det_thresh", type=float, default=0.7, help=" ")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--match_thresh_2", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--dataset", type=str, default="MOT17", help=' ')
    parser.add_argument("--tram", action='store_true')
    parser.add_argument("--sram", action='store_true')
    parser.add_argument("--stram", action='store_true')
    parser.add_argument('--merger_dropout', type=float, default=0.1)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument("--pretrained", type=str, default="", help=' ')
    parser.add_argument('--mark_ration', type=float, default=0.2)
    parser.add_argument('--weight_t', type=float, default=0.1)
    parser.add_argument('--weight_s', type=float, default=0.1)
    parser.add_argument('--weight_st', type=float, default=0.5)
    return parser

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def evaluate(args, val_seqs, results_folder):
    seqs_val_start = {"MOT17-02-FRCNN":302, "MOT17-04-FRCNN":527, "MOT17-05-FRCNN":420,
                "MOT17-09-FRCNN":264, "MOT17-10-FRCNN":329, "MOT17-11-FRCNN":452, "MOT17-13-FRCNN":377}
    yolox_dets = {"MOT17-02-FRCNN":{}, "MOT17-04-FRCNN":{}, "MOT17-05-FRCNN":{},
                "MOT17-09-FRCNN":{}, "MOT17-10-FRCNN":{}, "MOT17-11-FRCNN":{}, "MOT17-13-FRCNN":{}}
    for seq in val_seqs:
        file_path = f"datasets/YOLOX_detection/{args.dataset}/{seq}/byte065.txt"
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            line_data = line.split(",")
            frame_id = int(line_data[0])
            if frame_id >= seqs_val_start[seq]:
                if frame_id - seqs_val_start[seq] + 1 not in yolox_dets[seq]:
                    yolox_dets[seq][frame_id - seqs_val_start[seq] + 1] = [[float(line_data[2]), float(line_data[3]), float(line_data[4]), float(line_data[5]), float(line_data[6])]]
                else:
                    yolox_dets[seq][frame_id - seqs_val_start[seq] + 1].append([float(line_data[2]), float(line_data[3]), float(line_data[4]), float(line_data[5]), float(line_data[6])])
    tracker = Baseline(args=args)
    for seq, seq_yolox_dets in yolox_dets.items():
        results = []
        for frame_id, frame_yolox_dets in seq_yolox_dets.items():
            online_targets = tracker.update(torch.tensor(frame_yolox_dets).to(torch.device("cuda")))
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))
        write_results(f"{results_folder}/{seq}.txt", results)

def eval_seq(seq_num, gt_path, pred_path):
    result_filename = os.path.join(pred_path, f'{seq_num}.txt') # predict result
    evaluator = Evaluator(gt_path, seq_num)
    accs = evaluator.eval_file(result_filename)
    return accs

def main(args):
    results_folder = os.path.join("track_results")
    os.makedirs(results_folder, exist_ok=True)

    val_seqs = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
                "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
    # start evaluate
    evaluate(args, val_seqs, results_folder)

    accs = []
    for seq_num in val_seqs:
        accs.append(eval_seq(seq_num, f"datasets/{args.dataset}/train", results_folder))

    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, val_seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        # formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )

    print(strsummary)

if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)


