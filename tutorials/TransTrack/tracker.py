"""
Copyright (c) https://github.com/xingyizhou/CenterTrack
Modified by Peize Sun, Rufeng Zhang
"""
# coding: utf-8
import torch
from scipy.optimize import linear_sum_assignment
from util import box_ops
import copy
from scipy.spatial.distance import cdist
import numpy as np

def embedding_distance_1(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    track_features = np.asarray([track for track in tracks], dtype=np.float)
    det_features = np.asarray([detect for detect in detections], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix

class Tracker(object):
    def __init__(self, score_thresh, max_age=32):        
        self.score_thresh = score_thresh
        self.max_age = max_age        
        self.id_count = 0
        self.tracks_dict = dict()
        self.tracks = list()
        self.unmatched_tracks = list()
        self.reset_all()
        
    def reset_all(self):
        self.id_count = 0
        self.tracks_dict = dict()
        self.tracks = list()
        self.unmatched_tracks = list()
    
    def init_track(self, results):

        scores = results["scores"]
        classes = results["labels"]
        bboxes = results["boxes"]  # x1y1x2y2
        
        ret = list()
        ret_dict = dict()
        for idx in range(scores.shape[0]):
            if scores[idx] >= self.score_thresh:
                self.id_count += 1
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                obj["tracking_id"] = self.id_count
#                 obj['vxvy'] = [0.0, 0.0]
                obj['active'] = 1
                obj['age'] = 1
                ret.append(obj)
                ret_dict[idx] = obj
        
        self.tracks = ret
        self.tracks_dict = ret_dict
        return copy.deepcopy(ret)

    def init_track_sram(self, results, model_sram, mark_ration):

        scores = results["scores"]
        classes = results["labels"]
        bboxes = results["boxes"]  # x1y1x2y2
        
        ret = list()
        ret_dict = dict()
        device = torch.device("cuda")
        humans = [] # person
        marks = [] # mark
        for idx in range(scores.shape[0]):
            if scores[idx] >= self.score_thresh:
                tlbr = bboxes[idx, :].cpu().numpy().tolist()
                person_tlx = tlbr[0]
                person_tly = tlbr[1]
                person_brx = tlbr[2]
                person_bry = tlbr[3]
                person_w = person_brx - person_tlx
                person_h = person_bry - person_tly
                mark_tlx = person_tlx + mark_ration*person_w
                mark_tly = person_tly + mark_ration*person_h
                mark_brx = person_brx - mark_ration*person_w
                mark_bry = person_bry - mark_ration*person_h
                humans.append(np.array(tlbr))
                marks.append(np.array([mark_tlx, mark_tly, mark_brx, mark_bry]))
        humans = torch.tensor(humans, dtype=torch.float32).unsqueeze(0).to(device)
        marks = torch.tensor(marks, dtype=torch.float32).unsqueeze(0).to(device)
        humans_aligned, marks_aligned = model_sram(humans, marks)
        humans_aligned = humans_aligned.squeeze(0) # person
        marks_aligned = marks_aligned.squeeze(0) # mark
        det_index = 0       
        for idx in range(scores.shape[0]):
            if scores[idx] >= self.score_thresh:
                self.id_count += 1
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                obj["tracking_id"] = self.id_count
#                 obj['vxvy'] = [0.0, 0.0]
                obj['active'] = 1
                obj['age'] = 1
                obj['fuse_reid'] = humans_aligned[det_index]
                det_index += 1
                ret.append(obj)
                ret_dict[idx] = obj
        
        self.tracks = ret
        self.tracks_dict = ret_dict
        return copy.deepcopy(ret)
    
    def init_track_stram(self, results, model_stram, mark_ration):

        scores = results["scores"]
        classes = results["labels"]
        bboxes = results["boxes"]  # x1y1x2y2
        
        ret = list()
        ret_dict = dict()
        device = torch.device("cuda")
        
        humans = [] # person
        marks = [] # mark
        for idx in range(scores.shape[0]):
            if scores[idx] >= self.score_thresh:
                tlbr = bboxes[idx, :].cpu().numpy().tolist()
                person_tlx = tlbr[0]
                person_tly = tlbr[1]
                person_brx = tlbr[2]
                person_bry = tlbr[3]
                person_w = person_brx - person_tlx
                person_h = person_bry - person_tly
                mark_tlx = person_tlx + mark_ration*person_w
                mark_tly = person_tly + mark_ration*person_h
                mark_brx = person_brx - mark_ration*person_w
                mark_bry = person_bry - mark_ration*person_h
                humans.append(np.array(tlbr))
                marks.append(np.array([mark_tlx, mark_tly, mark_brx, mark_bry]))
        humans = torch.tensor(humans, dtype=torch.float32).unsqueeze(0).to(device)
        marks = torch.tensor(marks, dtype=torch.float32).unsqueeze(0).to(device)
        trajectories = humans.detach()
        
        trajectories_aligned, marks_aligned, humans_aligned = model_stram(trajectories, marks, humans)
        humans_aligned = humans_aligned.squeeze(0) # detection
        det_index = 0       
        for idx in range(scores.shape[0]):
            if scores[idx] >= self.score_thresh:
                self.id_count += 1
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                obj["tracking_id"] = self.id_count
#                 obj['vxvy'] = [0.0, 0.0]
                obj['active'] = 1
                obj['age'] = 1
                obj['fuse_reid'] = humans_aligned[det_index]
                det_index += 1
                ret.append(obj)
                ret_dict[idx] = obj
        
        self.tracks = ret
        self.tracks_dict = ret_dict
        return copy.deepcopy(ret)
    
    def step(self, output_results):
        scores = output_results["scores"]
        classes = output_results["labels"]
        bboxes = output_results["boxes"]  # x1y1x2y2
        track_bboxes = output_results["track_boxes"] if "track_boxes" in output_results else None # x1y1x2y2
        
        results = list()
        results_dict = dict()

        tracks = list()
        
        for idx in range(scores.shape[0]):
            if idx in self.tracks_dict and track_bboxes is not None:
                self.tracks_dict[idx]["bbox"] = track_bboxes[idx, :].cpu().numpy().tolist()

            if scores[idx] >= self.score_thresh:
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()               
                results.append(obj)        
                results_dict[idx] = obj
        
        tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
        N = len(results)
        M = len(tracks)
        
        ret = list()
        unmatched_tracks = [t for t in range(M)]
        unmatched_dets = [d for d in range(N)]
        if N > 0 and M > 0:
            det_box   = torch.stack([torch.tensor(obj['bbox']) for obj in results], dim=0) # N x 4        
            track_box = torch.stack([torch.tensor(obj['bbox']) for obj in tracks], dim=0) # M x 4                
            cost_bbox = 1.0 - box_ops.generalized_box_iou(det_box, track_box) # N x M

            matched_indices = linear_sum_assignment(cost_bbox)
            unmatched_dets = [d for d in range(N) if not (d in matched_indices[0])]
            unmatched_tracks = [d for d in range(M) if not (d in matched_indices[1])]

            matches = [[],[]]
            for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
                if cost_bbox[m0, m1] > 1.2:
                # if cost_bbox[m0, m1] > thresh_1:
                    unmatched_dets.append(m0)
                    unmatched_tracks.append(m1)
                else:
                    matches[0].append(m0)
                    matches[1].append(m1)

            for (m0, m1) in zip(matches[0], matches[1]):
                track = results[m0]
                track['tracking_id'] = tracks[m1]['tracking_id']
                track['age'] = 1
                track['active'] = 1
                pre_box = tracks[m1]['bbox']
                cur_box = track['bbox']
    #             pre_cx, pre_cy = (pre_box[0] + pre_box[2]) / 2, (pre_box[1] + pre_box[3]) / 2
    #             cur_cx, cur_cy = (cur_box[0] + cur_box[2]) / 2, (cur_box[1] + cur_box[3]) / 2
    #             track['vxvy'] = [cur_cx - pre_cx, cur_cy - pre_cy]
                ret.append(track)

        for i in unmatched_dets:
            track = results[i]
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] =  1
#             track['vxvy'] = [0.0, 0.0]
            ret.append(track)
        
        ret_unmatched_tracks = []
        for i in unmatched_tracks:
            track = tracks[i]
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
#                 x1, y1, x2, y2 = track['bbox']
#                 vx, vy = track['vxvy']
#                 track['bbox'] = [x1+vx, y1+vy, x2+vx, y2+vy]
                ret.append(track)
                ret_unmatched_tracks.append(track)
    
        self.tracks = ret
        self.tracks_dict = results_dict
        self.unmatched_tracks = ret_unmatched_tracks
        return copy.deepcopy(ret)
    
    def step_tram(self, output_results, model_tram_1, weight_1, thresh_1):
        scores = output_results["scores"]
        classes = output_results["labels"]
        bboxes = output_results["boxes"]  # x1y1x2y2
        track_bboxes = output_results["track_boxes"] if "track_boxes" in output_results else None # x1y1x2y2
        
        results = list()
        results_dict = dict()

        tracks = list()
        
        for idx in range(scores.shape[0]):
            if idx in self.tracks_dict and track_bboxes is not None:
                self.tracks_dict[idx]["bbox"] = track_bboxes[idx, :].cpu().numpy().tolist()

            if scores[idx] >= self.score_thresh:
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()               
                results.append(obj)        
                results_dict[idx] = obj
        
        tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
        N = len(results)
        M = len(tracks)
        
        ret = list()
        unmatched_tracks = [t for t in range(M)]
        unmatched_dets = [d for d in range(N)]
        if N > 0 and M > 0:
            det_box   = torch.stack([torch.tensor(obj['bbox']) for obj in results], dim=0) # N x 4        
            track_box = torch.stack([torch.tensor(obj['bbox']) for obj in tracks], dim=0) # M x 4                
            cost_bbox = 1.0 - box_ops.generalized_box_iou(det_box, track_box) # N x M

            device = torch.device("cuda")
            trajectories = track_box.unsqueeze(0).to(device)
            humans = det_box.unsqueeze(0).to(device)
            trajectories_aligned, humans_aligned = model_tram_1(trajectories, humans)
            trajectories_aligned = trajectories_aligned.squeeze(0) # track
            humans_aligned = humans_aligned.squeeze(0) # detection
            ious_dists = embedding_distance_1(trajectories_aligned.tolist(), humans_aligned.tolist())  # 使用pmf_fused_feature，不smooth
            cost_bbox = weight_1 * torch.tensor(ious_dists) + (1-weight_1) * cost_bbox

            matched_indices = linear_sum_assignment(cost_bbox)
            unmatched_dets = [d for d in range(N) if not (d in matched_indices[0])]
            unmatched_tracks = [d for d in range(M) if not (d in matched_indices[1])]

            matches = [[],[]]
            for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
                if cost_bbox[m0, m1] > thresh_1:
                    unmatched_dets.append(m0)
                    unmatched_tracks.append(m1)
                else:
                    matches[0].append(m0)
                    matches[1].append(m1)

            for (m0, m1) in zip(matches[0], matches[1]):
                track = results[m0]
                track['tracking_id'] = tracks[m1]['tracking_id']
                track['age'] = 1
                track['active'] = 1
                pre_box = tracks[m1]['bbox']
                cur_box = track['bbox']
    #             pre_cx, pre_cy = (pre_box[0] + pre_box[2]) / 2, (pre_box[1] + pre_box[3]) / 2
    #             cur_cx, cur_cy = (cur_box[0] + cur_box[2]) / 2, (cur_box[1] + cur_box[3]) / 2
    #             track['vxvy'] = [cur_cx - pre_cx, cur_cy - pre_cy]
                ret.append(track)

        for i in unmatched_dets:
            track = results[i]
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] =  1
#             track['vxvy'] = [0.0, 0.0]
            ret.append(track)
        
        ret_unmatched_tracks = []
        for i in unmatched_tracks:
            track = tracks[i]
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
#                 x1, y1, x2, y2 = track['bbox']
#                 vx, vy = track['vxvy']
#                 track['bbox'] = [x1+vx, y1+vy, x2+vx, y2+vy]
                ret.append(track)
                ret_unmatched_tracks.append(track)
    
        self.tracks = ret
        self.tracks_dict = results_dict
        self.unmatched_tracks = ret_unmatched_tracks
        return copy.deepcopy(ret)
    
    def step_sram(self, output_results, model_sram, weight_1, thresh_1, mark_ration):
        scores = output_results["scores"]
        classes = output_results["labels"]
        bboxes = output_results["boxes"]  # x1y1x2y2
        track_bboxes = output_results["track_boxes"] if "track_boxes" in output_results else None # x1y1x2y2
        
        results = list()
        results_dict = dict()
        device = torch.device("cuda")
        humans = [] # person
        marks = [] # mark
        for idx in range(scores.shape[0]):
            if scores[idx] >= self.score_thresh:
                tlbr = bboxes[idx, :].cpu().numpy().tolist()
                person_tlx = tlbr[0]
                person_tly = tlbr[1]
                person_brx = tlbr[2]
                person_bry = tlbr[3]
                person_w = person_brx - person_tlx
                person_h = person_bry - person_tly
                mark_tlx = person_tlx + mark_ration*person_w
                mark_tly = person_tly + mark_ration*person_h
                mark_brx = person_brx - mark_ration*person_w
                mark_bry = person_bry - mark_ration*person_h
                humans.append(np.array(tlbr))
                marks.append(np.array([mark_tlx, mark_tly, mark_brx, mark_bry]))
        humans = torch.tensor(humans, dtype=torch.float32).unsqueeze(0).to(device)
        marks = torch.tensor(marks, dtype=torch.float32).unsqueeze(0).to(device)
        humans_aligned, marks_aligned = model_sram(humans, marks)
        humans_aligned = humans_aligned.squeeze(0) # person
        marks_aligned = marks_aligned.squeeze(0) # mark
        
        tracks = list()
        
        det_index = 0
        for idx in range(scores.shape[0]):
            if idx in self.tracks_dict and track_bboxes is not None:
                self.tracks_dict[idx]["bbox"] = track_bboxes[idx, :].cpu().numpy().tolist()

            if scores[idx] >= self.score_thresh:
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                obj['fuse_reid'] = humans_aligned[det_index]  
                det_index += 1             
                results.append(obj)        
                results_dict[idx] = obj
        
        tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
        N = len(results)
        M = len(tracks)
        
        ret = list()
        unmatched_tracks = [t for t in range(M)]
        unmatched_dets = [d for d in range(N)]
        if N > 0 and M > 0:
            det_box   = torch.stack([torch.tensor(obj['bbox']) for obj in results], dim=0) # N x 4        
            track_box = torch.stack([torch.tensor(obj['bbox']) for obj in tracks], dim=0) # M x 4                
            cost_bbox = 1.0 - box_ops.generalized_box_iou(det_box, track_box) # N x M

            det_feature = [] # det_feature
            for obj in results:
                det_feature.append(obj['fuse_reid'].detach().cpu().numpy())
            track_feature = [] # track_feature
            for obj in tracks:
                track_feature.append(obj['fuse_reid'].detach().cpu().numpy())
            ious_dists = embedding_distance_1(det_feature, track_feature)
            cost_bbox = weight_1*torch.tensor(ious_dists) + (1-weight_1)*cost_bbox
            
            matched_indices = linear_sum_assignment(cost_bbox)
            unmatched_dets = [d for d in range(N) if not (d in matched_indices[0])]
            unmatched_tracks = [d for d in range(M) if not (d in matched_indices[1])]

            matches = [[],[]]
            for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
                if cost_bbox[m0, m1] > thresh_1:
                    unmatched_dets.append(m0)
                    unmatched_tracks.append(m1)
                else:
                    matches[0].append(m0)
                    matches[1].append(m1)

            for (m0, m1) in zip(matches[0], matches[1]):
                track = results[m0]
                track['tracking_id'] = tracks[m1]['tracking_id']
                track['age'] = 1
                track['active'] = 1
                pre_box = tracks[m1]['bbox']
                cur_box = track['bbox']
    #             pre_cx, pre_cy = (pre_box[0] + pre_box[2]) / 2, (pre_box[1] + pre_box[3]) / 2
    #             cur_cx, cur_cy = (cur_box[0] + cur_box[2]) / 2, (cur_box[1] + cur_box[3]) / 2
    #             track['vxvy'] = [cur_cx - pre_cx, cur_cy - pre_cy]
                ret.append(track)

        for i in unmatched_dets:
            track = results[i]
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] =  1
#             track['vxvy'] = [0.0, 0.0]
            ret.append(track)

        ret_unmatched_tracks = []
        for i in unmatched_tracks:
            track = tracks[i]
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
#                 x1, y1, x2, y2 = track['bbox']
#                 vx, vy = track['vxvy']
#                 track['bbox'] = [x1+vx, y1+vy, x2+vx, y2+vy]
                ret.append(track)
                ret_unmatched_tracks.append(track)

        self.tracks = ret
        self.tracks_dict = results_dict
        self.unmatched_tracks = ret_unmatched_tracks
        return copy.deepcopy(ret)

    def step_stram(self, output_results, model_stram, mark_ration, weight_s, weight_t, weight_st, thresh_1):
        scores = output_results["scores"]
        classes = output_results["labels"]
        bboxes = output_results["boxes"]  # x1y1x2y2
        track_bboxes = output_results["track_boxes"] if "track_boxes" in output_results else None # x1y1x2y2

        results = list()
        results_dict = dict()
        device = torch.device("cuda")
        
        humans = [] # person
        marks = [] # mark
        for idx in range(scores.shape[0]):
            if scores[idx] >= self.score_thresh:
                tlbr = bboxes[idx, :].cpu().numpy().tolist()
                person_tlx = tlbr[0]
                person_tly = tlbr[1]
                person_brx = tlbr[2]
                person_bry = tlbr[3]
                person_w = person_brx - person_tlx
                person_h = person_bry - person_tly
                mark_tlx = person_tlx + mark_ration*person_w
                mark_tly = person_tly + mark_ration*person_h
                mark_brx = person_brx - mark_ration*person_w
                mark_bry = person_bry - mark_ration*person_h
                humans.append(np.array(tlbr))
                marks.append(np.array([mark_tlx, mark_tly, mark_brx, mark_bry]))
        humans = torch.tensor(humans, dtype=torch.float32).unsqueeze(0).to(device)
        marks = torch.tensor(marks, dtype=torch.float32).unsqueeze(0).to(device)
        tracks = list()
        tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
        M = len(tracks)
        if M > 0:
            track_box = torch.stack([torch.tensor(obj['bbox']) for obj in tracks], dim=0) # M x 4
            trajectories = track_box.unsqueeze(0).to(device) 
        else:
            trajectories = humans.detach()
            
        trajectories_aligned, marks_aligned, humans_aligned = model_stram(trajectories, marks, humans)
        trajectories_aligned = trajectories_aligned.squeeze(0) # track
        humans_aligned = humans_aligned.squeeze(0) # detection
            
        det_index = 0
        for idx in range(scores.shape[0]):
            if idx in self.tracks_dict and track_bboxes is not None:
                self.tracks_dict[idx]["bbox"] = track_bboxes[idx, :].cpu().numpy().tolist()

            if scores[idx] >= self.score_thresh:
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                obj['fuse_reid'] = humans_aligned[det_index]  
                det_index += 1             
                results.append(obj)        
                results_dict[idx] = obj

        N = len(results)
        ret = list()
        unmatched_tracks = [t for t in range(M)]
        unmatched_dets = [d for d in range(N)]
        if N > 0 and M > 0:
            det_box   = torch.stack([torch.tensor(obj['bbox']) for obj in results], dim=0) # N x 4        
            track_box = torch.stack([torch.tensor(obj['bbox']) for obj in tracks], dim=0) # M x 4                
            cost_bbox = 1.0 - box_ops.generalized_box_iou(det_box, track_box) # N x M

            # Spatial
            det_feature = [] # det_feature
            for obj in results:
                det_feature.append(obj['fuse_reid'].detach().cpu().numpy())
            track_feature = [] # track_feature
            for obj in tracks:
                track_feature.append(obj['fuse_reid'].detach().cpu().numpy())
            spatial_ious_dists = embedding_distance_1(det_feature, track_feature)
            spatial_ious_dists = weight_s*torch.tensor(spatial_ious_dists) + (1-weight_s)*cost_bbox

            # Temporal
            temporal_ious_dists = embedding_distance_1(humans_aligned.tolist(), trajectories_aligned.tolist())
            temporal_ious_dists = weight_t * torch.tensor(temporal_ious_dists) + (1-weight_t) * cost_bbox

            cost_bbox = weight_st * temporal_ious_dists + (1-weight_st) * spatial_ious_dists

            matched_indices = linear_sum_assignment(cost_bbox)
            unmatched_dets = [d for d in range(N) if not (d in matched_indices[0])]
            unmatched_tracks = [d for d in range(M) if not (d in matched_indices[1])]

            matches = [[],[]]
            for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
                if cost_bbox[m0, m1] > thresh_1:
                    unmatched_dets.append(m0)
                    unmatched_tracks.append(m1)
                else:
                    matches[0].append(m0)
                    matches[1].append(m1)

            for (m0, m1) in zip(matches[0], matches[1]):
                track = results[m0]
                track['tracking_id'] = tracks[m1]['tracking_id']
                track['age'] = 1
                track['active'] = 1
                pre_box = tracks[m1]['bbox']
                cur_box = track['bbox']
    #             pre_cx, pre_cy = (pre_box[0] + pre_box[2]) / 2, (pre_box[1] + pre_box[3]) / 2
    #             cur_cx, cur_cy = (cur_box[0] + cur_box[2]) / 2, (cur_box[1] + cur_box[3]) / 2
    #             track['vxvy'] = [cur_cx - pre_cx, cur_cy - pre_cy]
                ret.append(track)

        for i in unmatched_dets:
            track = results[i]
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] =  1
#             track['vxvy'] = [0.0, 0.0]
            ret.append(track)
        
        ret_unmatched_tracks = []
        for i in unmatched_tracks:
            track = tracks[i]
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
#                 x1, y1, x2, y2 = track['bbox']
#                 vx, vy = track['vxvy']
#                 track['bbox'] = [x1+vx, y1+vy, x2+vx, y2+vy]
                ret.append(track)
                ret_unmatched_tracks.append(track)
    
        self.tracks = ret
        self.tracks_dict = results_dict
        self.unmatched_tracks = ret_unmatched_tracks
        return copy.deepcopy(ret)