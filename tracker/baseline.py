import numpy as np
from .kalman_filter import KalmanFilter
from .basetrack import BaseTrack, TrackState
import sys
import torch
import numpy
sys.path.append(".")
from tracker import matching
from tools.models.tram import build as build_model_t
from tools.models.sram import build as build_model_s
from tools.models.stram import build as build_model_st
from tools.utils.tool import load_model

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.aligned_feature_sram = None # storage aligned feature for sram

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        if new_track.aligned_feature_sram is not None:
            self.aligned_feature_sram = new_track.aligned_feature_sram
            
    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if new_track.aligned_feature_sram is not None:
            self.aligned_feature_sram = new_track.aligned_feature_sram
            
    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class Baseline(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.det_thresh
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.device = torch.device("cuda")
        
        if args.tram:
            self.model = build_model_t(args)
            self.args.weight_t = 0.03
        elif args.sram:
            self.model = build_model_s(args)
            self.args.match_thresh = 0.71
            self.args.weight_s = 0.1
        elif args.stram:
            self.model = build_model_st(args)
            self.args.match_thresh = 0.75
            self.args.weight_t = 0.05
            self.args.weight_s = 0.13
            self.args.weight_st = 0.54
        else:
            self.model = None
        if self.model is not None:
            self.model.to(self.device)
            self.model = load_model(self.model, args.pretrained)
            self.model.eval()
        
    def update(self, output_results):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        dets = bboxes
        scores_keep = scores

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(tlwh, s) for(tlwh, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if self.args.tram:
            if len(strack_pool) > 0 and len(detections) > 0 :
                trajectories = []
                for strack in strack_pool:
                    trajectories.append(strack.tlbr)
                trajectories = torch.tensor(numpy.array(trajectories), dtype=torch.float32).unsqueeze(0).to(self.device)
                humans = []
                for detection in detections:
                    humans.append(detection.tlbr)
                humans = torch.tensor(numpy.array(humans), dtype=torch.float32).unsqueeze(0).to(self.device)
                trajectories_aligned, humans_aligned = self.model(trajectories, humans)
                trajectories_aligned = trajectories_aligned.squeeze(0) # track
                humans_aligned = humans_aligned.squeeze(0) # detection
                temporal_dist = matching.embedding_distance_1(trajectories_aligned.tolist(), humans_aligned.tolist())
                dists = self.args.weight_t*temporal_dist + (1-self.args.weight_t)*dists
        elif self.args.sram:
            if len(detections) > 0 :
                humans = [] # person
                marks = [] # mark
                for detection in detections:
                    person_tlx = detection.tlbr[0]
                    person_tly = detection.tlbr[1]
                    person_brx = detection.tlbr[2]
                    person_bry = detection.tlbr[3]
                    person_w = person_brx - person_tlx
                    person_h = person_bry - person_tly
                    mark_tlx = person_tlx + self.args.mark_ration*person_w
                    mark_tly = person_tly + self.args.mark_ration*person_h
                    mark_brx = person_brx - self.args.mark_ration*person_w
                    mark_bry = person_bry - self.args.mark_ration*person_h
                    humans.append(detection.tlbr)
                    marks.append([mark_tlx, mark_tly, mark_brx, mark_bry])
                humans = torch.tensor(numpy.array(humans), dtype=torch.float32).unsqueeze(0).to(self.device)
                marks = torch.tensor(numpy.array(marks), dtype=torch.float32).unsqueeze(0).to(self.device)
                humans_aligned, marks_aligned = self.model(humans, marks)
                humans_aligned = humans_aligned.squeeze(0) # person
                marks_aligned = marks_aligned.squeeze(0) # mark
                for d_i,detection in enumerate(detections):
                    detection.aligned_feature_sram = humans_aligned[d_i]
            if self.frame_id > 1:
                trajectory_aligned_feature = []
                for strack in strack_pool:
                    trajectory_aligned_feature.append(strack.aligned_feature_sram.detach().cpu().numpy())
                human_aligned_feature = []
                for detection in detections:
                    human_aligned_feature.append(detection.aligned_feature_sram.detach().cpu().numpy())
                spatial_dist = matching.embedding_distance_1(trajectory_aligned_feature, human_aligned_feature)
                dists = self.args.weight_s*spatial_dist + (1-self.args.weight_s)*dists
        elif self.args.stram:
            if len(detections) > 0 :
                humans = [] # person
                marks = [] # mark
                for detection in detections:
                    person_tlx = detection.tlbr[0]
                    person_tly = detection.tlbr[1]
                    person_brx = detection.tlbr[2]
                    person_bry = detection.tlbr[3]
                    person_w = person_brx - person_tlx
                    person_h = person_bry - person_tly
                    mark_tlx = person_tlx + self.args.mark_ration*person_w
                    mark_tly = person_tly + self.args.mark_ration*person_h
                    mark_brx = person_brx - self.args.mark_ration*person_w
                    mark_bry = person_bry - self.args.mark_ration*person_h
                    humans.append(detection.tlbr)
                    marks.append([mark_tlx, mark_tly, mark_brx, mark_bry])
                humans = torch.tensor(numpy.array(humans), dtype=torch.float32).unsqueeze(0).to(self.device)
                marks = torch.tensor(numpy.array(marks), dtype=torch.float32).unsqueeze(0).to(self.device)
                
                if len(strack_pool) > 0:
                    trajectories = []
                    for strack in strack_pool:
                        trajectories.append(strack.tlbr)
                    trajectories = torch.tensor(numpy.array(trajectories), dtype=torch.float32).unsqueeze(0).to(self.device)
                else:
                    trajectories = humans.detach()

                trajectories_aligned, marks_aligned, humans_aligned = self.model(trajectories, marks, humans)
                trajectories_aligned = trajectories_aligned.squeeze(0) # track
                humans_aligned = humans_aligned.squeeze(0) # detection
                for d_i,detection in enumerate(detections):
                    detection.aligned_feature_sram = humans_aligned[d_i]
                if len(strack_pool) > 0:
                    temporal_dist = matching.embedding_distance_1(trajectories_aligned.tolist(), humans_aligned.tolist())
                    temporal_dist = self.args.weight_t*temporal_dist + (1-self.args.weight_t)*dists
                else:
                    temporal_dist = dists
            if self.frame_id == 1:
                dists_fused = dists
            elif self.frame_id == 2:
                dists_fused = temporal_dist
            else:
                trajectory_aligned_feature = []
                for strack in strack_pool:
                    trajectory_aligned_feature.append(strack.aligned_feature_sram.detach().cpu().numpy())
                human_aligned_feature = []
                for detection in detections:
                    human_aligned_feature.append(detection.aligned_feature_sram.detach().cpu().numpy())
                spatial_dist = matching.embedding_distance_1(trajectory_aligned_feature, human_aligned_feature)
                spatial_dist = self.args.weight_s*spatial_dist + (1-self.args.weight_s)*dists
                dists_fused = self.args.weight_st*temporal_dist + (1-self.args.weight_st)*spatial_dist
                dists = dists_fused

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh_2)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks