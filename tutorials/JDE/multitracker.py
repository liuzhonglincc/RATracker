from numba import jit
from collections import deque
import torch
from utils.kalman_filter import KalmanFilter
from utils.log import logger
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):

    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

        self.aligned_temporal_feature = None
        self.fuse_reid = None

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat 
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha *self.smooth_feat + (1-self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        
    @staticmethod
    def multi_predict(stracks, kalman_filter):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
#            multi_mean, multi_covariance = STrack.kalman_filter.multi_predict(multi_mean, multi_covariance)
            multi_mean, multi_covariance = kalman_filter.multi_predict(multi_mean, multi_covariance)
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
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.aligned_temporal_feature = new_track.aligned_temporal_feature
        if new_track.fuse_reid is not None:
            self.fuse_reid = new_track.fuse_reid

    def update(self, new_track, frame_id, update_feature=True):
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
        if update_feature:
            self.update_features(new_track.curr_feat)
        
        self.aligned_temporal_feature = new_track.aligned_temporal_feature
        if new_track.fuse_reid is not None:
            self.fuse_reid = new_track.fuse_reid

    @property
    @jit
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
    @jit
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    @jit
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
    @jit
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    @jit
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        self.model = Darknet(opt.cfg, nID=14455)
        # load_darknet_weights(self.model, opt.weights)
        self.model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model'], strict=False)
        self.model.cuda().eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()
        self.device = torch.device("cuda")

    def update(self, im_blob, img0):
        """
        Processes the image frame and finds bounding box(detections).

        Associates the detection with corresponding tracklets and also handles lost, removed, refound and active tracklets

        Parameters
        ----------
        im_blob : torch.float32
                  Tensor of shape depending upon the size of image. By default, shape of this tensor is [1, 3, 608, 1088]

        img0 : ndarray
               ndarray of shape depending on the input image sequence. By default, shape is [608, 1080, 3]

        Returns
        -------
        output_stracks : list of Strack(instances)
                         The list contains information regarding the online_tracklets for the recieved image tensor.

        """

        self.frame_id += 1
        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        t1 = time.time()
        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            pred = self.model(im_blob)
        # pred is tensor of all the proposals (default number of proposals: 54264). Proposals have information associated with the bounding box and embeddings
        pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score
        if len(pred) > 0:
            dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].cpu()
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales
            scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
            '''Detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred)'''
            # class_pred is the embeddings.

            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f.numpy(), 30) for
                          (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])]
        else:
            detections = []

        t2 = time.time()
        # print('Forward: {} s'.format(t2-t1))

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
                # print("Should not be here, in unconfirmed")
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.kalman_filter)


        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # The dists is the list of distances of the detection with the tracks in strack_pool
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # The matches is the array for corresponding matches of the detection with the corresponding strack_pool

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # We have obtained a detection from a track which is not active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        r_tracked_stracks = [] # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        for i in u_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        # matches is the list of detections which matched with corresponding tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Same process done for some unmatched detections, but now considering IOU_distance as measure

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # If no detections are obtained for tracks (u_track), the tracks are added to lost_tracks list and are marked lost

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it is initialized for a new track
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        # print('Remained match {} s'.format(t4-t3))

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        # print('Final {} s'.format(t5-t4))
        return output_stracks


    def update_tram(self, im_blob, img0, model_tram_1, model_tram_2, model_tram_3):
        """
        Processes the image frame and finds bounding box(detections).

        Associates the detection with corresponding tracklets and also handles lost, removed, refound and active tracklets

        Parameters
        ----------
        im_blob : torch.float32
                  Tensor of shape depending upon the size of image. By default, shape of this tensor is [1, 3, 608, 1088]

        img0 : ndarray
               ndarray of shape depending on the input image sequence. By default, shape is [608, 1080, 3]

        Returns
        -------
        output_stracks : list of Strack(instances)
                         The list contains information regarding the online_tracklets for the recieved image tensor.

        """

        self.frame_id += 1
        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        t1 = time.time()
        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            pred = self.model(im_blob)
        # pred is tensor of all the proposals (default number of proposals: 54264). Proposals have information associated with the bounding box and embeddings
        pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score
        if len(pred) > 0:
            dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].cpu()
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales
            scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
            '''Detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred)'''
            # class_pred is the embeddings.

            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f.numpy(), 30) for
                          (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])]
        else:
            detections = []

        t2 = time.time()
        # print('Forward: {} s'.format(t2-t1))

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
                # print("Should not be here, in unconfirmed")
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.kalman_filter)


        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # The dists is the list of distances of the detection with the tracks in strack_pool

        if len(strack_pool) > 0 and len(detections) > 0 :
            trajectories = []
            for strack in strack_pool:
                trajectories.append(strack.tlbr)
            trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32).unsqueeze(0).to(self.device)
            humans = []
            for detection in detections:
                humans.append(detection.tlbr)
            humans = torch.tensor(np.array(humans), dtype=torch.float32).unsqueeze(0).to(self.device)
            trajectories_aligned, humans_aligned = model_tram_1(trajectories, humans)
            trajectories_aligned = trajectories_aligned.squeeze(0) # track
            humans_aligned = humans_aligned.squeeze(0) # detection
            temporal_dist = matching.embedding_distance_1(trajectories_aligned.tolist(), humans_aligned.tolist())
            dists = self.opt.weight_1*temporal_dist + (1-self.opt.weight_1)*dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.thresh_1)

        # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # The matches is the array for corresponding matches of the detection with the corresponding strack_pool

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # We have obtained a detection from a track which is not active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        r_tracked_stracks = [] # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        for i in u_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)

        if len(r_tracked_stracks) > 0 and len(detections) > 0 :
            trajectories = []
            for strack in r_tracked_stracks:
                trajectories.append(strack.tlbr)
            trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32).unsqueeze(0).to(self.device)
            humans = []
            for detection in detections:
                humans.append(detection.tlbr)
            humans = torch.tensor(np.array(humans), dtype=torch.float32).unsqueeze(0).to(self.device)
            trajectories_aligned, humans_aligned = model_tram_2(trajectories, humans)
            trajectories_aligned = trajectories_aligned.squeeze(0) # track
            humans_aligned = humans_aligned.squeeze(0) # detection
            temporal_dist = matching.embedding_distance_1(trajectories_aligned.tolist(), humans_aligned.tolist())
            dists = self.opt.weight_2*temporal_dist + (1-self.opt.weight_2)*dists
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.thresh_2)

        # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        # matches is the list of detections which matched with corresponding tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Same process done for some unmatched detections, but now considering IOU_distance as measure

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # If no detections are obtained for tracks (u_track), the tracks are added to lost_tracks list and are marked lost

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)

        if len(unconfirmed) > 0 and len(detections) > 0 :
            trajectories = []
            for strack in unconfirmed:
                trajectories.append(strack.tlbr)
            trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32).unsqueeze(0).to(self.device)
            humans = []
            for detection in detections:
                humans.append(detection.tlbr)
            humans = torch.tensor(np.array(humans), dtype=torch.float32).unsqueeze(0).to(self.device)
            trajectories_aligned, humans_aligned = model_tram_3(trajectories, humans)
            trajectories_aligned = trajectories_aligned.squeeze(0) # track
            humans_aligned = humans_aligned.squeeze(0) # detection
            temporal_dist = matching.embedding_distance_1(trajectories_aligned.tolist(), humans_aligned.tolist())
            dists = self.opt.weight_3*temporal_dist + (1-self.opt.weight_3)*dists
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.opt.thresh_3)

        # matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it is initialized for a new track
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        # print('Remained match {} s'.format(t4-t3))

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        # print('Final {} s'.format(t5-t4))
        return output_stracks

    def update_sram(self, im_blob, img0, model_sram):
        """
        Processes the image frame and finds bounding box(detections).

        Associates the detection with corresponding tracklets and also handles lost, removed, refound and active tracklets

        Parameters
        ----------
        im_blob : torch.float32
                  Tensor of shape depending upon the size of image. By default, shape of this tensor is [1, 3, 608, 1088]

        img0 : ndarray
               ndarray of shape depending on the input image sequence. By default, shape is [608, 1080, 3]

        Returns
        -------
        output_stracks : list of Strack(instances)
                         The list contains information regarding the online_tracklets for the recieved image tensor.

        """

        self.frame_id += 1
        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        t1 = time.time()
        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            pred = self.model(im_blob)
        # pred is tensor of all the proposals (default number of proposals: 54264). Proposals have information associated with the bounding box and embeddings
        pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score
        if len(pred) > 0:
            dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].cpu()
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales
            scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
            '''Detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred)'''
            # class_pred is the embeddings.

            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f.numpy(), 30) for
                          (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])]
        else:
            detections = []

        t2 = time.time()
        # print('Forward: {} s'.format(t2-t1))

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
                # print("Should not be here, in unconfirmed")
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.kalman_filter)


        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # The dists is the list of distances of the detection with the tracks in strack_pool

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
                mark_tlx = person_tlx + self.opt.mark_ration*person_w
                mark_tly = person_tly + self.opt.mark_ration*person_h
                mark_brx = person_brx - self.opt.mark_ration*person_w
                mark_bry = person_bry - self.opt.mark_ration*person_h
                humans.append(detection.tlbr)
                marks.append([mark_tlx, mark_tly, mark_brx, mark_bry])
            humans = torch.tensor(np.array(humans), dtype=torch.float32).unsqueeze(0).to(self.device)
            marks = torch.tensor(np.array(marks), dtype=torch.float32).unsqueeze(0).to(self.device)
            humans_aligned, marks_aligned = model_sram(humans, marks)
            humans_aligned = humans_aligned.squeeze(0) # person
            marks_aligned = marks_aligned.squeeze(0) # mark
            for d_i,detection in enumerate(detections):
                detection.fuse_reid = humans_aligned[d_i]
        if self.frame_id > 1:
            trajectory_aligned_feature = []
            for strack in strack_pool:
                trajectory_aligned_feature.append(strack.fuse_reid.detach().cpu().numpy())
            human_aligned_feature = []
            for detection in detections:
                human_aligned_feature.append(detection.fuse_reid.detach().cpu().numpy())
            spatial_dist = matching.embedding_distance_1(trajectory_aligned_feature, human_aligned_feature)
            dists = self.opt.weight_1*spatial_dist + (1-self.opt.weight_1)*dists
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.thresh_1)

        # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # The matches is the array for corresponding matches of the detection with the corresponding strack_pool

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # We have obtained a detection from a track which is not active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        r_tracked_stracks = [] # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        for i in u_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)

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
                mark_tlx = person_tlx + self.opt.mark_ration*person_w
                mark_tly = person_tly + self.opt.mark_ration*person_h
                mark_brx = person_brx - self.opt.mark_ration*person_w
                mark_bry = person_bry - self.opt.mark_ration*person_h
                humans.append(detection.tlbr)
                marks.append([mark_tlx, mark_tly, mark_brx, mark_bry])
            humans = torch.tensor(np.array(humans), dtype=torch.float32).unsqueeze(0).to(self.device)
            marks = torch.tensor(np.array(marks), dtype=torch.float32).unsqueeze(0).to(self.device)
            humans_aligned, marks_aligned = model_sram(humans, marks)
            humans_aligned = humans_aligned.squeeze(0) # person
            marks_aligned = marks_aligned.squeeze(0) # mark
            for d_i,detection in enumerate(detections):
                detection.fuse_reid = humans_aligned[d_i]
        if self.frame_id > 1:
            trajectory_aligned_feature = []
            for strack in r_tracked_stracks:
                trajectory_aligned_feature.append(strack.fuse_reid.detach().cpu().numpy())
            human_aligned_feature = []
            for detection in detections:
                human_aligned_feature.append(detection.fuse_reid.detach().cpu().numpy())
            spatial_dist = matching.embedding_distance_1(trajectory_aligned_feature, human_aligned_feature)
            dists = self.opt.weight_2*spatial_dist + (1-self.opt.weight_2)*dists
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.thresh_2)

        # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        # matches is the list of detections which matched with corresponding tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Same process done for some unmatched detections, but now considering IOU_distance as measure

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # If no detections are obtained for tracks (u_track), the tracks are added to lost_tracks list and are marked lost

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)

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
                mark_tlx = person_tlx + self.opt.mark_ration*person_w
                mark_tly = person_tly + self.opt.mark_ration*person_h
                mark_brx = person_brx - self.opt.mark_ration*person_w
                mark_bry = person_bry - self.opt.mark_ration*person_h
                humans.append(detection.tlbr)
                marks.append([mark_tlx, mark_tly, mark_brx, mark_bry])
            humans = torch.tensor(np.array(humans), dtype=torch.float32).unsqueeze(0).to(self.device)
            marks = torch.tensor(np.array(marks), dtype=torch.float32).unsqueeze(0).to(self.device)
            humans_aligned, marks_aligned = model_sram(humans, marks)
            humans_aligned = humans_aligned.squeeze(0) # person
            marks_aligned = marks_aligned.squeeze(0) # mark
            for d_i,detection in enumerate(detections):
                detection.fuse_reid = humans_aligned[d_i]
        if self.frame_id > 1:
            trajectory_aligned_feature = []
            for strack in unconfirmed:
                trajectory_aligned_feature.append(strack.fuse_reid.detach().cpu().numpy())
            human_aligned_feature = []
            for detection in detections:
                human_aligned_feature.append(detection.fuse_reid.detach().cpu().numpy())
            spatial_dist = matching.embedding_distance_1(trajectory_aligned_feature, human_aligned_feature)
            dists = self.opt.weight_3*spatial_dist + (1-self.opt.weight_3)*dists
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.opt.thresh_3)

        # matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it is initialized for a new track
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        # print('Remained match {} s'.format(t4-t3))

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        # print('Final {} s'.format(t5-t4))
        return output_stracks

    def update_stram(self, im_blob, img0, model_stram):
        """
        Processes the image frame and finds bounding box(detections).

        Associates the detection with corresponding tracklets and also handles lost, removed, refound and active tracklets

        Parameters
        ----------
        im_blob : torch.float32
                  Tensor of shape depending upon the size of image. By default, shape of this tensor is [1, 3, 608, 1088]

        img0 : ndarray
               ndarray of shape depending on the input image sequence. By default, shape is [608, 1080, 3]

        Returns
        -------
        output_stracks : list of Strack(instances)
                         The list contains information regarding the online_tracklets for the recieved image tensor.

        """

        self.frame_id += 1
        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        t1 = time.time()
        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            # pt_path = f"/media/disk2/lzl/iccv_abalation/Towards-Realtime-MOT-master/save_img_to_model_result/{seq}/{self.frame_id}.pt"
            # if os.path.exists(pt_path):
            #     pred = torch.load(pt_path)
            # else:
            #     if not os.path.exists(f"/media/disk2/lzl/iccv_abalation/Towards-Realtime-MOT-master/save_img_to_model_result/{seq}"):
            #         os.mkdir(f"/media/disk2/lzl/iccv_abalation/Towards-Realtime-MOT-master/save_img_to_model_result/{seq}")
            #     pred = self.model(im_blob)
            #     torch.save(pred, pt_path)
            pred = self.model(im_blob)
                
        # pred is tensor of all the proposals (default number of proposals: 54264). Proposals have information associated with the bounding box and embeddings
        pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score
        if len(pred) > 0:
            dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].cpu()
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales
            scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
            '''Detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred)'''
            # class_pred is the embeddings.

            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f.numpy(), 30) for
                          (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])]
        else:
            detections = []

        t2 = time.time()
        # print('Forward: {} s'.format(t2-t1))

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
                # print("Should not be here, in unconfirmed")
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.kalman_filter)


        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # The dists is the list of distances of the detection with the tracks in strack_pool
        temporal_bool = False
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
                mark_tlx = person_tlx + self.opt.mark_ration*person_w
                mark_tly = person_tly + self.opt.mark_ration*person_h
                mark_brx = person_brx - self.opt.mark_ration*person_w
                mark_bry = person_bry - self.opt.mark_ration*person_h
                humans.append(detection.tlbr)
                marks.append([mark_tlx, mark_tly, mark_brx, mark_bry])
            humans = torch.tensor(np.array(humans), dtype=torch.float32).unsqueeze(0).to(self.device)
            marks = torch.tensor(np.array(marks), dtype=torch.float32).unsqueeze(0).to(self.device)
            if len(strack_pool) > 0:
                trajectories = []
                for strack in strack_pool:
                    trajectories.append(strack.tlbr)
                trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                trajectories = humans.detach()

            trajectories_aligned, marks_aligned, humans_aligned = model_stram(trajectories, marks, humans)
            trajectories_aligned = trajectories_aligned.squeeze(0) # track
            humans_aligned = humans_aligned.squeeze(0) # detection
            for d_i,detection in enumerate(detections):
                detection.fuse_reid = humans_aligned[d_i]
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
                trajectory_aligned_feature.append(strack.fuse_reid.detach().cpu().numpy())
            human_aligned_feature = []
            for detection in detections:
                human_aligned_feature.append(detection.fuse_reid.detach().cpu().numpy())
            spatial_dist = matching.embedding_distance_1(trajectory_aligned_feature, human_aligned_feature)
            spatial_dist = self.args.weight_s*spatial_dist + (1-self.args.weight_s)*dists
            dists_fused = self.args.weight_st*temporal_dist + (1-self.args.weight_st)*spatial_dist
            dists = dists_fused
        matches, u_track, u_detection = matching.linear_assignment(dists_fused, thresh=self.opt.thresh_1)
        # The matches is the array for corresponding matches of the detection with the corresponding strack_pool

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # We have obtained a detection from a track which is not active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        r_tracked_stracks = [] # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        for i in u_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.thresh_2)
        # matches is the list of detections which matched with corresponding tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Same process done for some unmatched detections, but now considering IOU_distance as measure

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # If no detections are obtained for tracks (u_track), the tracks are added to lost_tracks list and are marked lost

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.opt.thresh_3)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it is initialized for a new track
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        # print('Remained match {} s'.format(t4-t3))

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        # print('Final {} s'.format(t5-t4))
        return output_stracks

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
    pairs = np.where(pdist<0.15)
    dupa, dupb = list(), list()
    for p,q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i,t in enumerate(stracksa) if not i in dupa]
    resb = [t for i,t in enumerate(stracksb) if not i in dupb]
    return resa, resb
            

