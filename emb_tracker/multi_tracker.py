from enum import Enum

import numpy as np

from .utils import embedding_distance, iou_distance, linear_assignment, centerdist


class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack(object):
    def __init__(self, tlwh, score, temp_feat, feat_alpha):
        self._count = 0  # used
        self.track_id = 0  # used
        self.state = TrackState.New.value
        self.start_frame = 0  # used
        self.frame_id = 0  # used
        self.alpha = feat_alpha
        # 所有tracklet score 来自 detect result 因此为1.0
        self.score = score   
        self.det_tlwh = np.asarray(tlwh, dtype=np.int)
            # wait activate
        self.is_activated = False
        self.tracklet_len = 0
        self.smooth_feat = None
        self.update_features(temp_feat)
        
       
    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        # det总会首先进行track的初始化，curr_feat是经过二范数归一化后的结果
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        # self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def activate(self, frame_id, _count):
        """Start a new tracklet"""
        self.track_id = _count
        self.tracklet_len = 0
        self.state = TrackState.Tracked.value
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, _count):
        # 每一帧都根据det更新det_tlwh
        self.det_tlwh = new_track.det_tlwh
        self.update_features(new_track.curr_feat)
        self.tracklet_len += 1
        self.state = TrackState.Tracked.value
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score

     

    def update(self, new_track, frame_id, update_feature=True):

        self.frame_id = frame_id
        self.tracklet_len += 1
        # 每一帧都根据det更新det_tlwh
        self.det_tlwh = new_track.det_tlwh
        self.state = TrackState.Tracked.value
        self.is_activated = True
        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property

    def tlbr(self):

        ret = self.det_tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
   
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.det_tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    def mark_lost(self):
        self.state = TrackState.Lost.value

    def mark_removed(self):
        self.state = TrackState.Removed.value

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


class EmbeddingTracker(object):
    def __init__(self, conf_thres=0.5, feat_alpha=0.05, frame_rate=10, remove_dist=0.15, linear_assignment_dist=0.8,spatial_dist=5):
        self.det_thresh = conf_thres
        self.max_time_lost = int(frame_rate)
        self.feat_alpha = feat_alpha
        self.remove_dist = remove_dist
        self.linear_assignment_dist = linear_assignment_dist
        self.spatial_dist = spatial_dist
        self.init_history_state()

    def init_history_state(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.count = 0
       

    def update_history_state(self, tem_refind_stracks, tem_activated_stracks, tem_lost_stracks, tem_removed_stracks):
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame >= self.max_time_lost:
                track.mark_removed()
                tem_removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked.value]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, tem_refind_stracks + tem_activated_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(tem_lost_stracks)
        self.removed_stracks.extend(tem_removed_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

    def remove_duplicate_stracks(self, stracksa, stracksb):
        pdist = iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < self.remove_dist)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb

    def next_count(self, new_id):
        if new_id:
            self.count += 1
        return self.count

    def update(self, all_det_data):
        self.frame_id += 1
        # 每一次matching的的所有临时状态：
        # 四个track状态 一个检测状态
        tem_activated_stracks, tem_refind_stracks, tem_lost_stracks, tem_removed_stracks, tem_detections = [], [], [], [], []
        for det_data in all_det_data:
            tem_detections.append(STrack(det_data["bbox_tlwh"], np.asarray(det_data["conf"]), np.asarray(det_data["emb"]), self.feat_alpha))

        unconfirmed_stracks = []
        confirmed_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed_stracks.append(track)
            else:
                confirmed_stracks.append(track)

        """ Step 2: First association, only with embedding"""
        strack_pool = joint_stracks(confirmed_stracks, self.lost_stracks)
        # 所以检测到的目标和待匹配目标计算距离矩阵
        dists = embedding_distance(strack_pool, tem_detections)
        dists = centerdist(dists,strack_pool,tem_detections,dist_thresh=self.spatial_dist)
        # 线性匹配,距离的阈值上限,大于阈值后置为inf,默认inf
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.linear_assignment_dist)
        # 匹配到的track和det的基础上：
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = tem_detections[idet]
            # 如果被匹配到的track本身是Tracked的，也就是之前一直是一个轨迹中的
            # 该track被检测更新，然后加入tem_activated_stracks
            if track.state == TrackState.Tracked.value:
                track.update(tem_detections[idet], self.frame_id)
                tem_activated_stracks.append(track)
            # 如果被匹配到的track本身不是Tracked的，也就lost_stracks，暂时丢失的轨迹
            # 该轨迹被重新激活，并添加到tem_refind_stracks
            else:
                track.re_activate(det, self.frame_id, self.next_count(new_id=False))
                tem_refind_stracks.append(track)

        # 没有匹配到的track
        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost.value:
                track.mark_lost()
                tem_lost_stracks.append(track)
        # 没有匹配到的det
        tem_detections = [tem_detections[i] for i in u_detection]

        """Deal with unconfirmed_stracks, usually tracks with only one beginning frame"""
        # 处理unconfirmed_stracks（3），剩余的检测
        # 修改为embbeding距离
        dists = embedding_distance(unconfirmed_stracks, tem_detections)
        dists = centerdist(dists,unconfirmed_stracks,tem_detections,dist_thresh=self.spatial_dist)

        # 距离的阈值上限 ,大于阈值后置为,inf默认inf
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=self.linear_assignment_dist)
        # unconfirmed_stracks如果匹配到，添加到 tem_activated_stracks
        for itracked, idet in matches:
            unconfirmed_stracks[itracked].update(tem_detections[idet], self.frame_id)
            tem_activated_stracks.append(unconfirmed_stracks[itracked])
        for it in u_unconfirmed:
            track = unconfirmed_stracks[it]
            track.mark_removed()
            tem_removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = tem_detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id, self.next_count(new_id=True))
            tem_activated_stracks.append(track)

        """ Step 5: Update state"""
        self.update_history_state(tem_refind_stracks, tem_activated_stracks, tem_lost_stracks, tem_removed_stracks)
        return [track for track in self.tracked_stracks if track.is_activated]
       



def joint_stracks(tlista, tlistb):
    exists = [t.track_id for t in tlista]
    res = tlista + [t for t in tlistb if t.track_id not in exists]
    return res


def sub_stracks(tlista, tlistb):
    to_be_removed = [t.track_id for t in tlistb]
    res = [t for t in tlista if t.track_id not in to_be_removed]
    return res