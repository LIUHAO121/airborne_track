import lap
import numpy as np
import os
from typing import Dict
from yaml import safe_load

def read_yaml(yaml_fn):
    with open(yaml_fn, "r") as f:
        data = safe_load(f)
    return data

def restructDet(mot_det_file):
    with open(mot_det_file,"r") as f:
        det_lines = f.readlines()

    valid_redict = {}
    for line in det_lines:
        linelist = line.split(",")
        if len(linelist) == 10:
            if linelist[0] == '':
                continue
            tlwh = tuple(map(float, linelist[2:6]))
            track_id, frame_id, confidence = int(linelist[1]), int(linelist[0]), float(linelist[6])
            valid_redict.setdefault(frame_id, list())
            if confidence != 0:
                valid_redict[frame_id].append([tlwh, track_id, confidence])
    return valid_redict

def write_results(filename, results_dict: Dict, data_type: str):
    if not filename:
        return
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    if data_type in ('mot', 'mcmot', 'lab'):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian -1 -1 -10 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 {score}\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, frame_data in results_dict.items():
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in frame_data:
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, score=1.0)
                f.write(line)


def row_norms(x):
    norms = np.einsum("ij,ij->i", x, x)
    return np.sqrt(norms, out=norms)


def convert_to_double(x):
    if x.dtype != np.double:
        x = x.astype(np.double)
    if not x.flags.contiguous:
        x = x.copy()
    return x


def cdist(xa, xb):
    dm = np.zeros((xa.shape[0], xb.shape[0]), dtype=np.double)
    xa = np.asarray(xa, order="c")
    xb = np.asarray(xb, order="c")
    xa = convert_to_double(xa)
    xb = convert_to_double(xb)
    norm_a = row_norms(xa)
    norm_b = row_norms(xb)
    np.dot(xa, xb.T, out=dm)
    dm /= norm_a.reshape(-1, 1)
    dm /= norm_b
    dm *= -1
    dm += 1
    return dm


def bbox_ious(bboxes1, bboxes2):
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(y_end - y_start + 1, 0)
        union = area1[i] + area2 - overlap
        ious[i, :] = overlap / union
    return ious


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious
    ious = bbox_ious(np.ascontiguousarray(atlbrs, dtype=np.float), np.ascontiguousarray(btlbrs, dtype=np.float))
    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) and (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features))  # Nomalized features
    return cost_matrix

def centerdist(dist, tracks, detections, dist_thresh):
    """
    :param: cost_matrix np.ndarray
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param dist_thresh: dist > dist_thresh*width = inf
    """
    if len(tracks) == 0 or len(detections) == 0:
        return dist
    track_loc = []
    track_w = []
    for track in tracks:
        tlwh = track.det_tlwh
        w = tlwh[2]
        h = tlwh[3]
        cx = tlwh[0] + w/2
        cy = tlwh[1] + h/2
        track_loc.append([cx, cy]) 
        track_w.append(max(w,h)) # select a big value as base 
    track_loc = np.array(track_loc)
    track_w = np.array(track_w)
    
    det_loc = []
    for det in detections:
        tlwh = det.det_tlwh
        w = tlwh[2]
        h = tlwh[3]
        cx = tlwh[0] + w/2
        cy = tlwh[1] + h/2
        det_loc.append([cx, cy])
    det_loc = np.array(det_loc)

    dist_ = (((track_loc.reshape(1, -1, 2) - \
              det_loc.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M
    dist_ = np.sqrt(dist_)

    invalid = dist_ > dist_thresh * track_w
    dist = dist + invalid.T * 1e8

    return dist


def auto_dist(dist, tracks, detections, dist_thresh,frame_id):
    if len(tracks) == 0 or len(detections) == 0:
        return dist
    track_loc = []
    track_w = []
    for track in tracks:
        tlwh = track.det_tlwh
        w = tlwh[2]
        h = tlwh[3]
        cx = tlwh[0] + w/2
        cy = tlwh[1] + h/2
        track_loc.append([cx, cy]) 
        track_w.append(max(w,h)) # select a big value as base 
    track_loc = np.array(track_loc)
    track_w = np.array(track_w)
    
    det_loc = []
    for det in detections:
        tlwh = det.det_tlwh
        w = tlwh[2]
        h = tlwh[3]
        cx = tlwh[0] + w/2
        cy = tlwh[1] + h/2
        det_loc.append([cx, cy])
    det_loc = np.array(det_loc)

    auto_dist = np.array([min(int(frame_id - track.frame_id),2) * 0.5 + dist_thresh  for track in tracks])
    auto_dist = auto_dist.reshape((1,-1))
    auto_dist = np.repeat(auto_dist,len(det_loc),0)


    dist_ = (((track_loc.reshape(1, -1, 2) - \
              det_loc.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M
    dist_ = np.sqrt(dist_)

    invalid = dist_ > auto_dist * track_w
    dist = dist + invalid.T * 1e8

    return dist


def area_dist(dist, tracks, detections, dist_thresh):
    """
    :param: cost_matrix np.ndarray
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param dist_thresh: dist > dist_thresh*width = inf
    """
    if len(tracks) == 0 or len(detections) == 0:
        return dist
    track_loc = []
    track_areas = []
    for track in tracks:
        tlwh = track.det_tlwh
        w = tlwh[2]
        h = tlwh[3]
        cx = tlwh[0] + w/2
        cy = tlwh[1] + h/2
        track_loc.append([cx, cy]) 
        track_areas.append(w*h) # select a big value as base 
    track_areas = np.array(track_areas)
    
    det_areas = []
    for det in detections:
        tlwh = det.det_tlwh
        w = tlwh[2]
        h = tlwh[3]
        det_areas.append(w*h)
    det_areas = np.array(det_areas)

    area_ratio_matrix = cal_area_ratio_matrix(track_areas,det_areas)


    invalid = area_ratio_matrix > dist_thresh
    dist = dist + invalid * 1e8

    return dist

def cal_area_ratio_matrix(array_a,array_b):
    na=len(array_a)
    nb=len(array_b)
    ratio_matrix = np.zeros((na,nb))
    if ratio_matrix.size==0:
        return ratio_matrix
    array_a = array_a.reshape((-1,1))
    array_b = array_b.reshape((1,-1))
    repeat_a = np.repeat(array_a,nb,axis=1)
    repeat_b = np.repeat(array_b,na,axis=0)
    max_area_matrix = np.maximum(repeat_a,repeat_b)
    min_area_matrix = np.minimum(repeat_a,repeat_b)
    ratio_matrix=max_area_matrix/min_area_matrix
    return ratio_matrix