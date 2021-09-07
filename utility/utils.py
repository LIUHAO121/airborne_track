import json
import os
import numpy as np
from yaml import safe_load
import json

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_for_each_fid(root):
    fids = os.listdir(root)
    for fid in fids:
        fid_result_file = os.path.join(root,fid,"result.json")
        fid_result = load_json(fid_result_file)
        re_result = re_construct_out(fid_result)
        img_names = list(re_result.keys())
        fid_nms = []
        for img_name in img_names:
            
            detections = re_result[img_name]
            nms_input = []
            for det in detections:
                obj = [det['x'],det['y'],det['w'],det['h'],det['s']]
                nms_input.append(obj)
            nms_result = py_cpu_nms(np.array(nms_input),0.3)
            nms_dets = [detections[i] for i in nms_result]
            for nms_det in nms_dets:
                temp = {}
                temp['img_name'] = img_name
                temp['detections'] = [nms_det]
                fid_nms.append(temp)
        nms_fid_result_file = fid_result_file[:-5]+"_nms.json"
        dump_json(nms_fid_result_file,fid_nms)
        print(nms_fid_result_file)


def load_json(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    return data


def dump_json(json_path, data):
    with open(json_path, "w") as json_file:
        json.dump(data, json_file)


class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)

def read_yaml(yaml_fn):
    with open(yaml_fn, "r") as f:
        data = safe_load(f)
    return data

def replace_result_jpg2png(file):
    new = []
    datas = json.load(open(file,'r'))
    for data in datas:
        data["img_name"].replace("jpg","png")
        new.append(data)
    json.dump(new,open(file[:-5]+"2.json","w"))


def merge_forward_result(dir):
    fids = os.listdir(dir)
    outputs = []
    for fid in fids:
        fid_dir = os.path.join(dir,fid)
        result_file = os.path.join(fid_dir,"result.json")
        output = load_json(result_file)
        outputs.extend(output)
    output_file = os.path.join(dir,"result.json")
    dump_json(output_file,outputs)

def merge_val_forward_result(dir):
    fids = os.listdir(dir)
    # val_fids = load_json('data/airmot/val_flight_ids.json')
    outputs = []
    for fid in fids:
        # if fid in fids:
        fid_dir = os.path.join(dir,fid)
        result_file = os.path.join(fid_dir,"result.json")
        output = load_json(result_file)
        outputs.extend(output)
    output_file = os.path.join(dir,"result.json")
    dump_json(output_file,outputs)

def change_result_xyxy2xywh(file):
    new_result = []
    results = load_json(file)
    for res in results:
        res['detections'][0]['w'] = res['detections'][0]['w'] - res['detections'][0]['x']
        res['detections'][0]['h'] = res['detections'][0]['h'] - res['detections'][0]['y']
        new_result.append(res)
    json.dump(new_result,open(file[:-5]+"2.json","w"))

def filter_result_by_wh(file):
    new_result = []
    results = load_json(file)
    for res in results:
        if res['detections'][0]['w'] > 0 and res['detections'][0]['w'] > 0:
            new_result.append(res)
    json.dump(new_result,open(file[:-5]+"filter_size0.json","w"))

def change_result_xywh2xyxy(file):
    new_result = []
    results = load_json(file)
    for res in results:
        res['detections'][0]['w'] = max(0,res['detections'][0]['w'] - res['detections'][0]['x'])
        res['detections'][0]['h'] = max(0,res['detections'][0]['h'] - res['detections'][0]['y'])
        new_result.append(res)
    json.dump(new_result,open(file,"w"))

def change_result_xywh2cxcywh(file):
    new_result = []
    results = load_json(file)
    for res in results:
        res['detections'][0]['x'] = max(0,res['detections'][0]['x'] + res['detections'][0]['w']/2)
        res['detections'][0]['y'] = max(0,res['detections'][0]['y'] + res['detections'][0]['h']/2)
        new_result.append(res)
    json.dump(new_result,open(file[:-5]+"2.json","w"))

def change_result_xyxy2cxcywh(file):
    new_result = []
    results = load_json(file)
    for res in results:
        res['detections'][0]['w'] = max(0,res['detections'][0]['w'] - res['detections'][0]['x'])
        res['detections'][0]['h'] = max(0,res['detections'][0]['h'] - res['detections'][0]['y'])
        res['detections'][0]['x'] = max(0,res['detections'][0]['x'] + res['detections'][0]['w']/2)
        res['detections'][0]['y'] = max(0,res['detections'][0]['y'] + res['detections'][0]['h']/2)
        new_result.append(res)
    json.dump(new_result,open(file,"w"))

def expand_wh(file,ratio):
    new_result = []
    results = load_json(file)
    for res in results:
        res['detections'][0]['w'] = res['detections'][0]['w'] * ratio
        res['detections'][0]['h'] = res['detections'][0]['h'] * ratio
        new_result.append(res)
    json.dump(new_result,open(file,"w"))


def check_val(valdir,outdir):
    val_fids = os.listdir(valdir)
    out_fids = os.listdir(outdir)
    for val_id in val_fids:
        if val_id not in out_fids:
            print(val_id)


def mean_score_above_05(file):
    count = 0
    accumulate_score = 0
    results = load_json(file)
    for res in results:
        detections = res['detections']
        for det in detections:
            if det['s']>0.5:
                count+=1
                accumulate_score += det['s']
    print("the number of score > 0.5 is {}".format(count))
    print("the average of all scores is {}".format(accumulate_score/count))


def re_construct_out(outputs):

    new_result = {}
    for output in outputs:
        img_name = output['img_name']
        detections = output['detections']
        if img_name not in new_result:
            new_result[img_name] = []
        new_result[img_name].extend(detections)
    return new_result


def filter_result(result_file):
    all_result = load_json(result_file)
    filter_res = []
    for obj in all_result:
        if obj['detections'][0]['w'] * obj['detections'][0]['h'] > 1500 or obj['detections'][0]['s']>0.5:#or (obj['detections'][0]['w'] * obj['detections'][0]['h']<00 and obj['detections'][0]['s']>0.3)
            filter_res.append(obj)
    dump_json(result_file,filter_res)

            

def see_valgt_img_names(file="data/airmot/annotation/gt/val/sub2/groundtruth.json"):
    gt = load_json(file)
    samples = gt["samples"]
    count=0
    fids = list(samples.keys())
    val_fids = os.listdir('data/airmot/airmot_png/val/sub2')
    img_names = []
    for fid in val_fids:
        for entrty in samples[fid[5:]]['entities']:
            if "bb" in entrty and entrty['img_name'] not in img_names:
                if "range_distance_m" in entrty['blob']:
                    if entrty['blob']["range_distance_m"] is not None:
                        img_names.append(entrty['img_name'])
    print(len(img_names))


def max_score_row(detections):
    if len(detections) == 0:
        return detections
    detections = np.array(detections) #[x,y,x,y,s]

    max_row = int(np.where(detections[:,-1]==np.max(detections[:,-1]))[0])
    return np.array([detections[max_row,:]])
    
def max_score_and_size(detections):
    if len(detections) == 0:
        return detections
    detections = np.array(detections)
    max_score_index = int(np.where(detections[:,-1]==np.max(detections[:,-1]))[0])
    areas = [abs(d[2]-d[0])*abs(d[3]-d[1]) for d in detections]
    max_area_index = areas.index(max(areas))
    if max_score_index==max_area_index:
        return np.array([detections[max_score_index,:]])
    else:
        return []

def MaxScoreMultiSize(detections):
    if len(detections) == 0:
        return detections
    # area_multi_scores = [abs(d[2]-d[0]) * abs(d[3]-d[1] * d[-1]) for d in detections]
    area_multi_scores = [(abs(d[2]-d[0])*0.5 + abs(d[3]-d[1])*0.5)*d[-1] for d in detections]
    max_index = area_multi_scores.index(max(area_multi_scores))
    detections = np.array(detections)
    return np.array([detections[max_index,:]])


def more_one_track_id(eval_result_dir):
    fid_result = []

    for fid in os.listdir(eval_result_dir):
        id_list = []
        fid_res_path = os.path.join(eval_result_dir,fid,"result.json")
        if os.path.exists(fid_res_path):
            track_result = load_json(fid_res_path)
            for frame_res in track_result:
                detections = frame_res["detections"]
                for det in detections:
                    id_list.append(det['track_id'])
            if len(set(id_list))>1:
                fid_result.append(fid)
    return fid_result



if __name__ == "__main__":
    # replace_result_jpg2png("/workspace/airborne-detection-starter-kit/data/results/run0/result.json")
    # filter_result_by_wh('data/results/centernet_e10_1024_png/result2.json')

    # merge
    # merge_forward_result("/workspace/airborne-detection-starter-kit/data/results/run0")
    
    # change_result_xywh2xyxy("/workspace/airborne-detection-starter-kit/data/results/centernet_e10_1024_png/result.json")
    # change_result_xyxy2xywh('/workspace/airborne-detection-starter-kit/data/results/centernet_e10_1024_png/result2filter_size0.json')
    # change_result_xywh2cxcywh("/workspace/airborne-detection-starter-kit/data/results/run0/result2.json")
    
    # check_val('data/val','data/results/run0')

    # nms_for_each_fid("data/results/run0/")
    # see_valgt_img_names("data/airmot/annotation/gt/val/sub2/groundtruth.json")
    # filter_result("data/results/run0/result.json")
    # mean_score_above_05('data/results/run0/result.json')

    # merge_val_forward_result("data/results/run0")
    # expand_wh("data/results/tbd_jpg_res18_e13_2048/result.json",1.5)

    change_result_xyxy2cxcywh("data/results/run0/result.json")
    # print(more_one_track_id('data/results/res18_e35_2048_fr3sd2_ml20_tl5_s0.5_ts0.5_maxone'))
    
