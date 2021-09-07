import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import shutil

class_set = ['Airborne',"Airplane",'Helicopter','Drone','Flock','Bird']


def load_json(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    return data


def dump_json(json_path, data):
    with open(json_path, "w") as json_file:
        json.dump(data, json_file)


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def draw_bbox(img,detections,format="xywh"):
    im = np.ascontiguousarray(np.copy(img))
    # im = img
    for entry in detections:
        id = 1
        color = get_color(id)
        x,y,w,h = int(entry[0]),int(entry[1]),int(entry[2]),int(entry[3])
        s = entry[4]
        if format=="xywh":
            x1,y1,x2,y2 = x,y,x+w,y+h
        elif format=="xyxy":
            x1,y1,x2,y2 = x,y,w,h
        elif format== "cxcywh":
            x1,y1,x2,y2 = x-w/2, y-h/2, x+w/2, y+h/2
        else:
            raise "unrecognise bbox format"
        cv2.rectangle(im,pt1=(int(x1),int(y1)),pt2=(int(x2),int(y2)),color=color,thickness=2)
        cv2.putText(im, str(id)+"_"+str(s)[:5], (int(x1),int(y1-1)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
    return im


def draw_track_bbox(img,detections,format="xywh"):
    im = np.ascontiguousarray(np.copy(img))
    # im = img
    for entry in detections:
        id = entry["track_id"]
        color = get_color(id)
        x,y,w,h = int(entry["x"]),int(entry["y"]),int(entry["w"]),int(entry["h"])
        s = entry["s"]
        if format=="xywh":
            x1,y1,x2,y2 = x,y,x+w,y+h
        elif format=="xyxy":
            x1,y1,x2,y2 = x,y,w,h
        elif format== "cxcywh":
            x1,y1,x2,y2 = x-w/2, y-h/2, x+w/2, y+h/2
        else:
            raise "unrecognise bbox format"
        cv2.rectangle(im,pt1=(int(x1),int(y1)),pt2=(int(x2),int(y2)),color=color,thickness=2)
        cv2.putText(im, str(id)+"_"+str(s)[:5], (int(x1),int(y1-1)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
    return im

def get_flight_id(img_name):
    return img_name[-36:-4]


def get_img_path(img_name,flight_id):
    
    flight_dir = f'data/airmot_jpg/val/sub2/' + flight_id
    if "_jpg" in flight_dir:
        img_name = img_name[:-3] + "jpg"
    img_path = os.path.join(flight_dir, img_name)
    return img_path


def re_construct_out(outputs):

    new_result = {}
    for output in outputs:
        img_name = output['img_name']
        detections = output['detections']
        if img_name not in new_result:
            new_result[img_name] = []
        new_result[img_name].extend(detections)
    return new_result


def complete_unvis(vis_fid_dir,src_dir,dst_dir='data/vis_result/baseline/baseline_png_2048'):
    
    val_fids = os.listdir(vis_fid_dir)
    for fid in val_fids:
        src_fid_dir = os.path.join(src_dir,fid)
        for img in os.listdir(src_fid_dir):
            src_img_path = os.path.join(src_fid_dir,img)
            dst_img_path = os.path.join(dst_dir,fid,img)
            dst_fid_dir = os.path.join(dst_dir,fid)
            os.makedirs(dst_fid_dir,exist_ok=True)
            if not os.path.exists(dst_img_path):
                shutil.copyfile(src_img_path,dst_img_path)
                print(dst_img_path)


def vis_det_output(out_file,vis_fid_dir,save_root,prefix="part3",format="xywh"):

    os.makedirs(save_root,exist_ok=True)
    outputs = load_json(out_file)

    # val_flight_ids = os.listdir(vis_fid_dir)
    val_flight_ids = ['part37af99a22970c4a2180a247e1117a2848','part39ebbf180342948fdae5eb1c351e01416']
    # re_outputs = re_construct_out(outputs)
    for fid in tqdm(list(outputs.keys())):
        imgs = list(outputs[fid].keys())
        # flight_id = get_flight_id(img_name)

        if fid in val_flight_ids:
            for img_name in imgs:
                img_path = get_img_path(img_name,fid)
                
                img = cv2.imread(img_path)
                detections = outputs[fid][img_name]
                drawed_img = draw_bbox(img,detections,format=format)
                save_dir = os.path.join(save_root,fid)
                os.makedirs(save_dir,exist_ok=True)
                if "jpg" in img_path:
                    img_name = img_name[:-3] + 'jpg'
                save_path = os.path.join(save_dir,img_name)
                cv2.imwrite(save_path,drawed_img)

    complete_unvis(vis_fid_dir,src_dir=vis_fid_dir,dst_dir=save_root)



def vis_track_output(out_file,vis_fid_dir,save_root,prefix="part3",format="xywh"):

    os.makedirs(save_root,exist_ok=True)
    outputs = load_json(out_file)

    val_flight_ids = os.listdir(vis_fid_dir)
    re_outputs = re_construct_out(outputs)
    for img_name in tqdm(list(re_outputs.keys())):

        flight_id = prefix + get_flight_id(img_name)
        detections = re_outputs[img_name]
        if flight_id in val_flight_ids:
            img_path = get_img_path(img_name,flight_id)
            img = cv2.imread(img_path)
            drawed_img = draw_track_bbox(img,detections,format=format)
            save_dir = os.path.join(save_root,flight_id)
            os.makedirs(save_dir,exist_ok=True)
            if "jpg" in img_path:
                img_name = img_name[:-3] + 'jpg'
            save_path = os.path.join(save_dir,img_name)
            cv2.imwrite(save_path,drawed_img)

    complete_unvis(vis_fid_dir,src_dir=vis_fid_dir,dst_dir=save_root)
                
    

if __name__ == "__main__":
    complete_unvis()

more_one_track_id=['part385f31212bfb349b292a3f1972614de62', 'part336666d895fd547788e6445566da84804', 'part387f82e3d7d1147f393ae50d2238853d6', 'part35543ff9c5aad48eaa315c094f400a1e6', 'part34d196dd04a23411193d525b2356a1187', 'part33e7c14aca91f4a7f81c61e44bc4fad6f', 'part3b60446f4ac1949e59483733431e0c99d', 'part3452619a2a4124743b46af72ef03d7126', 'part3f772783009b14b15887db395730c4ad5', 'part363aa38f2212347d0878ad1a29ff80ce6', 'part392e231d236a0495d832bef72c2094ce9', 'part3a0e9362dca144394a4ebbd3c3146c877', 'part3a3b702092d11421088f4752ce512c328']