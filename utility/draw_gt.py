import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import shutil

part = "part3"


part1_obj_classes = {'Bird3', 'Airborne11', 'Airplane5', 'Airborne9', 'Airplane10', 'Bird1', 'Bird4', 'Flock2', 'Bird11', 'Bird10', 'Drone1', 'Bird2', 'Airplane6', 'Airborne8', 'Airborne6', 'Airplane1', 'Airborne4', 'Bird16', 'Flock3', 'Airborne16', 'Helicopter3', 'Bird23', 'Airplane8', 'Airplane9', 'Bird18', 'Bird12', 'Bird14', 'Bird13', 'Airborne5', 'Airborne19', 'Bird20', 'Airborne18', 'Bird7', 'Bird15', 'Helicopter1', 'Airborne10', 'Airborne3', 'Airborne12', 'Bird24', 'Airplane7', 'Airborne13', 'Bird17', 'Bird6', 'Airplane2', 'Bird21', 'Airborne17', 'Airborne1', 'Airborne2', 'Bird5', 'Bird25', 'Flock1', 'Bird9', 'Helicopter2', 'Airborne7', 'Airplane3', 'Bird8', 'Bird19', 'Bird22', 'Airborne15', 'Airplane4', 'Airborne14'}
part2_obj_classes = {'Bird22', 'Bird3', 'Airborne2', 'Airplane5', 'Bird19', 'Airplane8', 'Bird30', 'Airplane2', 'Airborne1', 'Airborne6', 'Airborne7', 'Bird13', 'Bird26', 'Airplane7', 'Flock4', 'Bird27', 'Bird7', 'Airborne18', 'Bird24', 'Bird17', 'Bird20', 'Helicopter2', 'Bird4', 'Airborne9', 'Bird31', 'Bird15', 'Bird16', 'Bird29', 'Bird25', 'Airborne4', 'Airborne10', 'Airplane4', 'Bird8', 'Bird18', 'Airborne5', 'Airplane3', 'Bird11', 'Airborne3', 'Flock2', 'Flock6', 'BIrd5', 'Flock1', 'Airborne8', 'Airplane1', 'Bird21', 'Bird28', 'Airborne19', 'Bird23', 'Bird9', 'Helicopter3', 'Bird12', 'Drone1', 'Helicopter4', 'Flock3', 'Helicopter1', 'Bird14', 'Bird1', 'Bird5', 'Flock5', 'Bird10', 'Airplane6', 'Bird2', 'Bird6'}
part3_obj_classes = {'Bird33', 'Bird17', 'Airplane2', 'Bird29', 'Bird27', 'Bird12', 'Airborne17', 'Bird1', 'Bird4', 'Bird6', 'Airplane8', 'Airborne9', 'Airborne3', 'Bird9', 'Bird40', 'Airplane4', 'Bird16', 'Drone1', 'Airborne8', 'Helicopter3', 'Bird7', 'Flock5', 'Airplane6', 'Airborne7', 'Bird3', 'Airplane9', 'Airborne4', 'Bird24', 'Bird25', 'Bird10', 'Bird5', 'Flock2', 'Bird21', 'Bird23', 'Flock3', 'Airplane3', 'Bird38', 'Airborne13', 'Bird2', 'Airborne6', 'Bird36', 'Airborne15', 'Bird14', 'Airborne14', 'Flock4', 'Bird39', 'Helicopter2', 'Airborne1', 'Helicopter4', 'Bird13', 'Airborne16', 'Bird37', 'Airborne2', 'Airborne10', 'Airborne12', 'Bird8', 'Bird34', 'Flock1', 'Bird41', 'Bird31', 'Bird22', 'Bird20', 'Helicopter1', 'Bird28', 'Bird30', 'Bird15', 'Bird35', 'Bird19', 'Airplane5', 'Bird26', 'Airborne11', 'Bird32', 'Airplane1', 'Airplane7', 'Bird11', 'Airborne5', 'Bird18'}

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





def get_valid_encounter_imgs(valid_file_path):
    valid_encounters = {}
    ve = open(valid_file_path).read()
    for valid_encounter in ve.split('\n\n    '):
        valid_encounter = json.loads(valid_encounter)
        if valid_encounter["flight_id"] not in valid_encounters:
            valid_encounters[valid_encounter["flight_id"]] = []
        valid_encounters[valid_encounter["flight_id"]].extend(valid_encounter["img_name"])
    valid_encounter_imgs = valid_encounters
    return valid_encounter_imgs


def get_valid_encounter_gt(valid_file_path,gt_file,out_file):
    valid_encounter_gt = {}
    valid_encounter_imgs = get_valid_encounter_imgs(valid_file_path) #得到每个flightid对应的有效的图片名字的列表
    gt = json.load(open(gt_file,'r'))
    samples = gt["samples"]
    flight_ids = list(valid_encounter_imgs.keys())
    count=0
    for fid in flight_ids:
        if fid not in valid_encounter_gt:
            valid_encounter_gt[fid] = {}
        for entrty in samples[fid]['entities']:
            if "bb" in entrty and entrty['img_name'] in valid_encounter_imgs[fid]:
                if entrty['img_name'] not in valid_encounter_gt[fid]:
                    valid_encounter_gt[fid][entrty['img_name']] = []
                valid_encounter_gt[fid][entrty['img_name']].append(entrty)
                
            # if "id" in entrty:
            #     if "Airborne" in entrty["id"]:
            #         print(entrty)
    # print(count)
    # dump_json(out_file,valid_encounter_gt)

def calculate_each_class_size(valid_file_path,gt_file,class_name):
    valid_encounter_gt = {}
    valid_encounter_imgs = get_valid_encounter_imgs(valid_file_path)
    gt = json.load(open(gt_file,'r'))
    samples = gt["samples"]
    flight_ids = list(valid_encounter_imgs.keys())
    count=0
    w=0
    h=0
    for fid in flight_ids:
        if fid not in valid_encounter_gt:
            valid_encounter_gt[fid] = {}
        for entrty in samples[fid]['entities']:
            if "bb" in entrty and entrty['img_name'] in valid_encounter_imgs[fid]:
                if entrty['img_name'] not in valid_encounter_gt[fid]:
                    valid_encounter_gt[fid][entrty['img_name']] = []
                valid_encounter_gt[fid][entrty['img_name']].append(entrty)
                if "id" in entrty:
                    if class_name in entrty["id"]:
                        w += entrty['bb'][2]
                        h += entrty['bb'][3]
                        count+=1
    print("mean width:",w/count)
    print("mean height:",h/count)

            
                        
def draw_bbox(img,entrys):
    im = np.ascontiguousarray(np.copy(img))
    for entry in entrys:
        obj_class = entry['id']
        x,y,w,h = list(map(int,entry['bb']))
        x1,y1,x2,y2 = x,y,x+w,y+h
        cv2.rectangle(im,(x1,y1),(x2,y2),color=(255,0,0),thickness=2)
        cv2.putText(im, obj_class, (x1,y1-1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
    return im

def draw_part(in_dir="data/airmot/airmot_jpg", out_dir="data/vis_result/gt", part="part1"):
    gt_file = f"data/airmot/annotation/combine/train/{part}_annotation.json"
    gt = load_json(gt_file)
    in_dir = os.path.join(in_dir,part)
    out_dir = os.path.join(out_dir,part)
    os.makedirs(out_dir,exist_ok=True)
    for fid in tqdm(list(gt.keys())):
        imgs = gt[fid].keys()
        for img_name in imgs:
            if "jpg" in in_dir:
                img_path = os.path.join(in_dir,part+fid,img_name[:-3]+'jpg')
                if not os.path.exists(img_path):
                    print("no ",img_path)
                    png_path = img_path.replace("jpg/","")[:-3]+"png"
                    if os.path.exists(png_path):
                        shutil.copyfile(png_path,img_path)
                        print(" find ",img_path)
            else:
                img_path = os.path.join(in_dir,part+fid,img_name)
            
            img = cv2.imread(img_path)
            
            drawed_img = draw_bbox(img,gt[fid][img_name])
            flight_out_dir = os.path.join(out_dir,fid)
            os.makedirs(flight_out_dir,exist_ok=True)
            drawed_img_path = os.path.join(flight_out_dir,img_name[:-3] + "jpg")
            cv2.imwrite(drawed_img_path,drawed_img)


def check_ann(ann_path):
    anns = load_json(ann_path)
    flight_ids = list(anns.keys())
    for fid in flight_ids:
        for img_name in list(anns[fid].keys()):
            if len(anns[fid][img_name])>=4:
                print(anns[fid][img_name])
            

def split_part3(valid_file_path):
    val_ids = os.listdir("data/val")
    val_file = open("part3_valid_encounters_maxRange700_maxGap3_minEncLen30.json",'w')
    
    ve = open(valid_file_path).read()
    for valid_encounter in ve.split('\n\n    '):
        valid_encounter = json.loads(valid_encounter)
        if "part3"+valid_encounter["flight_id"] not in val_ids:
            valid_encounter = json.dumps(valid_encounter)
            val_file.write(valid_encounter)
            val_file.write('\n\n    ')
    val_file.close()
    
            



if __name__ == "__main__":
    part = "part1"
    gt_file = f"data/airmot/annotation/gt/train/{part}/groundtruth.json"
    valid_file_path = f"data/airmot/annotation/gt/train/{part}/valid_encounters_maxRange700_maxGap3_minEncLen30.json"
    
    # get_valid_encounter_imgs("data/airmot/annotation/gt/train/valid_encounters_maxRange700_maxGap3_minEncLen30.json")
    # split_part3("data/airmot/annotation/gt/train/valid_encounters_maxRange700_maxGap3_minEncLen30.json")

    re_gt_file =f"/workspace/airborne-detection-starter-kit/data/airmot/{part}_annotation.json"
    # get_valid_encounter_gt(valid_file_path,gt_file,re_gt_file)
    # draw_part(part=part)
    # ann_path = f"/workspace/airborne-detection-starter-kit/data/airmot/{part}_annotation.json"
    # ann_path="data/airmot/part3_annotation.json"
    # check_ann(ann_path)


    calculate_each_class_size(valid_file_path,gt_file,"Drone")
