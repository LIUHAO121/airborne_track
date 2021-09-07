from operator import inv
import os
import cv2
import json

part="part1"
train_dir = f'data/airmot/airmot_png/train/{part}'
flight_ids = os.listdir(train_dir)

invalid_png = []
parts = ["part1","part2","part3"]
for part in parts:
    train_dir = f'data/airmot/airmot_png/train/{part}'
    flight_ids = os.listdir(train_dir)
    for fid in flight_ids:
        fid_dir = os.path.join(train_dir,fid)
        for img in os.listdir(fid_dir):
                if len(img.split('.'))>2:
                    print(img.rsplit(".",1)[0])
                    invalid_png.append(img.rsplit(".",1)[0])

print(len(invalid_png))

json.dump(invalid_png,open("invalid.json",'w'))
# ann_file = f"/workspace/airborne/airborne-detection-starter-kit/data/airmot/annotation/combine/train/{part}_annotation.json"

# annotation = json.load(open(ann_file,'r'))

# for png_name in invalid_png:
#     fid_id = png_name[-36:-4]
#     img_keys = annotation[fid_id]
#     if png_name in list(img_keys):
#         print(png_name)

