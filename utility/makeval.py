import os
import random
import json
import shutil



def load_json(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    return data

# part3_root = "/data/airmot/part3/Images"

# f_ids = os.listdir(part3_root)
# random.shuffle(f_ids)
# print(f_ids[:200])

# with open("data/airmot/val_flight_ids.json",'w') as f:
#     json.dump(f_ids[:200],f)

val_ids = load_json('data/airmot/val_flight_ids.json')
src_path = 'data/airmot/part3/Images'
dst_path = 'data/airmot/val'

for id in val_ids:
    shutil.copytree(os.path.join(src_path,id),os.path.join(dst_path,id))
