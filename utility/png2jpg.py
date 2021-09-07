import sys
import os
from tqdm import tqdm
from multiprocessing import Pool

def get_name(file_name):
    name = file_name.split(".")[0]
    return name

def system_run(cmd):
    os.system(cmd)

part = "part3"


input_root = f"/workspace/airborne-detection-starter-kit/data/airmot/{part}/Images"
output_root = f"/workspace/airborne-detection-starter-kit/data/airmot/jpg/{part}/Images"
os.makedirs(output_root,exist_ok=True)

convert_pool = Pool(40)
convert_args = []
for flight_id in tqdm(os.listdir(input_root)):
    flight_id_outdir = os.path.join(output_root,flight_id)
    os.makedirs(flight_id_outdir,exist_ok=True)
    flight_id_indir = os.path.join(input_root,flight_id)
    for img in os.listdir(flight_id_indir):
        name = get_name(img)
        input_img_path = os.path.join(flight_id_indir,img)
        out_img_path = os.path.join(flight_id_outdir,name) + ".jpg"
        cmd = "ffmpeg -i " + input_img_path + " " + out_img_path
        convert_args.append(cmd)
        # os.system(cmd)

convert_pool.map(system_run,convert_args)
convert_pool.close()