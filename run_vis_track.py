import os
import argparse

from utility.vis_output import vis_track_output as run

def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_file', type=str, default="data/results/tbd_png_e5_2048/result.json",help="path to the output of model")
    parser.add_argument("--vis_fid_dir", type=str, default="data/airmot/airmot_jpg/val/sub1", help="find which flight id need to vis")
    parser.add_argument("--save_root", type=str, default="data/vis_result/baseline/tbd_png_e5_2048", help="expected output root path")

    parser.add_argument('--prefix', type=str, default="part3",help="part1,part2,part3")
    parser.add_argument('--format', type=str, default="cxcywh",help="xywh,xyxy,cxcywh")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    run(args.out_file,args.vis_fid_dir, args.save_root, args.prefix, args.format)