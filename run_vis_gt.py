import argparse

from utility.draw_gt import draw_part as run

def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_root', type=str, default="data/airmot/airmot_jpg/train",help="path to the input image")
    parser.add_argument("--output_root", type=str, default="data/vis_result/gt", help="expected output root path")
    parser.add_argument('--part', type=str, default="part1",help="part1,part2,part3")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    run(args.input_root, args.output_root, args.part)
