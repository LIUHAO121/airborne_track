python3 run_vis_det.py --out_file data/det_results03.json --vis_fid_dir data/airmot_jpg/val/sub2 --save_root data/vis_result/centernet/model_res18_e35_2048 --prefix part3 --format xyxy
python3 run_vis_track.py --out_file data/results/res18_e35_2048_fr3sd2_ml20_tl5_s0.5_ts0.5_maxone/result.json --vis_fid_dir data/airmot_jpg/val/sub1 --save_root data/vis_result/tbd/res18_e35_2048_fr3sd2_ml20_tl5_s0.5_ts0.5_maxone --prefix part3 --format cxcywh

python3 run_vis_track.py --out_file data/results/res18_e35_2048_fr3sd2_ml20_tl5_s0.5_ts0.5_fa005/result.json --vis_fid_dir data/airmot_jpg/val/more_one --save_root data/vis_result/tbd/res18_e35_2048_fr3sd2_ml20_tl5_s0.5_ts0.5_fa005 --prefix part3 --format cxcywh

python3 run_vis_track.py --out_file data/results/res18_e35_2048_fr3sd2_ml20_tl5_s0.5_ts0.5_fa0.05_single_none/result.json --vis_fid_dir data/airmot_jpg/val/fpfid --save_root data/vis_result/tbd/res18_e35_2048_fr3sd2_ml20_tl5_s0.5_ts0.5_fa005_single_none --prefix part3 --format cxcywh
python3 run_vis_track.py --out_file data/results/res18_e35_2048_fr3sd2_ml1_tl1_s0.5_ts0.5_fa0.05_single_none/result.json --vis_fid_dir data/airmot_jpg/val/fpfid --save_root data/vis_result/tbd/res18_e35_2048_fr3sd2_ml1_tl1_s0.5_ts0.5_fa0.05_single_none --prefix part3 --format cxcywh
python3 run_vis_track.py --out_file data/results/res18_e35_2048_fr3sd2_ml1_tl1_s0.5_ts0.5_fa0.05_emb_none/result.json --vis_fid_dir data/airmot_jpg/val/fpfid --save_root data/vis_result/tbd/res18_e35_2048_fr3sd2_ml1_tl1_s0.5_ts0.5_fa0.05_emb_none --prefix part3 --format cxcywh


python3 run_vis_track.py --out_file data/results/res18_e35_2048_fr3sd2_ml0_tl0_s0.5_ts0.5_fa0.05_single_MaxScoreAndArea/result.json --vis_fid_dir data/airmot_jpg/val/max_area_score --save_root data/vis_result/tbd/res18_e35_2048_fr3sd2_ml0_tl0_s0.5_ts0.5_fa0.05_single_MaxScoreAndArea --prefix part3 --format cxcywh

python3 run_vis_track.py --out_file data/results/res18_e35_2048_fr3sd2_ml0_tl0_s0.5_ts0.5_fa0.05_single_MaxScoreMultiSize/result.json --vis_fid_dir data/airmot_jpg/val/max_area_score --save_root data/vis_result/tbd/res18_e35_2048_fr3sd2_ml0_tl0_s0.5_ts0.5_fa0.05_single_MaxScoreMultiSize --prefix part3 --format cxcywh
