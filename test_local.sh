MIN_SCORE=0.5
TRACK_SCORE=0.5
MIN_TRACK_LEN=0
TRACK_LET_START=0
FRAME_RATE=3
SPATIAL_DIST=2
FEAT_ALPHA=0.05
TRACK_METHOD=single
TAG=MaxScoreMultiSize
CUDA_VISIBLE_DEVICES=1 python track_by_det_local.py $MIN_SCORE $TRACK_SCORE $MIN_TRACK_LEN $TRACK_LET_START $FRAME_RATE $SPATIAL_DIST $FEAT_ALPHA $TRACK_METHOD
python utility/utils.py
python3 core/metrics/run_airborne_metrics.py --dataset-folder data/part3_val --results-folder data/results/run0
out_name="res18_e35_2048_fr${FRAME_RATE}sd${SPATIAL_DIST}_ml${MIN_TRACK_LEN}_tl${TRACK_LET_START}_s${MIN_SCORE}_ts${TRACK_SCORE}_fa${FEAT_ALPHA}_${TRACK_METHOD}_${TAG}"
mv data/results/run0 data/results/${out_name}