DATASET_PATH="/mnt/data_ssd_4tb/Datasets/zipnerf/undistorted/nyc"
OUTPUT_PATH="output/zipnerf/undistorted/nyc_d4"

python train.py \
    -m $OUTPUT_PATH \
    -s $DATASET_PATH \
    --images images_4 \
    --eval

python render.py \
    -m $OUTPUT_PATH \
    --skip_train \

python render.py \
    -m $OUTPUT_PATH \
    --skip_train \
    --images images_8 \
    --cross_camera

python metrics.py \
    -m $OUTPUT_PATH

python metrics.py \
    -m $OUTPUT_PATH \
    --cross_camera
