DATASET_PATH="/mnt/data_ssd_4tb/Datasets/zipnerf/fisheye/nyc"
OUTPUT_PATH="output/zipnerf/fisheye/nyc_d8"

python train.py \
    -m $OUTPUT_PATH \
    -s $DATASET_PATH \
    --images images_8 \
    --eval

python render.py \
    -m $OUTPUT_PATH \
    --skip_train \

python render.py \
    -m $OUTPUT_PATH \
    --skip_train \
    --images images_4 \
    --cross_camera

python metrics.py \
    -m $OUTPUT_PATH

python metrics.py \
    -m $OUTPUT_PATH \
    --cross_camera