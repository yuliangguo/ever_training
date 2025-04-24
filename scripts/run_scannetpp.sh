#!/bin/bash

# Train a scene (Scannet++)
DATASET_PATH="/mnt/data_ssd_4tb/Datasets/scannetpp_tiny/data/1d003b07bd/dslr/"
OUTPUT_PATH="output/scannetpp/dslr/1d003b07bd"

python train.py \
    -s $DATASET_PATH \
    --eval \
    --images resized_images \
    -r 1 \
    -m $OUTPUT_PATH

# Render a trained model (Scannet++)
python render.py \
    -m $OUTPUT_PATH \
    --skip_train \
    -r 1

# Eval a rendered samples
python metrics.py \
    -m $OUTPUT_PATH
