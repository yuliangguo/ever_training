#!/bin/bash
for set in bicycle flowers garden stump treehill room counter kitchen bonsai; do
# for set in bicycle flowers garden stump treehill; do
# for set in kitchen bonsai; do
  # PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python train.py -s /data/nerf_synthetic/$set --densify_grad_threshold=3e-7 --convert_SHs_python
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python train.py -s /data/nerf_synthetic/$set --densify_grad_threshold=3e-7 --eval
  # PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python train.py -s /data/nerf_synthetic/$set --densify_grad_threshold=3e-7 --use_neural_network --data_device cpu --eval --feature_rest_lr 0.0025
done
