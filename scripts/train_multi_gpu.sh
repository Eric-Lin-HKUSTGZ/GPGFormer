#!/bin/bash
# Multi-GPU training script for GPGFormer
# Usage: bash scripts/train_multi_gpu.sh [num_gpus] [config_path] [master_port]

set -e

NUM_GPUS=${1:-4}
CONFIG=${2:-configs/config_ho3d.yaml}
MASTER_PORT=${3:-29500}

echo "Starting multi-GPU training with $NUM_GPUS GPUs"
echo "Config: $CONFIG"
echo "Master port: $MASTER_PORT"

torchrun --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  train.py --config "$CONFIG"


