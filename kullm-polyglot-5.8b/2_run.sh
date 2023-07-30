#!/bin/bash

set -e
source activate pytorch_p310

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

torchrun --nnodes=1 --nproc-per-node=$NUM_GPUS 2_local-inference-deepspeed.py