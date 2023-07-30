#!/bin/bash

set -e
pip install -r requirements.txt

mkdir -p /tmp/huggingface-cache/
export HF_DATASETS_CACHE="/tmp/huggingface-cache"

declare -a OPTS=(
    --base_model nlpai-lab/kullm-polyglot-12.8b-v2
    --pretrained_model_path /home/ec2-user/SageMaker/models/kullm-polyglot-12-8b-v2/
    --cache_dir $HF_DATASETS_CACHE
    --data_path ../train
    --output_dir ckpt/output
    --save_path ./model
    --batch_size 2
    --num_epochs 1
    --learning_rate 3e-4
    --lora_r 8 \
    --lora_alpha 32
    --lora_dropout 0.05
    --lora_target_modules "[query_key_value, xxx]"
    --logging_steps 1
    --eval_steps 40
    --weight_decay 0.
    --warmup_steps 0
    --warmup_ratio 0.1
    --lr_scheduler_type "cosine"
)

NUM_GPUS=4
echo torchrun --nnodes 1 --nproc_per_node "$NUM_GPUS" train.py "${OPTS[@]}" "$@"
torchrun --nnodes 1 --nproc_per_node "$NUM_GPUS" train.py "${OPTS[@]}" "$@"

