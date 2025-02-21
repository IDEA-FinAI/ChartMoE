#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

export MODEL="ckpt/InternLM-XComposer2_Enhanced"
export DATA="data/table_align.txt"

GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12700

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS train.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --dataloader_num_workers 4 \
    --img_size 490 \
    --hd_num -1 \
    --given_num True \
    --bf16 True \
    --fix_vit True \
    --fix_sampler False \
    --fix_llm True \
    --use_lora False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_length 4096 \
    --gradient_checkpointing True \
    --deepspeed ds_config_zero2.json \
    --output_dir output/table_proj \
    --report_to none
