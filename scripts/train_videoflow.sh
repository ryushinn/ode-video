#!/bin/bash
# Usage: train_videoflow.sh <data_path> <exp_path> <image_size> <num_processes> <accumulation_steps>


accelerate launch --num_processes=$4 --num_machines=1 --mixed_precision=bf16 --dynamo_backend="inductor" \
    video-flow.py data.path="$1" log.exp_path="$2" log.comment="video_flow" \
    branch.name="video_flow" \
    training.n_iter=200_000 training.batchsize=128 training.lr=1e-4 training.accumulation_steps=$5 \
    model.config="default_3_3_192" \
    data.image_size="[$3, $3]" \


accelerate launch --num_processes=$4 --num_machines=1 --mixed_precision=bf16 --dynamo_backend="inductor" \
    video-flow.py data.path="$1" log.exp_path="$2" log.comment="condiff" \
    branch.name="condiff" \
    training.n_iter=200_000 training.batchsize=128 training.lr=1e-4 training.accumulation_steps=$5 \
    model.config="default_6_3_192" \
    data.image_size="[$3, $3]" \


accelerate launch --num_processes=$4 --num_machines=1 --mixed_precision=bf16 --dynamo_backend="inductor" \
    video-flow.py data.path="$1" log.exp_path="$2" log.comment="video_biflow" \
    branch.name="video_biflow" training.power=1 \
    training.n_iter=200_000 training.batchsize=128 training.lr=1e-4 training.accumulation_steps=$5 \
    model.config="two_model_default_cond_3_3_128" \
    data.image_size="[$3, $3]" \
