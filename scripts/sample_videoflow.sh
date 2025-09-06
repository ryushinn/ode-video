#!/bin/bash
# Usage: sample_videoflow.sh <data_path> <model_path> <exp_path> <image_size> <n_samples> <batchsize> <n_frames>

python sample.py data.path="$1" \
    branch.name="video_flow" \
    model.config="default_3_3_192" \
    model.weights_path="$2/video_flow/checkpoints/model.safetensors" \
    data.image_size="[$4, $4]" \
    inference.n_samples=$5 inference.batchsize=$6 inference.n_frames=$7 \
    log.exp_path="$3" log.comment="video_flow"

python sample.py data.path="$1" \
    branch.name="condiff" \
    model.config="default_6_3_192" \
    model.weights_path="$2/condiff/checkpoints/model.safetensors" \
    data.image_size="[$4, $4]" \
    inference.n_samples=$5 inference.batchsize=$6 inference.n_frames=$7 \
    log.exp_path="$3" log.comment="condiff"

solve_backward=false
noise_levels=(0.00 0.10 0.20 0.30)
for noise in "${noise_levels[@]}"; do
    python sample.py data.path="$1" \
        branch.name="video_biflow" \
        model.config="two_model_default_cond_3_3_128" \
        model.weights_path="$2/video_biflow/checkpoints/model.safetensors" \
        data.image_size="[$4, $4]" \
        inference.n_samples=$5 inference.batchsize=$6 inference.n_frames=$7 \
        inference.backward=$solve_backward \
        log.exp_path="$3" log.comment="video_biflow_$noise" inference.noise_level=$noise
done