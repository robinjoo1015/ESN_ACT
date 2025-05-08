#!/bin/bash

export MUJOCO_GL=egl

for seed in 0 1 2 3 4; do
    for task_name in "sim_transfer_cube_scripted"; do
        CUDA_VISIBLE_DEVICES=4 python3 imitate_episodes_esn.py \
        --policy_class ESN \
        --seed $seed \
        --task_name "$task_name" \
        --eval \
        --ckpt_dir /data/ysjoo/home/kcc2025/sim_transfer_cube/seed_$seed/
    done
done