#!/bin/bash
# Quick test to verify training with fp16_eval works

cd /home/georgepearse/rf-detr-mask

python scripts/train.py \
    --num_classes 2 \
    --masks \
    --epochs 1 \
    --lr_drop 1 \
    --num_workers 2 \
    --batch_size 2 \
    --output_dir test_output_fp16 \
    --fp16_eval \
    --eval_every_epoch 1 \
    --warmup_epochs 0 \
    --save_checkpoint_every_epoch 1