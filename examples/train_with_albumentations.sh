#!/bin/bash
# Example script showing how to train with albumentations

# Train with default detection augmentations
python scripts/train.py \
    --use_albumentations \
    --albumentations_config configs/transforms/default_detection.yaml \
    --steps_per_validation 20 \
    --test_limit 20

# Train with strong augmentations for segmentation
python scripts/train.py \
    --use_albumentations \
    --albumentations_config configs/transforms/segmentation_strong_aug.yaml \
    --masks \
    --steps_per_validation 20 \
    --test_limit 20

# Train with DINOv2 square transforms
python scripts/train.py \
    --use_albumentations \
    --albumentations_config configs/transforms/dinov2_square.yaml \
    --square_resize \
    --steps_per_validation 20 \
    --test_limit 20
