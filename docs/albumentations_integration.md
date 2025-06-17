# Albumentations Integration Guide

This guide explains how to use albumentations for data augmentation in RF-DETR-MASK training.

## Installation

First, install albumentations:

```bash
pip install albumentations>=1.3.0 opencv-python pyyaml
```

## Usage

### Command Line Arguments

To use albumentations instead of the built-in transforms, add these arguments:

- `--use_albumentations`: Enable albumentations transforms
- `--albumentations_config`: Path to YAML configuration file

Example:
```bash
python scripts/train.py \
    --use_albumentations \
    --albumentations_config configs/transforms/default_detection.yaml
```

### YAML Configuration Format

Create YAML files to define your augmentation pipeline. The format is:

```yaml
train_transforms:
  - name: HorizontalFlip
    params:
      p: 0.5

  - name: RandomBrightnessContrast
    params:
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.5

  - name: Normalize
    params:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

  - name: ToTensorV2

val_transforms:
  - name: Normalize
    params:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

  - name: ToTensorV2
```

### Pre-configured Transforms

We provide several pre-configured transform sets:

1. **Default Detection** (`configs/transforms/default_detection.yaml`)
   - Standard augmentations for object detection
   - Includes flip, color jitter, random scale, and crop

2. **Strong Augmentation** (`configs/transforms/segmentation_strong_aug.yaml`)
   - More aggressive augmentations for instance segmentation
   - Includes perspective, rotation, blur, noise, and cutout

3. **DINOv2 Square** (`configs/transforms/dinov2_square.yaml`)
   - Square resizing for DINOv2 models
   - Ensures dimensions are divisible by 64

### Custom Transforms

To create custom transforms:

1. Create a new YAML file with your desired augmentations
2. Use any transform from the [albumentations documentation](https://albumentations.ai/docs/api_reference/augmentations/)
3. Ensure you include `ToTensorV2` as the last transform

Example custom transform:
```yaml
train_transforms:
  - name: RandomRotate90
    params:
      p: 0.5

  - name: OneOf
    params:
      transforms:
        - name: GaussianBlur
          params:
            blur_limit: [3, 7]
        - name: MedianBlur
          params:
            blur_limit: [3, 7]
      p: 0.3

  - name: CLAHE
    params:
      clip_limit: 4.0
      p: 0.3

  - name: Normalize
    params:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

  - name: ToTensorV2
```

### Important Notes

1. **Bounding Box Format**: COCO uses `[x, y, width, height]` format. The wrapper handles conversions automatically.

2. **Mask Support**: Masks are automatically handled when `--masks` is enabled.

3. **Performance**: Albumentations is generally faster than torchvision transforms due to optimized implementations.

4. **Compatibility**: The integration maintains full backward compatibility - you can still use the original transforms by not specifying `--use_albumentations`.

### Troubleshooting

If you get import errors:
```bash
pip install albumentations opencv-python pyyaml
```

If transforms aren't working as expected:
- Check that all transform names match exactly (case-sensitive)
- Verify parameter names match the albumentations documentation
- Ensure `ToTensorV2` is the last transform in the pipeline
