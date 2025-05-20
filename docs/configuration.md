# Configuration System

RF-DETR-Mask now supports a YAML-based configuration system as an alternative to command-line arguments.
This makes it easier to manage and reuse configurations for different experiments.

## Configuration Structure

The configuration is divided into five main sections:

1. `model`: Configuration for the RF-DETR model architecture
2. `training`: Parameters for training (learning rates, batch size, etc.)
3. `dataset`: Dataset paths and file names
4. `mask`: Mask prediction related settings
5. `other`: Miscellaneous settings like seed, device, etc.

## Using Configuration Files

### Basic Usage

To train using a YAML configuration file:

```bash
python scripts/train_from_config.py --config configs/cmr_segmentation.yaml
```

### Overriding Configuration Values

You can override specific values from the command line:

```bash
python scripts/train_from_config.py --config configs/cmr_segmentation.yaml --batch_size 2 --epochs 50
```

### Available Command-Line Overrides

- `--output_dir`: Override output directory
- `--pretrain_weights`: Override pretrained weights path
- `--batch_size`: Override batch size
- `--epochs`: Override number of epochs
- `--resume`: Resume from checkpoint
- `--eval`: Run in evaluation-only mode
- `--seed`: Override random seed

## Configuration Validation

The configuration system uses Pydantic for validation, ensuring that:

- All required parameters are present
- Parameters are of the correct type
- Numerical parameters are within valid ranges
- Training dimensions (width and height) are divisible by 14 for DINOv2
- Configuration sections are consistent with each other

## Example Configuration

```yaml
# Model Configuration
model:
  encoder: "dinov2_windowed_small"
  out_feature_indexes: [2, 5, 8, 11]
  dec_layers: 3
  two_stage: true
  projector_scale: ["P4"]
  hidden_dim: 256
  sa_nheads: 8
  ca_nheads: 16
  dec_n_points: 2
  bbox_reparam: true
  lite_refpoint_refine: true
  layer_norm: true
  amp: true
  num_classes: 69  # CMR has 69 classes
  pretrain_weights: "rf-detr-base.pth"
  device: "cuda"
  training_width: 560
  training_height: 560
  group_detr: 13
  gradient_checkpointing: false
  num_queries: 300
  num_select: 300

# Training Configuration
training:
  lr: 5e-5
  lr_encoder: 5e-6
  batch_size: 1
  grad_accum_steps: 4
  epochs: 100
  # ...and more training parameters...

# Dataset Configuration
dataset:
  coco_path: "/home/georgepearse/data/cmr/annotations"
  coco_train: "2025-05-15_12:38:23.077836_train_ordered.json"
  coco_val: "2025-05-15_12:38:38.270134_val_ordered.json"
  coco_img_path: "/home/georgepearse/data/images"
  
# Mask Configuration
mask:
  enabled: true
  loss_mask_coef: 1.0
  loss_dice_coef: 1.0

# Other Configuration
other:
  seed: 42
  device: "cuda"
  world_size: 1
  dist_url: "env://"
  clip_max_norm: 0.5
  steps_per_validation: 0
```

## Creating Custom Configurations

1. Start by copying one of the example configurations:
   - `base_config.yaml`: Basic configuration
   - `cmr_segmentation.yaml`: Configuration optimized for CMR segmentation

2. Modify the parameters as needed for your experiment.

3. Save your custom configuration in the `configs` directory.

## Implementation Details

The configuration system is implemented using:
- `rfdetr/config_utils.py`: Contains the Pydantic models for configuration validation
- `scripts/train_from_config.py`: Training script that uses YAML configuration

The Pydantic models provide:
- Type checking and validation
- Default values for optional parameters
- Value range validation
- Cross-field validation