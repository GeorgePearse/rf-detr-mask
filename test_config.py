#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR-Mask Test Configuration
# ------------------------------------------------------------------------

"""
Script to test loading and validating YAML configuration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rfdetr.config_utils import load_config

# Test loading a configuration
config_path = "configs/cmr_segmentation.yaml"
print(f"Loading configuration from {config_path}")

try:
    config = load_config(config_path)
    print("Configuration loaded successfully!")
    print("\nModel configuration:")
    print(f"  Encoder: {config.model.encoder}")
    print(f"  Hidden dimension: {config.model.hidden_dim}")
    print(f"  Number of classes: {config.model.num_classes}")
    
    print("\nTraining configuration:")
    print(f"  Learning rate: {config.training.lr}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.epochs}")
    
    print("\nDataset configuration:")
    print(f"  COCO path: {config.dataset.coco_path}")
    print(f"  Training file: {config.dataset.coco_train}")
    
    print("\nMask configuration:")
    print(f"  Enabled: {config.mask.enabled}")
    print(f"  Mask loss coefficient: {config.mask.loss_mask_coef}")
    
    print("\nOther configuration:")
    print(f"  Seed: {config.other.seed}")
    print(f"  Device: {config.other.device}")
    
    # Convert to args for backward compatibility
    args = config.to_args()
    print("\nArgparse namespace created successfully!")
    print(f"  num_classes: {args.num_classes}")
    print(f"  hidden_dim: {args.hidden_dim}")
    print(f"  lr: {args.lr}")
    print(f"  batch_size: {args.batch_size}")
    
    print("\nConfiguration validation passed!")
except Exception as e:
    print(f"Error loading configuration: {e}")
    sys.exit(1)

print("\nAll tests passed!")