#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR-Mask Training Script for CMR Dataset with Fixed Size
# ------------------------------------------------------------------------

"""
Script to train RF-DETR with mask head on CMR segmentation data with fixed image dimensions.
This script uses 896x1232 images, where both dimensions are divisible by 56.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rfdetr.config_utils import load_config
from scripts.train import main as train_main

def parse_args():
    parser = argparse.ArgumentParser("Train RF-DETR-Mask with Fixed Size Images")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fixed_size_config.yaml", 
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--fixed_width", 
        type=int,
        default=1232,
        help="Fixed width dimension (must be divisible by 56)"
    )
    parser.add_argument(
        "--fixed_height",
        type=int,
        default=896,
        help="Fixed height dimension (must be divisible by 56)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_fixed_size",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run with a smaller batch size and fewer epochs for testing"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the configuration
    config = load_config(args.config)
    
    # Override with command-line arguments
    config_args = config.to_args()
    
    # Update fixed dimensions from command-line
    config_args.fixed_width = args.fixed_width
    config_args.fixed_height = args.fixed_height
    config_args.output_dir = args.output_dir
    
    # Set test mode if requested
    if args.test:
        config_args.epochs = 2
        config_args.train_batch_size = 1
        config_args.val_batch_size = 1
        config_args.steps_per_validation = 10
    
    # Print the settings
    print(f"Training with fixed dimensions: {config_args.fixed_height}x{config_args.fixed_width}")
    print(f"Output directory: {config_args.output_dir}")
    
    # Run the training
    train_main(config_args)

if __name__ == "__main__":
    main()