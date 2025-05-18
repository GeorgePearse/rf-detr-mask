#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR-Mask Test Script for test_limit Parameter
# ------------------------------------------------------------------------

"""
Script to test the test_limit parameter implementation.
This script creates a dataset with a specified test_limit and prints the dataset size.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rfdetr.datasets import build_dataset
from rfdetr.util.logging_config import get_logger

logger = get_logger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser("Test test_limit parameter", add_help=True)

    # Dataset parameters
    parser.add_argument(
        "--coco_path",
        type=str,
        default="/home/georgepearse/data/cmr/annotations",
        help="Path to the annotations directory",
    )
    parser.add_argument(
        "--coco_train",
        type=str,
        default="2025-05-15_12:38:23.077836_train_ordered.json",
        help="Training annotation file name",
    )
    parser.add_argument(
        "--coco_val",
        type=str,
        default="2025-05-15_12:38:38.270134_val_ordered.json",
        help="Validation annotation file name",
    )
    parser.add_argument(
        "--coco_img_path",
        type=str,
        default="/home/georgepearse/data/images",
        help="Path to the images directory",
    )
    parser.add_argument("--dataset_file", default="coco", type=str, help="Dataset file name")
    parser.add_argument("--resolution", default=644, type=int, help="Input resolution")
    
    # Test limit parameter
    parser.add_argument(
        "--test_limit",
        default=None,
        type=int,
        help="Limit dataset to first N samples for faster testing"
    )
    
    # Model parameters required for build_dataset
    parser.add_argument("--multi_scale", action="store_true", help="Use multi-scale training")
    parser.add_argument("--expanded_scales", action="store_true", help="Use expanded scales for multi-scale training")
    parser.add_argument("--square_resize_div_64", action="store_true", help="Use square resize with dimensions divisible by 64")
    
    return parser


def main(args):
    logger.info(f"Testing test_limit parameter with limit={args.test_limit}")

    # Build datasets
    dataset_train = build_dataset(image_set="train", args=args, resolution=args.resolution)
    dataset_val = build_dataset(image_set="val", args=args, resolution=args.resolution)

    # Print dataset sizes
    logger.info(f"Training dataset size: {len(dataset_train)}")
    logger.info(f"Validation dataset size: {len(dataset_val)}")
    
    # Print first few sample IDs to verify correct subsetting
    if hasattr(dataset_train, "dataset"):
        # If it's a Subset, get the first few indices
        logger.info(f"Training dataset is a Subset with indices: {dataset_train.indices[:5]}...")
    else:
        # Otherwise print the first few image IDs
        logger.info(f"Training dataset first few image IDs: {[dataset_train.ids[i] for i in range(min(5, len(dataset_train.ids)))]}...")
        
    if hasattr(dataset_val, "dataset"):
        # If it's a Subset, get the first few indices
        logger.info(f"Validation dataset is a Subset with indices: {dataset_val.indices[:5]}...")
    else:
        # Otherwise print the first few image IDs
        logger.info(f"Validation dataset first few image IDs: {[dataset_val.ids[i] for i in range(min(5, len(dataset_val.ids)))]}...")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    
    main(args)