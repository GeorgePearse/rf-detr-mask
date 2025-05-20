#!/usr/bin/env python
"""
Test script to verify custom COCO annotation file loading
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rfdetr.datasets import build_dataset


def test_custom_annotations():
    """Test loading custom COCO annotation files"""

    # Create mock args object
    args = argparse.Namespace()
    args.dataset_file = "coco"
    args.coco_path = "/home/georgepearse/data/cmr/annotations"
    args.coco_train = "2025-05-15_12:38:23.077836_train_ordered.json"
    args.coco_val = "2025-05-15_12:38:38.270134_val_ordered.json"
    args.coco_img_path = "/home/georgepearse/data/images"
    args.multi_scale = False
    args.expanded_scales = False

    print("Testing with custom annotation files:")
    print(f"coco_path: {args.coco_path}")
    print(f"coco_train: {args.coco_train}")
    print(f"coco_val: {args.coco_val}")
    print(f"coco_img_path: {args.coco_img_path}")

    # Test building train dataset
    print("\nBuilding train dataset...")
    try:
        train_dataset = build_dataset(image_set="train", args=args, resolution=640)
        print(f"✓ Train dataset loaded successfully with {len(train_dataset)} samples")

        # Check a sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"  Sample image shape: {sample[0].shape}")
            print(f"  Sample target keys: {sample[1].keys()}")
    except Exception as e:
        print(f"✗ Failed to load train dataset: {e}")

    # Test building val dataset
    print("\nBuilding val dataset...")
    try:
        val_dataset = build_dataset(image_set="val", args=args, resolution=640)
        print(f"✓ Val dataset loaded successfully with {len(val_dataset)} samples")

        # Check a sample
        if len(val_dataset) > 0:
            sample = val_dataset[0]
            print(f"  Sample image shape: {sample[0].shape}")
            print(f"  Sample target keys: {sample[1].keys()}")
    except Exception as e:
        print(f"✗ Failed to load val dataset: {e}")

    # Test with absolute paths
    print("\nTesting with absolute paths...")
    args.coco_train = Path(args.coco_path) / args.coco_train
    args.coco_val = Path(args.coco_path) / args.coco_val

    try:
        train_dataset_abs = build_dataset(image_set="train", args=args, resolution=640)
        print(
            f"✓ Train dataset with absolute path loaded successfully with {len(train_dataset_abs)} samples"
        )
    except Exception as e:
        print(f"✗ Failed to load train dataset with absolute path: {e}")

    # Test without custom annotations (should use default)
    print("\nTesting without custom annotations (should fail with default paths)...")
    args_default = argparse.Namespace()
    args_default.dataset_file = "coco"
    args_default.coco_path = "/home/georgepearse/data/cmr/annotations"
    args_default.multi_scale = False
    args_default.expanded_scales = False

    try:
        train_dataset_default = build_dataset(image_set="train", args=args_default, resolution=640)
        print(
            f"✓ Default train dataset loaded successfully with {len(train_dataset_default)} samples"
        )
    except Exception as e:
        print(f"✗ Failed to load default train dataset (expected): {e}")

    print("\nTest completed!")


if __name__ == "__main__":
    test_custom_annotations()
