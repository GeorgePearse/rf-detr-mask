#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Test script to verify the validation dataset can be limited to a specific number of samples.
"""

import argparse
import unittest
from pathlib import Path

from rfdetr.config import load_config
from rfdetr.datasets import build_dataset


class TestValLimit(unittest.TestCase):
    """Test case for validation dataset limit functionality."""

    def test_val_limit(self):
        """Test that the validation dataset is limited to the specified number of samples."""
        # Load the default config
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        config = load_config(config_path)
        args = argparse.Namespace(
            device="cpu",
            num_classes=config.model.num_classes,
            resolution=config.model.resolution,
            val_limit=10,  # Set a specific validation limit for testing
        )

        # Set val_limit to a small number
        val_limit = 50
        config.dataset.val_limit = val_limit
        args = config.to_args()

        # Build dataset with val_limit
        dataset_limited = build_dataset(image_set="val", args=args, resolution=args.resolution)
        print(f"Validation dataset size with val_limit={val_limit}: {len(dataset_limited)}")

        # Now build dataset without val_limit to compare
        config.dataset.val_limit = None
        args = config.to_args()
        dataset_full = build_dataset(image_set="val", args=args, resolution=args.resolution)
        print(f"Full validation dataset size: {len(dataset_full)}")

        # Assert that val_limit is working
        self.assertEqual(len(dataset_limited), val_limit)
        self.assertGreater(
            len(dataset_full),
            val_limit,
            "Full dataset should have more samples than limited dataset",
        )


if __name__ == "__main__":
    unittest.main()
