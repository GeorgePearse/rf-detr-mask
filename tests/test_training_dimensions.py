#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Test script to verify the training_width and training_height parameters work correctly.
"""

import unittest
from pathlib import Path

import torch
from PIL import Image

from rfdetr.config_utils import load_config
from rfdetr.datasets import build_dataset


class TestTrainingDimensions(unittest.TestCase):
    """Test case for training dimensions functionality."""
    
    def test_training_dimensions(self):
        """Test that the training dimensions are applied correctly."""
        # Load the default config
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        config = load_config(config_path)
        
        # Set training dimensions
        training_width = 560  # Divisible by 56
        training_height = 672  # Divisible by 56
        config.model.training_width = training_width
        config.model.training_height = training_height
        args = config.to_args()
        
        # Build dataset with custom training dimensions
        dataset_train = build_dataset(image_set="train", args=args, resolution=args.resolution)
        
        # Get the first sample and check its dimensions
        img, _ = dataset_train[0]
        self.assertEqual(img.shape[-2:], (training_height, training_width), 
                         f"Expected image shape (3, {training_height}, {training_width}), got {img.shape}")
        
        print(f"Successfully verified training dimensions: {training_width}x{training_height}")
        
    def test_missing_training_dimensions(self):
        """Test that an error is raised when training dimensions are missing."""
        # Create a minimal config dict with missing training dimensions
        config_dict = {
            "model": {
                "encoder": "dinov2_windowed_small",
                "out_feature_indexes": [2, 5, 8, 11],
                "projector_scale": ["P4"],
                "resolution": 560,
                # training_width and training_height deliberately omitted
                "group_detr": 13,
                "num_queries": 100,
                "num_select": 100,
                "two_stage": True
            },
            "training": {
                "batch_size": 1,
                "num_select": 100,
                "group_detr": 13,
                "output_dir": "test_output"
            },
            "dataset": {},
            "mask": {"enabled": True},
            "other": {}
        }
        
        # Test that config validation raises an exception for missing required fields
        from pydantic import ValidationError
        from rfdetr.config_utils import RFDETRConfig
        
        with self.assertRaises(ValidationError) as context:
            RFDETRConfig.model_validate(config_dict)
            
        error_str = str(context.exception)
        self.assertTrue("training_width" in error_str and "training_height" in error_str,
                       f"Expected exception about missing training dimensions, got: {error_str}")
        
        print("Successfully validated that training dimensions are required")


if __name__ == "__main__":
    unittest.main()