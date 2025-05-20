#!/usr/bin/env python3

"""
Test script to verify that batch size and num_workers parameters
from config.py correctly propagate to data_module.py
"""

import unittest
from pathlib import Path

from rfdetr.adapters.config import DataConfig
from rfdetr.adapters.data_module import RFDETRDataModule


class TestDataModuleConfig(unittest.TestCase):
    """Test the configuration propagation between config.py and data_module.py."""
    
    def test_batch_size_and_num_workers_propagation(self):
        """Test that batch size and num_workers correctly propagate."""
        # Create a DataConfig with custom values
        test_config = DataConfig(
            training_batch_size=16,  # Custom batch size
            training_num_workers=8,  # Custom num workers
            validation_batch_size=32,
            validation_num_workers=4,
            training_width=560,
            training_height=560,
            image_directory="/tmp",
            training_annotation_file="/tmp/train.json",
            validation_annotation_file="/tmp/val.json",
        )
        
        # Create data module with this config
        data_module = RFDETRDataModule(test_config)
        
        # Verify that values were correctly propagated
        self.assertEqual(data_module.training_batch_size, 16)
        self.assertEqual(data_module.training_num_workers, 8)
        self.assertEqual(data_module.validation_batch_size, 32)
        self.assertEqual(data_module.validation_num_workers, 4)
        
        # Verify other settings were also correctly propagated
        self.assertEqual(data_module.training_width, 560)
        self.assertEqual(data_module.training_height, 560)
        self.assertEqual(data_module.image_directory, "/tmp")
        self.assertEqual(data_module.training_annotation_file, "/tmp/train.json")
        self.assertEqual(data_module.validation_annotation_file, "/tmp/val.json")
        
        print("All configuration parameters correctly propagated!")


if __name__ == "__main__":
    unittest.main()