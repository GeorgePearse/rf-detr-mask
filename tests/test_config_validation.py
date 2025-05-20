#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Test to validate that the default YAML config properly instantiates a valid config object
and that all keys are used (no mismatches between config file and Pydantic models).
"""

import unittest
from pathlib import Path
from typing import Any, Dict, Set

import yaml

from rfdetr.adapters.config import RFDETRConfig
from rfdetr.util.logging_config import get_logger

logger = get_logger(__name__)


class TestConfigValidation(unittest.TestCase):
    """Test to validate config.yaml instantiation and key usage."""

    def test_default_yaml_instantiation(self):
        """Test that the default YAML config can be instantiated as a valid config object."""
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        self.assertTrue(config_path.exists(), f"Config file not found: {config_path}")

        # Load the config file
        config = RFDETRConfig.from_yaml(config_path)

        # Verify basic properties to ensure it loaded correctly
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.training)
        self.assertIsNotNone(config.data)
        self.assertIsNotNone(config.mask)
        self.assertIsNotNone(config.other)

        # Assert some specific values to confirm proper loading
        self.assertEqual(config.model.hidden_dim, 256)
        self.assertEqual(config.model.encoder, "dinov2_windowed_small")
        self.assertEqual(config.data.coco_path, "/home/georgepearse/data/cmr/annotations")
        self.assertEqual(config.mask.enabled, True)
        self.assertEqual(config.other.seed, 42)

    def test_all_yaml_keys_used(self):
        """Test that all keys in the YAML file are properly used in the config object."""
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

        # Read the raw YAML file
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)

        # Instantiate the config
        config = RFDETRConfig.from_yaml(config_path)

        # Convert the config back to a dict for comparison
        config_dict = config.model_dump()

        # Check all YAML keys are used in the config
        yaml_keys = self.extract_all_keys_from_dict(yaml_data)
        config_keys = self.extract_all_keys_from_dict(config_dict)

        # Find keys in YAML but not in config (indicates unused keys)
        unused_keys = yaml_keys - config_keys
        if unused_keys:
            logger.warning(f"Found keys in YAML not used in config: {unused_keys}")
        self.assertEqual(len(unused_keys), 0, f"Found unused keys in YAML: {unused_keys}")

        # Also check for keys in config not in YAML (might indicate default values used)
        missing_from_yaml = config_keys - yaml_keys
        if missing_from_yaml:
            logger.info(
                f"Keys in config not specified in YAML (using defaults): {missing_from_yaml}"
            )

    def test_transformer_config_consistency(self):
        """Test that transformer config values in model section match the nested transformer config."""
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

        # Read the raw YAML file
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)

        # Check transformer configuration consistency
        transformer_config = yaml_data["model"]["transformer"]
        model_config = yaml_data["model"]

        # Check for duplicate keys with different values
        for key in transformer_config:
            if key in model_config and key not in ["transformer"]:
                # If the same key exists at both levels, they should have the same value
                self.assertEqual(
                    transformer_config[key],
                    model_config[key],
                    f"Key '{key}' has different values in model ({model_config[key]}) and transformer ({transformer_config[key]})",
                )

    def extract_all_keys_from_dict(self, d: Dict[str, Any], prefix: str = "") -> Set[str]:
        """Extract all keys including nested ones from a dictionary."""
        keys = set()
        for k, v in d.items():
            key_path = f"{prefix}.{k}" if prefix else k
            keys.add(key_path)
            if isinstance(v, dict):
                keys.update(self.extract_all_keys_from_dict(v, key_path))
        return keys


if __name__ == "__main__":
    unittest.main()
