"""
Tests the config validation system, ensuring YAML config files are consistent with Pydantic models.
"""

import unittest
from pathlib import Path

import yaml

from rfdetr.config_utils import (
    DatasetConfig,
    MaskConfig,
    ModelConfig,
    OtherConfig,
    RFDETRConfig,
    TrainingConfig,
)


class TestConfigValidation(unittest.TestCase):
    def test_yaml_fields_exist_in_pydantic_models(self):
        """
        Test that all fields in the YAML config file are defined in the Pydantic models.
        This ensures that we don't have fields in YAML that aren't represented in the model,
        which would cause them to be silently ignored.
        """
        # Path to the default config file
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

        # Load the YAML config
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Get all fields from Pydantic models in the sections dictionary below

        # Track all missing fields to provide a comprehensive report
        all_missing_fields = {}

        # Check fields for each section
        sections = {
            "model": set(ModelConfig.model_fields.keys()),
            "training": set(TrainingConfig.model_fields.keys()),
            "dataset": set(DatasetConfig.model_fields.keys()),
            "mask": set(MaskConfig.model_fields.keys()),
            "other": set(OtherConfig.model_fields.keys()),
        }

        # Check all sections
        for section, fields in sections.items():
            yaml_fields = set(config_data[section].keys())
            missing_fields = yaml_fields - fields
            if missing_fields:
                all_missing_fields[section] = missing_fields

        # Assert no missing fields and provide a detailed error message if there are any
        self.assertEqual(
            all_missing_fields,
            {},
            f"Fields in YAML missing from Pydantic models: {all_missing_fields}\n"
            f"These fields will be silently ignored when loading the config.",
        )

    def test_yaml_loads_without_errors(self):
        """
        Test that the default YAML config can be loaded into the Pydantic model without errors.
        """
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        try:
            config = RFDETRConfig.from_yaml(config_path)
            self.assertIsInstance(config, RFDETRConfig)
        except Exception as e:
            self.fail(f"Loading config failed with error: {e}")

    def test_default_and_base_configs_validate(self):
        """
        Test that the default.yaml and base_config.yaml files can be loaded and validated.
        These are the main configuration files that should be complete and valid.
        """
        # List of configs we want to test (the main ones that should be complete)
        config_files = [
            Path(__file__).parent.parent / "configs" / "default.yaml",
            Path(__file__).parent.parent / "configs" / "base_config.yaml",
        ]

        # Try to load and validate each config file
        for config_file in config_files:
            self.assertTrue(config_file.exists(), f"Config file {config_file.name} not found")
            try:
                config = RFDETRConfig.from_yaml(config_file)
                self.assertIsInstance(
                    config, RFDETRConfig, f"Config {config_file.name} didn't load properly"
                )
            except Exception as e:
                self.fail(f"Loading config {config_file.name} failed with error: {e}")

    def test_pydantic_model_fields_all_in_yaml(self):
        """
        Test that every field in the Pydantic models has a corresponding entry in the YAML.
        This helps detect fields that are defined in the model but missing from the config,
        which could lead to unexpected default values being used.
        """
        # Path to the default config file
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

        # Load the YAML config
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Get all Pydantic model fields - we'll use these to compare with YAML fields
        # Read model fields

        # Get all fields from YAML
        model_yaml_fields = set(config_data["model"].keys())
        training_yaml_fields = set(config_data["training"].keys())
        dataset_yaml_fields = set(config_data["dataset"].keys())
        mask_yaml_fields = set(config_data["mask"].keys())
        other_yaml_fields = set(config_data["other"].keys())

        # Track all fields in Pydantic but missing from YAML
        # Don't include optional fields with default values
        all_missing_from_yaml = {}

        # Get required fields for each model
        required_model_fields = {
            f for f, field in ModelConfig.model_fields.items() if field.is_required()
        }
        required_training_fields = {
            f for f, field in TrainingConfig.model_fields.items() if field.is_required()
        }
        required_dataset_fields = {
            f for f, field in DatasetConfig.model_fields.items() if field.is_required()
        }
        required_mask_fields = {
            f for f, field in MaskConfig.model_fields.items() if field.is_required()
        }
        required_other_fields = {
            f for f, field in OtherConfig.model_fields.items() if field.is_required()
        }

        # Check required fields that are missing from YAML
        missing_model_fields = required_model_fields - model_yaml_fields
        if missing_model_fields:
            all_missing_from_yaml["model"] = missing_model_fields

        missing_training_fields = required_training_fields - training_yaml_fields
        if missing_training_fields:
            all_missing_from_yaml["training"] = missing_training_fields

        missing_dataset_fields = required_dataset_fields - dataset_yaml_fields
        if missing_dataset_fields:
            all_missing_from_yaml["dataset"] = missing_dataset_fields

        missing_mask_fields = required_mask_fields - mask_yaml_fields
        if missing_mask_fields:
            all_missing_from_yaml["mask"] = missing_mask_fields

        missing_other_fields = required_other_fields - other_yaml_fields
        if missing_other_fields:
            all_missing_from_yaml["other"] = missing_other_fields

        # Assert no missing fields
        self.assertEqual(
            all_missing_from_yaml,
            {},
            f"Required fields in Pydantic model missing from YAML: {all_missing_from_yaml}",
        )

    def test_config_validation(self):
        """
        Test that the default YAML config can be loaded into the Pydantic model without errors.
        This helps ensure that the YAML values match the expected types in the Pydantic model.
        """
        # Path to the default config file
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

        # Load the YAML config
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Try to validate the config
        try:
            config = RFDETRConfig.model_validate(config_data)
            self.assertIsInstance(config, RFDETRConfig)
        except Exception as e:
            self.fail(f"Validation failed: {e}")


if __name__ == "__main__":
    unittest.main()
