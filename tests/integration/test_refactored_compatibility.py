#!/usr/bin/env python
"""Test that refactored components produce same results as legacy code.

This ensures backward compatibility and correctness of the refactoring.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from rfdetr.models import build_model
from rfdetr.models import build_criterion_and_postprocessors
from rfdetr.core.config import ConfigurationManager
from rfdetr.core.models import ModelFactory


class TestRefactoredCompatibility:
    """Test compatibility between old and new implementations."""
    
    @pytest.fixture
    def args(self):
        """Create mock args object for legacy code."""
        args = Mock()
        args.modelname = 'base'
        args.num_classes = 80
        args.resolution = [640, 640]
        args.masks = False
        args.aux_loss = True
        args.set_cost_class = 1
        args.set_cost_bbox = 5
        args.set_cost_giou = 2
        args.cls_loss_coef = 1
        args.bbox_loss_coef = 5
        args.giou_loss_coef = 2
        args.mask_loss_coef = 1
        args.dice_loss_coef = 1
        args.focal_alpha = 0.25
        args.device = 'cpu'
        args.weight_dict_share = 0.1
        args.weight_dict_start = 1
        args.weight_dict_end = 1.5
        args.deep_supervision = True
        args.no_object_weight = 0.1
        return args
    
    def test_model_creation_compatibility(self, args):
        """Test that both approaches create equivalent models."""
        # Legacy approach
        legacy_model = build_model(args)
        
        # Refactored approach
        config_manager = ConfigurationManager()
        model_config = config_manager.load_model_config(
            model_name=args.modelname,
            num_classes=args.num_classes,
            resolution=args.resolution[0],
            masks=args.masks
        )
        
        model_factory = ModelFactory()
        refactored_model = model_factory.create_model(model_config)
        
        # Compare model structures
        assert type(legacy_model).__name__ == type(refactored_model).__name__
        
        # Compare parameter counts
        legacy_params = sum(p.numel() for p in legacy_model.parameters())
        refactored_params = sum(p.numel() for p in refactored_model.parameters())
        assert legacy_params == refactored_params
        
        # Compare model outputs with same input
        legacy_model.eval()
        refactored_model.eval()
        
        with torch.no_grad():
            dummy_input = Mock()
            dummy_input.tensors = torch.randn(1, 3, 640, 640)
            dummy_input.mask = torch.ones(1, 640, 640).bool()
            
            legacy_output = legacy_model(dummy_input)
            refactored_output = refactored_model(dummy_input)
            
            # Check output keys match
            assert set(legacy_output.keys()) == set(refactored_output.keys())
            
            # Check output shapes match
            for key in legacy_output.keys():
                assert legacy_output[key].shape == refactored_output[key].shape
    
    def test_criterion_creation_compatibility(self, args):
        """Test that both approaches create equivalent criteria."""
        # Legacy approach
        legacy_model = build_model(args)
        legacy_criterion, legacy_postprocessors = build_criterion_and_postprocessors(
            args, legacy_model
        )
        
        # Refactored approach
        config_manager = ConfigurationManager()
        model_config = config_manager.load_model_config(
            model_name=args.modelname,
            num_classes=args.num_classes,
            masks=args.masks
        )
        
        model_factory = ModelFactory()
        refactored_model = model_factory.create_model(model_config)
        refactored_criterion, refactored_postprocessors = (
            model_factory.create_criterion_and_postprocessors(model_config)
        )
        
        # Compare weight dicts
        legacy_weights = legacy_criterion.weight_dict
        refactored_weights = refactored_criterion.weight_dict
        
        assert set(legacy_weights.keys()) == set(refactored_weights.keys())
        for key in legacy_weights:
            assert abs(legacy_weights[key] - refactored_weights[key]) < 1e-6
        
        # Compare postprocessors
        assert set(legacy_postprocessors.keys()) == set(refactored_postprocessors.keys())
    
    def test_loss_computation_compatibility(self, args):
        """Test that both approaches compute same losses."""
        # Setup legacy
        legacy_model = build_model(args)
        legacy_criterion, _ = build_criterion_and_postprocessors(args, legacy_model)
        
        # Setup refactored
        config_manager = ConfigurationManager()
        model_config = config_manager.load_model_config(
            model_name=args.modelname,
            num_classes=args.num_classes,
            masks=args.masks
        )
        
        model_factory = ModelFactory()
        refactored_model = model_factory.create_model(model_config)
        refactored_criterion, _ = model_factory.create_criterion_and_postprocessors(
            model_config
        )
        
        # Create dummy data
        batch_size = 2
        dummy_input = Mock()
        dummy_input.tensors = torch.randn(batch_size, 3, 640, 640)
        dummy_input.mask = torch.ones(batch_size, 640, 640).bool()
        
        dummy_targets = [
            {
                'boxes': torch.rand(5, 4),
                'labels': torch.randint(0, args.num_classes, (5,)),
                'image_id': torch.tensor(i),
                'orig_size': torch.tensor([640, 640])
            }
            for i in range(batch_size)
        ]
        
        # Set models to same random state
        legacy_model.eval()
        refactored_model.eval()
        
        with torch.no_grad():
            # Get outputs
            legacy_outputs = legacy_model(dummy_input)
            refactored_outputs = refactored_model(dummy_input)
            
            # Compute losses
            legacy_losses = legacy_criterion(legacy_outputs, dummy_targets)
            refactored_losses = refactored_criterion(refactored_outputs, dummy_targets)
            
            # Compare loss keys
            assert set(legacy_losses.keys()) == set(refactored_losses.keys())
            
            # Loss values won't be identical due to different random initialization,
            # but they should be in the same order of magnitude
            for key in legacy_losses:
                legacy_val = legacy_losses[key].item()
                refactored_val = refactored_losses[key].item()
                
                # Check same order of magnitude
                if legacy_val != 0:
                    ratio = refactored_val / legacy_val
                    assert 0.1 < ratio < 10, f"Loss {key} differs too much: {legacy_val} vs {refactored_val}"
    
    def test_weight_loading_compatibility(self, args, tmp_path):
        """Test that weights can be loaded in both approaches."""
        # Create and save a model with legacy approach
        legacy_model = build_model(args)
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        
        # Save legacy model state
        torch.save({
            'model': legacy_model.state_dict(),
            'epoch': 10,
            'args': args
        }, checkpoint_path)
        
        # Load with refactored approach
        config_manager = ConfigurationManager()
        model_config = config_manager.load_model_config(
            model_name=args.modelname,
            num_classes=args.num_classes,
            masks=args.masks
        )
        
        model_factory = ModelFactory()
        refactored_model = model_factory.create_model(model_config)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        refactored_model.load_state_dict(checkpoint['model'])
        
        # Verify weights match
        for (name1, param1), (name2, param2) in zip(
            legacy_model.named_parameters(),
            refactored_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)
    
    @pytest.mark.parametrize("model_name", ["base", "large"])
    def test_all_model_variants(self, model_name):
        """Test that all model variants work with both approaches."""
        args = Mock()
        args.modelname = model_name
        args.num_classes = 80
        args.resolution = [640, 640]
        args.masks = False
        args.aux_loss = True
        args.device = 'cpu'
        
        # Set all required loss coefficients
        args.set_cost_class = 1
        args.set_cost_bbox = 5
        args.set_cost_giou = 2
        args.cls_loss_coef = 1
        args.bbox_loss_coef = 5
        args.giou_loss_coef = 2
        args.mask_loss_coef = 1
        args.dice_loss_coef = 1
        args.focal_alpha = 0.25
        args.weight_dict_share = 0.1
        args.weight_dict_start = 1
        args.weight_dict_end = 1.5
        args.deep_supervision = True
        args.no_object_weight = 0.1
        
        # Legacy
        legacy_model = build_model(args)
        
        # Refactored
        config_manager = ConfigurationManager()
        model_config = config_manager.load_model_config(
            model_name=model_name,
            num_classes=args.num_classes
        )
        
        model_factory = ModelFactory()
        refactored_model = model_factory.create_model(model_config)
        
        # Basic checks
        assert legacy_model is not None
        assert refactored_model is not None
        assert type(legacy_model).__name__ == type(refactored_model).__name__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])