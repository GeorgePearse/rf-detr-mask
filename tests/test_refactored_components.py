"""Tests for refactored components."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.core.checkpoint import CheckpointManager
from rfdetr.core.config import ConfigurationManager
from rfdetr.core.models import ModelFactory
from rfdetr.core.training import CallbackManager


class TestConfigurationManager:
    """Test the unified configuration management system."""
    
    def test_load_model_config(self):
        """Test loading model configuration."""
        config_manager = ConfigurationManager()
        
        # Test loading base config
        config = config_manager.load_model_config(model_name="base")
        assert isinstance(config, RFDETRBaseConfig)
        assert config.num_classes == 91  # Default COCO classes
        
        # Test with overrides
        config = config_manager.load_model_config(
            model_name="base",
            num_classes=10,
            resolution=512
        )
        assert config.num_classes == 10
        assert config.resolution == 512
    
    def test_load_training_config(self):
        """Test loading training configuration."""
        config_manager = ConfigurationManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = config_manager.load_training_config(
                dataset_dir=tmpdir,
                output_dir=tmpdir,
                batch_size=4,
                epochs=2
            )
            assert isinstance(config, TrainConfig)
            assert config.batch_size == 4
            assert config.epochs == 2
    
    def test_config_compatibility_validation(self):
        """Test configuration compatibility validation."""
        config_manager = ConfigurationManager()
        
        model_config = config_manager.load_model_config(
            model_name="base",
            resolution=640
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_config = config_manager.load_training_config(
                dataset_dir=tmpdir,
                output_dir=tmpdir,
                batch_size=8,
                grad_accum_steps=2
            )
        
        # Should not raise
        config_manager.validate_config_compatibility(model_config, train_config)
        
        # Test incompatible batch size
        train_config.batch_size = 7  # Not divisible by grad_accum_steps
        with pytest.raises(ValueError, match="must be divisible"):
            config_manager.validate_config_compatibility(model_config, train_config)


class TestModelFactory:
    """Test the model factory."""
    
    def test_create_model(self):
        """Test model creation."""
        factory = ModelFactory()
        config = RFDETRBaseConfig()
        
        # Create model without pretrained weights
        config.pretrain_weights = None
        model = factory.create_model(config)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, "forward")
    
    def test_checkpoint_saving_loading(self):
        """Test checkpoint save/load functionality."""
        factory = ModelFactory()
        
        # Create a simple model for testing
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pth"
            
            # Save checkpoint
            factory.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=5,
                path=checkpoint_path,
                metrics={"loss": 0.5}
            )
            
            assert checkpoint_path.exists()
            
            # Load checkpoint
            new_model = nn.Linear(10, 5)
            new_optimizer = torch.optim.Adam(new_model.parameters())
            
            factory.load_pretrained_weights(
                new_model,
                str(checkpoint_path),
                strict=True
            )


class TestCheckpointManager:
    """Test the checkpoint manager."""
    
    def test_checkpoint_lifecycle(self):
        """Test checkpoint save, load, and pruning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                keep_last_n=2,
                keep_best_n=1,
                metric_name="loss",
                metric_mode="min"
            )
            
            # Create a simple model
            model = nn.Linear(10, 5)
            optimizer = torch.optim.Adam(model.parameters())
            
            # Save multiple checkpoints
            checkpoints = []
            for epoch in range(5):
                loss = 5 - epoch  # Decreasing loss
                path = manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=epoch * 100,
                    metrics={"loss": loss}
                )
                checkpoints.append(path)
            
            # Check that old checkpoints were pruned
            existing_checkpoints = list(Path(tmpdir).glob("checkpoint_*.pth"))
            assert len(existing_checkpoints) <= 3  # keep_last_n + keep_best_n
            
            # Test getting latest checkpoint
            latest = manager.get_latest_checkpoint()
            assert latest is not None
            assert "epoch0004" in str(latest)
            
            # Test getting best checkpoint
            best = manager.get_best_checkpoint()
            assert best is not None
            
            # Test loading checkpoint
            checkpoint_data = manager.load_checkpoint(latest)
            assert checkpoint_data["epoch"] == 4
            assert checkpoint_data["metrics"]["loss"] == 1.0


class TestCallbackManager:
    """Test the callback system."""
    
    def test_callback_registration_and_triggering(self):
        """Test callback registration and event triggering."""
        manager = CallbackManager()
        
        # Track callback executions
        executions = []
        
        def callback1(**kwargs):
            executions.append(("callback1", kwargs.get("epoch")))
        
        def callback2(**kwargs):
            executions.append(("callback2", kwargs.get("epoch")))
        
        # Register callbacks
        manager.register("on_train_epoch_start", callback1)
        manager.register("on_train_epoch_start", callback2, priority=10)
        
        # Trigger event
        manager.trigger("on_train_epoch_start", epoch=1)
        
        # Check execution order (higher priority first)
        assert len(executions) == 2
        assert executions[0] == ("callback2", 1)  # Higher priority
        assert executions[1] == ("callback1", 1)
    
    def test_callback_error_handling(self):
        """Test that callback errors don't stop execution."""
        manager = CallbackManager()
        
        executions = []
        
        def failing_callback(**kwargs):
            raise ValueError("Test error")
        
        def working_callback(**kwargs):
            executions.append("worked")
        
        manager.register("on_train_end", failing_callback)
        manager.register("on_train_end", working_callback)
        
        # Should not raise, but continue with other callbacks
        manager.trigger("on_train_end")
        
        assert len(executions) == 1
        assert executions[0] == "worked"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])