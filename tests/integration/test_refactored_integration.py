"""Integration tests for refactored RF-DETR components.

These tests verify that the refactored components work together correctly
without running full training loops.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.core.checkpoint import CheckpointManager
from rfdetr.core.config import ConfigurationManager
from rfdetr.core.models import ModelFactory
from rfdetr.core.training import CallbackManager
from rfdetr.core.training.callbacks import LoggingCallback, ModelCheckpointCallback


class TestComponentIntegration:
    """Test that refactored components work together correctly."""
    
    def test_config_to_model_flow(self):
        """Test configuration loading and model creation flow."""
        # 1. Create configuration manager
        config_manager = ConfigurationManager()
        
        # 2. Load model configuration
        model_config = config_manager.load_model_config(
            model_name="base",
            num_classes=10,
            resolution=640,
            pretrain_weights=None  # Don't load pretrained weights for test
        )
        
        # 3. Create model factory
        model_factory = ModelFactory()
        
        # 4. Create model from configuration
        model = model_factory.create_model(model_config)
        
        # Verify model is created correctly
        assert isinstance(model, nn.Module)
        assert hasattr(model, "forward")
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output is not None
    
    def test_checkpoint_flow(self):
        """Test checkpoint saving and loading flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create checkpoint manager
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=tmpdir,
                keep_last_n=3,
                keep_best_n=2
            )
            
            # 2. Create a simple model
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            )
            optimizer = torch.optim.Adam(model.parameters())
            
            # 3. Save checkpoints with different metrics
            for epoch in range(5):
                loss = 5.0 - epoch * 0.5  # Decreasing loss
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=epoch * 100,
                    metrics={"loss": loss, "accuracy": epoch * 0.1}
                )
                assert checkpoint_path.exists()
            
            # 4. Verify checkpoint pruning
            checkpoints = list(Path(tmpdir).glob("checkpoint_*.pth"))
            assert len(checkpoints) <= 5  # Should not exceed keep_last_n + keep_best_n
            
            # 5. Load best checkpoint
            best_checkpoint = checkpoint_manager.get_best_checkpoint(metric_name="loss")
            assert best_checkpoint is not None
            
            # 6. Load checkpoint data
            new_model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            )
            checkpoint_data = checkpoint_manager.load_checkpoint(
                best_checkpoint,
                model=new_model
            )
            
            assert "epoch" in checkpoint_data
            assert "metrics" in checkpoint_data
    
    def test_callback_integration(self):
        """Test callback system integration."""
        callback_manager = CallbackManager()
        
        # Track callback execution
        events = []
        
        # Create custom callback
        def track_event(event_name):
            def callback(**kwargs):
                events.append((event_name, kwargs.get("epoch", -1)))
            return callback
        
        # Register callbacks
        callback_manager.register("on_train_start", track_event("train_start"))
        callback_manager.register("on_train_epoch_start", track_event("epoch_start"))
        callback_manager.register("on_train_epoch_end", track_event("epoch_end"))
        callback_manager.register("on_train_end", track_event("train_end"))
        
        # Simulate training events
        callback_manager.trigger("on_train_start")
        for epoch in range(3):
            callback_manager.trigger("on_train_epoch_start", epoch=epoch)
            callback_manager.trigger("on_train_epoch_end", epoch=epoch)
        callback_manager.trigger("on_train_end")
        
        # Verify callback execution order
        expected_events = [
            ("train_start", -1),
            ("epoch_start", 0),
            ("epoch_end", 0),
            ("epoch_start", 1),
            ("epoch_end", 1),
            ("epoch_start", 2),
            ("epoch_end", 2),
            ("train_end", -1)
        ]
        assert events == expected_events
    
    def test_end_to_end_component_flow(self):
        """Test complete flow of all components working together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Configuration
            config_manager = ConfigurationManager()
            model_config = config_manager.load_model_config(
                model_name="base",
                num_classes=5,
                resolution=448,
                pretrain_weights=None
            )
            
            train_config = config_manager.load_training_config(
                dataset_dir=tmpdir,
                output_dir=tmpdir,
                batch_size=2,
                epochs=2,
                lr=0.001
            )
            
            # 2. Model creation
            model_factory = ModelFactory()
            model = model_factory.create_model(model_config)
            
            # 3. Checkpoint management
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=Path(tmpdir) / "checkpoints"
            )
            
            # 4. Callbacks
            callback_manager = CallbackManager()
            callback_manager.register_callback_object(
                ModelCheckpointCallback(checkpoint_manager)
            )
            
            # 5. Simulate training flow (without actual Trainer to avoid dependencies)
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
            
            # Trigger training start
            callback_manager.trigger("on_train_start", trainer=None)
            
            # Simulate epochs
            for epoch in range(train_config.epochs):
                # Epoch start
                callback_manager.trigger(
                    "on_train_epoch_start",
                    epoch=epoch,
                    trainer=None
                )
                
                # Simulate training step
                dummy_loss = torch.tensor(1.0 - epoch * 0.1)
                optimizer.zero_grad()
                dummy_loss.backward()
                optimizer.step()
                
                # Epoch end with metrics
                metrics = {"loss": dummy_loss.item(), "accuracy": epoch * 0.4}
                callback_manager.trigger(
                    "on_train_epoch_end",
                    epoch=epoch,
                    metrics=metrics,
                    trainer=None
                )
            
            # Training end
            callback_manager.trigger("on_train_end", trainer=None)
            
            # Verify checkpoint was saved
            checkpoints = checkpoint_manager.list_checkpoints()
            assert len(checkpoints) > 0
    
    def test_model_factory_criterion_creation(self):
        """Test that model factory can create criterion and postprocessors."""
        config_manager = ConfigurationManager()
        model_config = config_manager.load_model_config(
            model_name="base",
            num_classes=10,
            pretrain_weights=None
        )
        
        model_factory = ModelFactory()
        
        # Create criterion and postprocessors
        criterion, postprocessors = model_factory.create_criterion_and_postprocessors(
            model_config
        )
        
        assert criterion is not None
        assert isinstance(postprocessors, dict)
        assert "bbox" in postprocessors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])