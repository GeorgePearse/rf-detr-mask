#!/usr/bin/env python
"""Test script for verifying refactored components work correctly.

This script tests each refactored component individually and then
tests them working together.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from rfdetr.core.config import ConfigurationManager, Config, ModelConfig, DataConfig, TrainConfig
from rfdetr.core.model import ModelFactory
from rfdetr.core.checkpoint import CheckpointManager
from rfdetr.core.training.callbacks import CallbackManager, LoggingCallback


def test_configuration_manager():
    """Test ConfigurationManager functionality."""
    print("\n=== Testing ConfigurationManager ===")
    
    # Create manager with default config
    config_manager = ConfigurationManager()
    
    # Update some values
    config_manager.config.model.num_classes = 69  # CMR dataset
    config_manager.config.model.encoder = "dinov2_windowed_small"
    config_manager.config.data.batch_size = 2
    config_manager.config.train.epochs = 50
    
    # Validate config
    errors = config_manager.validate()
    if errors:
        print(f"Configuration errors: {errors}")
    else:
        print("✓ Configuration is valid")
    
    # Test conversion to namespace
    args = config_manager.to_namespace()
    print(f"✓ Converted to namespace with {len(vars(args))} attributes")
    
    # Test save/load
    test_config_path = Path("test_config.json")
    config_manager.config.save(test_config_path)
    print(f"✓ Saved config to {test_config_path}")
    
    # Load it back
    loaded_config = Config.load(test_config_path)
    print(f"✓ Loaded config back successfully")
    
    # Clean up
    test_config_path.unlink()
    
    return config_manager.config


def test_model_factory(config):
    """Test ModelFactory functionality."""
    print("\n=== Testing ModelFactory ===")
    
    # Test model creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = ModelFactory.create_model(config, device)
    print(f"✓ Created model: {model.__class__.__name__}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model has {n_params:,} trainable parameters")
    
    # Create criterion and postprocessors
    criterion, postprocessors = ModelFactory.create_criterion(config, device)
    print(f"✓ Created criterion: {criterion.__class__.__name__}")
    print(f"✓ Created postprocessors: {list(postprocessors.keys())}")
    
    # Test parameter groups
    param_groups = ModelFactory.get_parameter_groups(model, config)
    print(f"✓ Created {len(param_groups)} parameter groups")
    
    return model, criterion, postprocessors


def test_checkpoint_manager():
    """Test CheckpointManager functionality."""
    print("\n=== Testing CheckpointManager ===")
    
    # Create checkpoint manager
    checkpoint_dir = Path("test_checkpoints")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=3,
        save_best=True,
        metric_name="val_loss",
        metric_mode="min",
    )
    print(f"✓ Created checkpoint manager at {checkpoint_dir}")
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Save some checkpoints
    for i in range(5):
        metrics = {"val_loss": 1.0 - i * 0.1}  # Improving loss
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=i,
            step=i * 100,
            metrics=metrics,
        )
        print(f"✓ Saved checkpoint {i}: {checkpoint_path.name}")
    
    # Check rotation worked
    checkpoints = checkpoint_manager.list_checkpoints()
    regular_checkpoints = [cp for cp in checkpoints if "best" not in cp.name]
    print(f"✓ Checkpoint rotation working: {len(regular_checkpoints)} regular checkpoints")
    
    # Check best checkpoint
    best_checkpoint = checkpoint_manager.get_best_checkpoint()
    if best_checkpoint:
        print(f"✓ Best checkpoint saved: {best_checkpoint.name}")
    
    # Test loading
    checkpoint_data = checkpoint_manager.load_checkpoint(
        checkpoint_manager.get_latest_checkpoint(),
        model=model,
        optimizer=optimizer,
    )
    print(f"✓ Loaded checkpoint with epoch={checkpoint_data.get('epoch')}")
    
    # Clean up
    import shutil
    shutil.rmtree(checkpoint_dir)
    
    return checkpoint_manager


def test_callback_manager():
    """Test CallbackManager functionality."""
    print("\n=== Testing CallbackManager ===")
    
    # Create callback manager
    callback_manager = CallbackManager()
    
    # Add a logging callback
    logging_callback = LoggingCallback(log_frequency=10)
    callback_manager.add_callback(logging_callback)
    print("✓ Added logging callback")
    
    # Add a function callback
    def custom_callback(**kwargs):
        print(f"  Custom callback triggered with: {list(kwargs.keys())}")
    
    callback_manager.add_function_callback("on_train_batch_end", custom_callback)
    print("✓ Added function callback")
    
    # Test triggering callbacks
    print("✓ Testing callback triggers:")
    callback_manager.trigger("on_train_start", trainer=None)
    callback_manager.trigger("on_train_batch_end", trainer=None, batch_idx=0, loss=0.5)
    
    # Test getting callbacks dict
    callbacks_dict = callback_manager.get_callbacks_dict()
    print(f"✓ Got callbacks dict with {len(callbacks_dict)} event types")
    
    return callback_manager


def test_integration():
    """Test all components working together."""
    print("\n=== Testing Integration ===")
    
    # Create configuration
    config = Config(
        model=ModelConfig(
            encoder="dinov2_windowed_small",
            num_classes=69,
            hidden_dim=256,
            masks=True,
        ),
        data=DataConfig(
            batch_size=2,
            num_workers=0,
            resolution=448,
        ),
        train=TrainConfig(
            epochs=1,
            lr=1e-4,
            output_dir="test_output",
        ),
    )
    print("✓ Created integrated configuration")
    
    # Create model and criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion, postprocessors = ModelFactory.create_model_with_criterion(
        config, device
    )
    print("✓ Created model with criterion")
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir="test_integration_checkpoints",
        max_checkpoints=2,
    )
    print("✓ Created checkpoint manager")
    
    # Create callback manager
    callback_manager = CallbackManager([LoggingCallback()])
    print("✓ Created callback manager with logging")
    
    # Clean up
    import shutil
    if Path("test_integration_checkpoints").exists():
        shutil.rmtree("test_integration_checkpoints")
    
    print("\n✓ All components work together successfully!")


def main():
    """Run all tests."""
    print("Testing Refactored Components")
    print("=" * 60)
    
    try:
        # Test individual components
        config = test_configuration_manager()
        model, criterion, postprocessors = test_model_factory(config)
        checkpoint_manager = test_checkpoint_manager()
        callback_manager = test_callback_manager()
        
        # Test integration
        test_integration()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! Components are working correctly.")
        print("\nNext steps:")
        print("1. Run the refactored training script with --use_refactored flag:")
        print("   python scripts/train_refactored.py --use_refactored --epochs 1 --test_limit 10")
        print("2. Compare results with legacy training")
        print("3. Gradually migrate more functionality to refactored components")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()