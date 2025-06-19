"""Example integration of refactored components.

This module demonstrates how the refactored components work together
to replace the god object pattern.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rfdetr.config import TrainConfig
from rfdetr.core.checkpoint import CheckpointManager
from rfdetr.core.config import ConfigurationManager
from rfdetr.core.models import ModelFactory
from rfdetr.core.training import Trainer, CallbackManager
from rfdetr.core.training.callbacks import LoggingCallback, ModelCheckpointCallback


class SimpleDataLoaderFactory:
    """Simple data loader factory for demonstration."""
    
    def create_train_loader(self, config: TrainConfig) -> DataLoader:
        """Create training data loader."""
        # In real implementation, this would create actual data loaders
        # from config.dataset_dir, config.dataset_file, etc.
        pass
    
    def create_val_loader(self, config: TrainConfig) -> DataLoader:
        """Create validation data loader."""
        pass


class SimpleOptimizerFactory:
    """Simple optimizer factory for demonstration."""
    
    def create_optimizer(self, model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
        """Create optimizer from configuration."""
        # Group parameters for different learning rates
        param_groups = [
            {"params": model.parameters(), "lr": config.lr}
        ]
        
        return torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer, config: TrainConfig) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler."""
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_drop,
            gamma=0.1,
        )


class SimpleMetricsTracker:
    """Simple metrics tracker for demonstration."""
    
    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log metrics."""
        print(f"Step {step}: {metrics}")
    
    def flush(self) -> None:
        """Flush any pending metrics."""
        pass


def main():
    """Demonstrate the refactored architecture."""
    
    # 1. Configuration Management
    config_manager = ConfigurationManager()
    
    # Load model configuration with overrides
    model_config = config_manager.load_model_config(
        model_name="base",
        num_classes=80,  # Custom number of classes
        resolution=640,
    )
    
    # Load training configuration
    train_config = config_manager.load_training_config(
        dataset_dir="/path/to/dataset",
        output_dir="/path/to/output",
        batch_size=8,
        epochs=50,
        early_stopping=True,
    )
    
    # Validate compatibility
    config_manager.validate_config_compatibility(model_config, train_config)
    
    # 2. Model Creation
    model_factory = ModelFactory()
    
    # Create model with dependency injection
    model = model_factory.create_model(model_config)
    criterion, postprocessors = model_factory.create_criterion_and_postprocessors(model_config)
    
    # 3. Checkpoint Management
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Path(train_config.output_dir) / "checkpoints",
        keep_last_n=5,
        keep_best_n=3,
        metric_name="val_loss",
        metric_mode="min",
    )
    
    # 4. Callback Setup
    callback_manager = CallbackManager()
    
    # Register callbacks
    callback_manager.register_callback_object(LoggingCallback())
    callback_manager.register_callback_object(
        ModelCheckpointCallback(
            checkpoint_manager=checkpoint_manager,
            save_freq=train_config.checkpoint_interval,
        )
    )
    
    # 5. Training Orchestration
    trainer = Trainer(
        model=model,
        criterion=criterion,
        data_loader_factory=SimpleDataLoaderFactory(),
        optimizer_factory=SimpleOptimizerFactory(),
        checkpoint_manager=checkpoint_manager,
        metrics_tracker=SimpleMetricsTracker(),
        callback_manager=callback_manager,
    )
    
    # Train the model
    results = trainer.train(
        config=train_config,
        resume_from=None,  # Or path to checkpoint
    )
    
    print(f"Training completed: {results}")
    
    # 6. Export trained model
    if results["best_checkpoint"]:
        # Load best checkpoint
        checkpoint_data = checkpoint_manager.load_checkpoint(
            results["best_checkpoint"],
            model=model,
        )
        
        # Export model (ONNX, TorchScript, etc.)
        # model_exporter.export(model, format="onnx", path="model.onnx")


if __name__ == "__main__":
    # This is just a demonstration of the architecture
    print("Refactored RF-DETR Architecture Components:")
    print("1. ConfigurationManager - Unified config management")
    print("2. ModelFactory - Model creation with DI")
    print("3. CheckpointManager - Fault-tolerant checkpointing")
    print("4. Trainer - Clean training orchestration")
    print("5. CallbackManager - Extensible callback system")
    print("\nThis replaces the god object pattern with clean separation of concerns.")