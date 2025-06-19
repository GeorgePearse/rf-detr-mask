"""Main training orchestrator with dependency injection.

This module provides the refactored Trainer class that separates training
orchestration from model management and other concerns.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from rfdetr.config import TrainConfig
from rfdetr.core.checkpoint import CheckpointManager
from rfdetr.core.training.callbacks import CallbackManager
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.engine import train_one_epoch, evaluate
# Early stopping is implemented inline below


class EarlyStopping:
    """Simple early stopping implementation."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
    
    def should_stop(self, current_value: float) -> bool:
        """Check if training should stop.
        
        Args:
            current_value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class MetricsTracker(Protocol):
    """Protocol for metrics tracking."""
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None: ...
    def flush(self) -> None: ...


class DataLoaderFactory(Protocol):
    """Protocol for data loader creation."""
    
    def create_train_loader(self, config: TrainConfig) -> DataLoader: ...
    def create_val_loader(self, config: TrainConfig) -> DataLoader: ...


class OptimizerFactory(Protocol):
    """Protocol for optimizer creation."""
    
    def create_optimizer(self, model: nn.Module, config: TrainConfig) -> Optimizer: ...
    def create_scheduler(self, optimizer: Optimizer, config: TrainConfig) -> Any: ...


class Trainer:
    """Main training orchestrator using dependency injection.
    
    This class coordinates the training process without directly handling
    model creation, data loading, or other concerns.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        data_loader_factory: DataLoaderFactory,
        optimizer_factory: OptimizerFactory,
        checkpoint_manager: CheckpointManager,
        metrics_tracker: Optional[MetricsTracker] = None,
        callback_manager: Optional[CallbackManager] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize the trainer with injected dependencies.
        
        Args:
            model: Model to train
            criterion: Loss criterion
            data_loader_factory: Factory for creating data loaders
            optimizer_factory: Factory for creating optimizers
            checkpoint_manager: Checkpoint management
            metrics_tracker: Optional metrics tracking
            callback_manager: Optional callback management
            device: Device to train on
        """
        self.model = model
        self.criterion = criterion
        self.data_loader_factory = data_loader_factory
        self.optimizer_factory = optimizer_factory
        self.checkpoint_manager = checkpoint_manager
        self.metrics_tracker = metrics_tracker
        self.callback_manager = callback_manager or CallbackManager()
        
        # Set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.early_stopping = None
    
    def train(
        self,
        config: TrainConfig,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Train the model according to configuration.
        
        Args:
            config: Training configuration
            resume_from: Optional checkpoint to resume from
            
        Returns:
            Dictionary of training results and metrics
        """
        # Create data loaders
        train_loader = self.data_loader_factory.create_train_loader(config)
        val_loader = self.data_loader_factory.create_val_loader(config)
        
        # Create optimizer and scheduler
        optimizer = self.optimizer_factory.create_optimizer(self.model, config)
        lr_scheduler = self.optimizer_factory.create_scheduler(optimizer, config)
        
        # Setup early stopping if enabled
        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
            )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(
                resume_from,
                model=self.model,
                optimizer=optimizer,
                map_location=self.device,
            )
            start_epoch = checkpoint_data.get("epoch", 0) + 1
            self.global_step = checkpoint_data.get("step", 0)
        
        # Training loop
        training_start_time = time.time()
        
        for epoch in range(start_epoch, config.epochs):
            self.current_epoch = epoch
            
            # Trigger epoch start callbacks
            self.callback_manager.trigger(
                "on_train_epoch_start",
                trainer=self,
                epoch=epoch,
                config=config,
            )
            
            # Train one epoch
            epoch_metrics = self._train_epoch(
                train_loader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                config=config,
                epoch=epoch,
            )
            
            # Validation
            if epoch % config.val_frequency == 0:
                val_metrics = self._validate(
                    val_loader=val_loader,
                    config=config,
                    epoch=epoch,
                )
                epoch_metrics.update(val_metrics)
            
            # Log metrics
            if self.metrics_tracker:
                self.metrics_tracker.log_metrics(epoch_metrics, self.global_step)
            
            # Save checkpoint
            if epoch % config.checkpoint_frequency == 0:
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=self.global_step,
                    metrics=epoch_metrics,
                    config=config,
                )
            
            # Check early stopping
            if self.early_stopping and "val_loss" in epoch_metrics:
                if self.early_stopping.should_stop(epoch_metrics["val_loss"]):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Trigger epoch end callbacks
            self.callback_manager.trigger(
                "on_train_epoch_end",
                trainer=self,
                epoch=epoch,
                metrics=epoch_metrics,
                config=config,
            )
        
        # Training complete
        training_time = time.time() - training_start_time
        
        # Save final checkpoint
        final_checkpoint = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=optimizer,
            epoch=self.current_epoch,
            step=self.global_step,
            metrics=epoch_metrics,
            config=config,
            training_time=training_time,
        )
        
        # Trigger training end callbacks
        self.callback_manager.trigger(
            "on_train_end",
            trainer=self,
            config=config,
            training_time=training_time,
        )
        
        # Flush metrics
        if self.metrics_tracker:
            self.metrics_tracker.flush()
        
        return {
            "final_checkpoint": str(final_checkpoint),
            "best_checkpoint": str(self.checkpoint_manager.get_best_checkpoint()),
            "training_time": training_time,
            "epochs_trained": self.current_epoch + 1,
            "final_metrics": epoch_metrics,
        }
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        lr_scheduler: Any,
        config: TrainConfig,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
            config: Training configuration
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Use the existing train_one_epoch function
        # This maintains compatibility while we refactor
        train_metrics = train_one_epoch(
            model=self.model,
            criterion=self.criterion,
            lr_scheduler=lr_scheduler,
            data_loader=train_loader,
            optimizer=optimizer,
            device=self.device,
            epoch=epoch,
            batch_size=config.batch_size,
            max_norm=config.clip_max_norm,
            args=config,  # Pass config as args for compatibility
            callbacks=self.callback_manager.get_callbacks_dict(),
        )
        
        # Update global step
        self.global_step += len(train_loader)
        
        return train_metrics
    
    def _validate(
        self,
        val_loader: DataLoader,
        config: TrainConfig,
        epoch: int,
    ) -> Dict[str, float]:
        """Run validation.
        
        Args:
            val_loader: Validation data loader
            config: Training configuration
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        # Create evaluator
        # Note: This currently assumes COCO dataset
        # TODO: Make this more generic through dependency injection
        from rfdetr.datasets.coco import build_evaluator
        
        base_ds = val_loader.dataset
        if hasattr(base_ds, "dataset"):
            base_ds = base_ds.dataset
        
        postprocessors = self.postprocessors  # TODO: Inject this
        evaluator = build_evaluator(config, base_ds, postprocessors)
        
        # Run evaluation
        val_metrics = evaluate(
            model=self.model,
            criterion=self.criterion,
            postprocessors=postprocessors,
            data_loader=val_loader,
            evaluator=evaluator,
            device=self.device,
            args=config,  # Pass config as args for compatibility
        )
        
        # Prefix metrics with "val_"
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        
        return val_metrics