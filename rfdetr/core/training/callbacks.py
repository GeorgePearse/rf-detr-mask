"""Callback system for extensible training hooks.

This module provides a flexible callback system that allows
extending training behavior without modifying core logic.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional


class Callback(ABC):
    """Base class for training callbacks."""
    
    @abstractmethod
    def on_train_start(self, trainer: Any, **kwargs) -> None:
        """Called at the start of training."""
        pass
    
    @abstractmethod
    def on_train_end(self, trainer: Any, **kwargs) -> None:
        """Called at the end of training."""
        pass
    
    @abstractmethod
    def on_train_epoch_start(self, trainer: Any, epoch: int, **kwargs) -> None:
        """Called at the start of each epoch."""
        pass
    
    @abstractmethod
    def on_train_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_train_batch_start(self, trainer: Any, batch_idx: int, **kwargs) -> None:
        """Called at the start of each batch."""
        pass
    
    @abstractmethod
    def on_train_batch_end(self, trainer: Any, batch_idx: int, loss: float, **kwargs) -> None:
        """Called at the end of each batch."""
        pass
    
    @abstractmethod
    def on_validation_start(self, trainer: Any, **kwargs) -> None:
        """Called at the start of validation."""
        pass
    
    @abstractmethod
    def on_validation_end(self, trainer: Any, metrics: Dict[str, float], **kwargs) -> None:
        """Called at the end of validation."""
        pass


class CallbackManager:
    """Manages and orchestrates callbacks during training."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """Initialize callback manager.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []
        self._function_callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    def add_callback(self, callback: Callback) -> None:
        """Add a callback instance.
        
        Args:
            callback: Callback to add
        """
        self.callbacks.append(callback)
    
    def add_function_callback(self, event: str, func: Callable) -> None:
        """Add a function callback for a specific event.
        
        Args:
            event: Event name (e.g., "on_train_batch_start")
            func: Function to call
        """
        self._function_callbacks[event].append(func)
    
    def remove_callback(self, callback: Callback) -> None:
        """Remove a callback instance.
        
        Args:
            callback: Callback to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def trigger(self, event: str, **kwargs) -> None:
        """Trigger callbacks for a specific event.
        
        Args:
            event: Event name
            **kwargs: Arguments to pass to callbacks
        """
        # Trigger class-based callbacks
        for callback in self.callbacks:
            if hasattr(callback, event):
                method = getattr(callback, event)
                method(**kwargs)
        
        # Trigger function callbacks
        for func in self._function_callbacks.get(event, []):
            func(**kwargs)
    
    def get_callbacks_dict(self) -> Dict[str, List[Callable]]:
        """Get callbacks as a dictionary for compatibility.
        
        Returns:
            Dictionary mapping event names to callback lists
        """
        callbacks_dict = defaultdict(list)
        
        # Add class-based callbacks
        for callback in self.callbacks:
            for event in [
                "on_train_start",
                "on_train_end",
                "on_train_epoch_start",
                "on_train_epoch_end",
                "on_train_batch_start",
                "on_train_batch_end",
                "on_validation_start",
                "on_validation_end",
            ]:
                if hasattr(callback, event):
                    method = getattr(callback, event)
                    callbacks_dict[event].append(method)
        
        # Add function callbacks
        for event, funcs in self._function_callbacks.items():
            callbacks_dict[event].extend(funcs)
        
        return dict(callbacks_dict)


# Example callbacks

class LoggingCallback(Callback):
    """Simple logging callback."""
    
    def __init__(self, log_frequency: int = 50):
        """Initialize logging callback.
        
        Args:
            log_frequency: How often to log (in batches)
        """
        self.log_frequency = log_frequency
        self.batch_count = 0
    
    def on_train_start(self, trainer: Any, **kwargs) -> None:
        """Called at the start of training."""
        print("Training started")
    
    def on_train_end(self, trainer: Any, **kwargs) -> None:
        """Called at the end of training."""
        print("Training completed")
    
    def on_train_epoch_start(self, trainer: Any, epoch: int, **kwargs) -> None:
        """Called at the start of each epoch."""
        print(f"\nEpoch {epoch} started")
        self.batch_count = 0
    
    def on_train_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Called at the end of each epoch."""
        print(f"Epoch {epoch} completed")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    def on_train_batch_start(self, trainer: Any, batch_idx: int, **kwargs) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_train_batch_end(self, trainer: Any, batch_idx: int, loss: float, **kwargs) -> None:
        """Called at the end of each batch."""
        self.batch_count += 1
        if self.batch_count % self.log_frequency == 0:
            print(f"  Batch {batch_idx}: loss = {loss:.4f}")
    
    def on_validation_start(self, trainer: Any, **kwargs) -> None:
        """Called at the start of validation."""
        print("Validation started")
    
    def on_validation_end(self, trainer: Any, metrics: Dict[str, float], **kwargs) -> None:
        """Called at the end of validation."""
        print("Validation completed")


class TensorBoardCallback(Callback):
    """TensorBoard logging callback."""
    
    def __init__(self, log_dir: str = "tensorboard_logs"):
        """Initialize TensorBoard callback.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
    
    def on_train_start(self, trainer: Any, **kwargs) -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer: Any, **kwargs) -> None:
        """Called at the end of training."""
        self.writer.close()
    
    def on_train_epoch_start(self, trainer: Any, epoch: int, **kwargs) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_train_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Called at the end of each epoch."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"epoch/{key}", value, epoch)
    
    def on_train_batch_start(self, trainer: Any, batch_idx: int, **kwargs) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_train_batch_end(self, trainer: Any, batch_idx: int, loss: float, **kwargs) -> None:
        """Called at the end of each batch."""
        self.global_step += 1
        self.writer.add_scalar("batch/loss", loss, self.global_step)
    
    def on_validation_start(self, trainer: Any, **kwargs) -> None:
        """Called at the start of validation."""
        pass
    
    def on_validation_end(self, trainer: Any, metrics: Dict[str, float], **kwargs) -> None:
        """Called at the end of validation."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"val/{key}", value, self.global_step)