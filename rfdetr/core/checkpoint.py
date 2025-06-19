"""Checkpoint management for training.

This module provides a unified interface for saving and loading checkpoints,
managing best models, and handling checkpoint rotation.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer


class CheckpointManager:
    """Manages model checkpoints during training."""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best: bool = True,
        metric_name: str = "val_loss",
        metric_mode: str = "min",
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of regular checkpoints to keep
            save_best: Whether to save best checkpoint separately
            metric_name: Metric name to track for best checkpoint
            metric_mode: "min" or "max" for metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        self.best_metric = None
        self.checkpoint_history: List[Path] = []
        
        # Load checkpoint metadata if it exists
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self._load_metadata()
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Path:
        """Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optional optimizer state
            epoch: Current epoch
            step: Current step
            metrics: Current metrics
            config: Training configuration
            **kwargs: Additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint_data = {
            "model": model.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            **kwargs,
        }
        
        if optimizer is not None:
            checkpoint_data["optimizer"] = optimizer.state_dict()
        
        if config is not None:
            # Convert config to dict if it has a to_dict method
            if hasattr(config, "to_dict"):
                checkpoint_data["config"] = config.to_dict()
            elif hasattr(config, "__dict__"):
                checkpoint_data["config"] = vars(config)
            else:
                checkpoint_data["config"] = config
        
        # Generate checkpoint filename
        if step is not None:
            filename = f"checkpoint_step_{step:08d}.pth"
        elif epoch is not None:
            filename = f"checkpoint_epoch_{epoch:04d}.pth"
        else:
            filename = "checkpoint_latest.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Track checkpoint in history
        self.checkpoint_history.append(checkpoint_path)
        
        # Manage checkpoint rotation
        self._rotate_checkpoints()
        
        # Handle best checkpoint
        if self.save_best and metrics and self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            if self._is_best_metric(metric_value):
                self.best_metric = metric_value
                best_path = self.checkpoint_dir / "checkpoint_best.pth"
                shutil.copy2(checkpoint_path, best_path)
                print(f"Saved best checkpoint: {best_path} ({self.metric_name}={metric_value:.4f})")
        
        # Save metadata
        self._save_metadata()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Optional model to load state into
            optimizer: Optional optimizer to load state into
            map_location: Device mapping location
            
        Returns:
            Dictionary of checkpoint data
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Handle special checkpoint names
        if checkpoint_path.name == "latest":
            checkpoint_path = self.get_latest_checkpoint()
        elif checkpoint_path.name == "best":
            checkpoint_path = self.get_best_checkpoint()
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state if provided
        if model is not None and "model" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model"])
            print(f"Loaded model from: {checkpoint_path}")
        
        # Load optimizer state if provided
        if optimizer is not None and "optimizer" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer"])
            print(f"Loaded optimizer from: {checkpoint_path}")
        
        return checkpoint_data
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent checkpoint."""
        if not self.checkpoint_history:
            return None
        return self.checkpoint_history[-1]
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get the best checkpoint."""
        best_path = self.checkpoint_dir / "checkpoint_best.pth"
        if best_path.exists():
            return best_path
        return None
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
        return sorted(checkpoints)
    
    def _is_best_metric(self, metric_value: float) -> bool:
        """Check if the current metric is the best so far."""
        if self.best_metric is None:
            return True
        
        if self.metric_mode == "min":
            return metric_value < self.best_metric
        else:
            return metric_value > self.best_metric
    
    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints to maintain max_checkpoints limit."""
        # Don't count best checkpoint in rotation
        regular_checkpoints = [
            cp for cp in self.checkpoint_history
            if "best" not in cp.name
        ]
        
        while len(regular_checkpoints) > self.max_checkpoints:
            oldest = regular_checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()
                print(f"Removed old checkpoint: {oldest}")
            self.checkpoint_history.remove(oldest)
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata."""
        metadata = {
            "best_metric": self.best_metric,
            "metric_name": self.metric_name,
            "metric_mode": self.metric_mode,
            "checkpoint_history": [str(cp) for cp in self.checkpoint_history],
        }
        
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self) -> None:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)
            
            self.best_metric = metadata.get("best_metric")
            self.checkpoint_history = [
                Path(cp) for cp in metadata.get("checkpoint_history", [])
            ]
            
            # Clean up missing checkpoints from history
            self.checkpoint_history = [
                cp for cp in self.checkpoint_history if cp.exists()
            ]