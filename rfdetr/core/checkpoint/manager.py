"""Checkpoint management for fault tolerance and model versioning.

This module provides comprehensive checkpoint management including saving,
loading, versioning, and pruning strategies.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from rfdetr.config import ModelConfig


class CheckpointManager:
    """Manages model checkpoints with versioning and fault tolerance.
    
    Features:
    - Automatic checkpoint versioning
    - Resume training from failures
    - Checkpoint pruning strategies
    - Distributed checkpoint coordination
    - Metadata tracking
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        keep_last_n: int = 5,
        keep_best_n: int = 3,
        metric_name: str = "val_loss",
        metric_mode: str = "min",
        save_optimizer: bool = True,
        save_gradients: bool = False,
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            keep_best_n: Number of best checkpoints to keep
            metric_name: Metric name for best checkpoint selection
            metric_mode: "min" or "max" for metric comparison
            save_optimizer: Whether to save optimizer state
            save_gradients: Whether to save gradient information
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.save_optimizer = save_optimizer
        self.save_gradients = save_gradients
        
        # Track checkpoint history
        self.checkpoint_history: List[Dict[str, Any]] = []
        self._load_checkpoint_history()
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[ModelConfig] = None,
        **metadata: Any,
    ) -> Path:
        """Save a checkpoint with automatic versioning.
        
        Args:
            model: Model to save
            optimizer: Optional optimizer state
            epoch: Current epoch
            step: Current training step
            metrics: Current metrics
            config: Model configuration
            **metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch{epoch:04d}_step{step:08d}_{timestamp}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "step": step,
            "timestamp": timestamp,
        }
        
        # Add optimizer state if requested
        if self.save_optimizer and optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
        
        # Add metrics
        if metrics is not None:
            checkpoint_data["metrics"] = metrics
        
        # Add configuration
        if config is not None:
            checkpoint_data["config"] = config.dict()
        
        # Add gradient information if requested
        if self.save_gradients:
            grad_info = self._extract_gradient_info(model)
            checkpoint_data["gradient_info"] = grad_info
        
        # Add custom metadata
        checkpoint_data.update(metadata)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update checkpoint history
        history_entry = {
            "path": str(checkpoint_path),
            "epoch": epoch,
            "step": step,
            "timestamp": timestamp,
            "metrics": metrics or {},
        }
        self.checkpoint_history.append(history_entry)
        self._save_checkpoint_history()
        
        # Prune old checkpoints
        self._prune_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        map_location: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint. If None, loads latest.
            model: Optional model to load state into
            optimizer: Optional optimizer to load state into
            map_location: Device mapping location
            
        Returns:
            Checkpoint data dictionary
            
        Raises:
            FileNotFoundError: If checkpoint not found
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoints found")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state if provided
        if model is not None:
            model.load_state_dict(checkpoint_data["model_state_dict"])
        
        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        
        return checkpoint_data
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        if not self.checkpoint_history:
            return None
        
        # Sort by step (secondary) and epoch (primary)
        latest = max(self.checkpoint_history, key=lambda x: (x["epoch"], x["step"]))
        return Path(latest["path"])
    
    def get_best_checkpoint(self, metric_name: Optional[str] = None) -> Optional[Path]:
        """Get the path to the best checkpoint based on metric.
        
        Args:
            metric_name: Metric to use for selection. Uses default if None.
            
        Returns:
            Path to best checkpoint or None if no checkpoints exist
        """
        if not self.checkpoint_history:
            return None
        
        metric_name = metric_name or self.metric_name
        
        # Filter checkpoints with the metric
        valid_checkpoints = [
            cp for cp in self.checkpoint_history
            if metric_name in cp.get("metrics", {})
        ]
        
        if not valid_checkpoints:
            return None
        
        # Find best based on metric mode
        if self.metric_mode == "min":
            best = min(valid_checkpoints, key=lambda x: x["metrics"][metric_name])
        else:
            best = max(valid_checkpoints, key=lambda x: x["metrics"][metric_name])
        
        return Path(best["path"])
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata.
        
        Returns:
            List of checkpoint metadata dictionaries
        """
        return self.checkpoint_history.copy()
    
    def delete_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Delete a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint to delete
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Remove from history
        self.checkpoint_history = [
            cp for cp in self.checkpoint_history
            if Path(cp["path"]) != checkpoint_path
        ]
        self._save_checkpoint_history()
        
        # Delete file
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    
    def _prune_checkpoints(self) -> None:
        """Prune old checkpoints based on retention policies."""
        if not self.checkpoint_history:
            return
        
        # Sort by step and epoch
        sorted_checkpoints = sorted(
            self.checkpoint_history,
            key=lambda x: (x["epoch"], x["step"]),
            reverse=True
        )
        
        # Keep last N checkpoints
        to_keep = set()
        for cp in sorted_checkpoints[:self.keep_last_n]:
            to_keep.add(Path(cp["path"]))
        
        # Keep best N checkpoints
        if self.metric_name:
            valid_checkpoints = [
                cp for cp in sorted_checkpoints
                if self.metric_name in cp.get("metrics", {})
            ]
            
            if valid_checkpoints:
                # Sort by metric
                if self.metric_mode == "min":
                    best_checkpoints = sorted(
                        valid_checkpoints,
                        key=lambda x: x["metrics"][self.metric_name]
                    )[:self.keep_best_n]
                else:
                    best_checkpoints = sorted(
                        valid_checkpoints,
                        key=lambda x: x["metrics"][self.metric_name],
                        reverse=True
                    )[:self.keep_best_n]
                
                for cp in best_checkpoints:
                    to_keep.add(Path(cp["path"]))
        
        # Delete checkpoints not in keep set
        for cp in self.checkpoint_history:
            cp_path = Path(cp["path"])
            if cp_path not in to_keep and cp_path.exists():
                cp_path.unlink()
        
        # Update history
        self.checkpoint_history = [
            cp for cp in self.checkpoint_history
            if Path(cp["path"]) in to_keep
        ]
        self._save_checkpoint_history()
    
    def _extract_gradient_info(self, model: nn.Module) -> Dict[str, Any]:
        """Extract gradient statistics from model.
        
        Args:
            model: Model to extract gradients from
            
        Returns:
            Dictionary of gradient statistics
        """
        grad_info = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_info[name] = {
                    "mean": param.grad.mean().item(),
                    "std": param.grad.std().item(),
                    "min": param.grad.min().item(),
                    "max": param.grad.max().item(),
                    "norm": param.grad.norm().item(),
                }
        
        return grad_info
    
    def _load_checkpoint_history(self) -> None:
        """Load checkpoint history from disk."""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        
        if history_file.exists():
            with open(history_file, "r") as f:
                self.checkpoint_history = json.load(f)
        else:
            # Scan directory for existing checkpoints
            self.checkpoint_history = []
            for cp_path in self.checkpoint_dir.glob("checkpoint_*.pth"):
                # Try to extract metadata from filename
                parts = cp_path.stem.split("_")
                epoch = int(parts[1].replace("epoch", ""))
                step = int(parts[2].replace("step", ""))
                
                self.checkpoint_history.append({
                    "path": str(cp_path),
                    "epoch": epoch,
                    "step": step,
                    "metrics": {},
                })
    
    def _save_checkpoint_history(self) -> None:
        """Save checkpoint history to disk."""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        
        with open(history_file, "w") as f:
            json.dump(self.checkpoint_history, f, indent=2)