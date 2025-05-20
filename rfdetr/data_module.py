import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset
from rfdetr.model_config import ModelConfig

class RFDETRDataModule(pl.LightningDataModule):
    """Lightning data module for RF-DETR-Mask."""

    def __init__(self, config):
        """Initialize the RF-DETR data module.

        Args:
            config: Configuration as a Pydantic model or compatible dict/object
        """
        super().__init__()
        
        # Convert config to ModelConfig if it's not already a ModelConfig
        if isinstance(config, dict):
            try:
                self.config = ModelConfig(**config)
            except Exception:
                self.config = config  # Keep original if conversion fails
        else:
            self.config = config
            
        # Extract configuration values with proper defaults
        # Handle dict type config
        if isinstance(self.config, dict):
            self.batch_size = self.config.get("batch_size", 4)
            self.num_workers = self.config.get("num_workers", 2)
            self.training_width = self.config.get("training_width", 560)
            self.training_height = self.config.get("training_height", 560)
        else:
            # For structured config objects, check for training attributes first
            # Training parameters
            try:
                self.batch_size = self.config.training.batch_size
            except (AttributeError, KeyError):
                # Fallback to direct attribute or default
                try:
                    self.batch_size = self.config.batch_size
                except (AttributeError, KeyError):
                    self.batch_size = 4
                    
            try:
                self.num_workers = self.config.training.num_workers
            except (AttributeError, KeyError):
                try:
                    self.num_workers = self.config.num_workers
                except (AttributeError, KeyError):
                    self.num_workers = 2
                    
            # Model parameters
            try:
                self.training_width = self.config.model.training_width
            except (AttributeError, KeyError):
                try:
                    self.training_width = self.config.training_width
                except (AttributeError, KeyError):
                    self.training_width = 560
                    
            try:
                self.training_height = self.config.model.training_height
            except (AttributeError, KeyError):
                try:
                    self.training_height = self.config.training_height
                except (AttributeError, KeyError):
                    self.training_height = 560

    def setup(self, stage=None):
        """Set up datasets for training and validation."""
        # We need to set up datasets for all stages
        self.dataset_train = build_dataset(
            image_set="train",
            args=self.config,
            training_width=self.training_width,
            training_height=self.training_height,
        )
        self.dataset_val = build_dataset(
            image_set="val",
            args=self.config,
            training_width=self.training_width,
            training_height=self.training_height,
        )

    def train_dataloader(self):
        """Create training data loader."""
        if self.trainer.world_size > 1:
            sampler_train = DistributedSampler(self.dataset_train)
        else:
            sampler_train = RandomSampler(self.dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, self.batch_size, drop_last=True
        )

        dataloader_train = DataLoader(
            self.dataset_train,
            batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=self.num_workers,
        )

        return dataloader_train

    def val_dataloader(self):
        """Create validation data loader."""
        # WARNING: Keep validation sequential to ensure consistent evaluation results
        # Shuffling validation data would make metrics less stable between runs
        # DO NOT change this to RandomSampler unless specifically testing data randomization effects
        if self.trainer.world_size > 1:
            sampler_val = DistributedSampler(self.dataset_val, shuffle=False)
        else:
            sampler_val = SequentialSampler(self.dataset_val)

        dataloader_val = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=self.num_workers,
        )

        return dataloader_val
