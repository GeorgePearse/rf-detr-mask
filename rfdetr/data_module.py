import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset
from rfdetr.config import RFDETRConfig
import os

class RFDETRDataModule(pl.LightningDataModule):
    """Lightning data module for RF-DETR-Mask."""

    def __init__(self, config: RFDETRConfig):
        """Initialize the RF-DETR data module.

        Args:
            config: Configuration as a Pydantic model or compatible dict/object
        """
        super().__init__()
        
        self.config = RFDETRConfig(**config)
            
        # Extract configuration values with proper defaults
        # Handle dict type config
        self.batch_size = self.config.model.batch_size
        self.num_workers = self.config.model.num_workers
        self.training_width = self.config.model.training_width
        self.training_height = self.config.model.training_height
    

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
