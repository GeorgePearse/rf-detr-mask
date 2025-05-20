from typing import Optional

import lightning.pytorch as pl

import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

import rfdetr.util.misc as utils
from rfdetr.adapters.config import DataConfig
from rfdetr.adapters.dataset import CocoDetection
from rfdetr.util.logging_config import get_logger
import os

logger = get_logger(__name__)


def get_training_transforms(image_width: int, image_height: int, mask_enabled: bool = True) -> A.Compose:
    """Get training transforms using Albumentations.
    
    These transforms include data augmentation suitable for training.
    
    Args:
        image_width: Target image width
        image_height: Target image height
        mask_enabled: Whether to include mask transformations
        
    Returns:
        Albumentations composition of transforms
    """
    transform_list = [
        A.Resize(height=image_height, width=image_width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    
    # Create proper transform parameters including mask handling if needed
    transform_params = {
        'bbox_params': A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
    }
    
    # Add mask handling if enabled
    if mask_enabled:
        transform_params['mask_params'] = A.MaskParams(format='mask')
    
    return A.Compose(transform_list, **transform_params)


def get_validation_transforms(image_width: int, image_height: int, mask_enabled: bool = True) -> A.Compose:
    """Get validation transforms using Albumentations.
    
    These are minimal transforms without augmentation, suitable for validation.
    
    Args:
        image_width: Target image width
        image_height: Target image height
        mask_enabled: Whether to include mask transformations
        
    Returns:
        Albumentations composition of transforms
    """
    transform_list = [
        A.Resize(height=image_height, width=image_width),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    
    # Always use the same Compose parameters, we'll handle masks separately in the dataset
    return A.Compose(
        transform_list,
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
    )


class RFDETRDataModule(pl.LightningDataModule):
    """Lightning data module for RF-DETR-Mask."""

    def __init__(self, config: DataConfig):
        """Initialize the RF-DETR data module.

        Args:
            config: Configuration as a Pydantic model or compatible dict/object
        """
        super().__init__()
        # Core batch size and workers settings
        self.training_batch_size = config.training_batch_size
        self.training_num_workers = config.training_num_workers
        self.validation_batch_size = config.validation_batch_size
        self.validation_num_workers = config.validation_num_workers

        # Image dimensions from model config
        self.training_width = config.input_training_width
        self.training_height = config.input_training_height

        # Data paths
        self.image_directory = config.image_directory
        self.annotation_directory = config.annotation_directory
        self.training_annotation_file = config.training_annotation_file
        self.validation_annotation_file = config.validation_annotation_file

        # Check if mask is enabled from config
        self.mask_enabled = getattr(config, 'mask_enabled', True)
        
        # Initialize transforms
        self.training_transforms = get_training_transforms(
            self.training_width, self.training_height, self.mask_enabled
        )
        self.validation_transforms = get_validation_transforms(
            self.training_width, self.training_height, self.mask_enabled
        )

        # Initialize datasets to None
        self.dataset_train: Optional[CocoDetection] = None
        self.dataset_val: Optional[CocoDetection] = None

        # Log paths for debugging
        logger.info(f"Image directory: {self.image_directory}")
        logger.info(f"Training annotation file: {self.training_annotation_file}")
        logger.info(f"Validation annotation file: {self.validation_annotation_file}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training and validation.

        Args:
            stage: Optional stage parameter required by Lightning, but not used here
        """
        # We need to set up datasets for all stages
        logger.info(
            f"Setting up training dataset with annotation file: {self.training_annotation_file}"
        )
        annotation_file = os.path.join(self.annotation_directory, self.training_annotation_file)
        assert os.path.exists(annotation_file), f"Annotation file does not exist: {annotation_file}"
        self.dataset_train = CocoDetection(
            img_folder=self.image_directory,
            ann_file=annotation_file,
            transforms=self.training_transforms,
            test_limit=None,
        )
        logger.info(f"Training dataset has {len(self.dataset_train)} samples")
        logger.info(
            f"Setting up validation dataset with annotation file: {self.validation_annotation_file}"
        )
        annotation_file = os.path.join(self.annotation_directory, self.validation_annotation_file)
        assert os.path.exists(annotation_file), f"Annotation file does not exist: {annotation_file}"
        self.dataset_val = CocoDetection(
            img_folder=self.image_directory,
            ann_file=annotation_file,
            transforms=self.validation_transforms,
            test_limit=None,
        )
        logger.info(f"Validation dataset has {len(self.dataset_val)} samples")

    def train_dataloader(self) -> DataLoader:
        """Create training data loader.

        Returns:
            DataLoader: The training data loader
        """
        if self.trainer is not None and self.trainer.world_size > 1:
            sampler_train: torch.utils.data.Sampler = DistributedSampler(self.dataset_train)
        else:
            sampler_train = RandomSampler(self.dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, self.training_batch_size, drop_last=True
        )

        dataloader_train = DataLoader(
            self.dataset_train,
            batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=self.training_num_workers,
        )

        return dataloader_train

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader.

        Returns:
            DataLoader: The validation data loader
        """
        # WARNING: Keep validation sequential to ensure consistent evaluation results
        # Shuffling validation data would make metrics less stable between runs
        # DO NOT change this to RandomSampler unless specifically testing data randomization effects
        if self.trainer is not None and self.trainer.world_size > 1:
            sampler_val: torch.utils.data.Sampler = DistributedSampler(
                self.dataset_val, shuffle=False
            )
        else:
            sampler_val = SequentialSampler(self.dataset_val)

        dataloader_val = DataLoader(
            self.dataset_val,
            batch_size=self.validation_batch_size,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=self.validation_num_workers,
        )

        return dataloader_val
