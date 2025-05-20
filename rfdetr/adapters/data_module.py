import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

import rfdetr.util.misc as utils
from rfdetr.adapters.dataset import CocoDetection
from rfdetr.adapters.config import DataConfig
import torchvision.transforms as transforms

def get_training_transforms(image_width: int, image_height: int):
    return transforms.Compose([
        transforms.ToTensor(),  
        transforms.Resize((image_width, image_height)),
    ])

def get_validation_transforms(image_width: int, image_height: int):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_width, image_height)),
    ])

class RFDETRDataModule(pl.LightningDataModule):
    """Lightning data module for RF-DETR-Mask."""

    def __init__(self, config: DataConfig):
        """Initialize the RF-DETR data module.

        Args:
            config: Configuration as a Pydantic model or compatible dict/object
        """
        super().__init__()
        self.num_workers = config.num_workers
        self.training_batch_size = config.batch_size
        self.training_num_workers = config .num_workers
        self.training_width = config.training.training_width
        self.training_height = config.training.training_height
    
        self.image_directory = config.training.image_directory
        self.training_annotation_file = config.training.training_annotation_file
        self.validation_annotation_file = config.validation.validation_annotation_file
        self.training_transforms = get_training_transforms(config.training.training_width, config.training.training_height)
        self.validation_transforms = get_validation_transforms(config.validation.validation_width, config.validation.validation_height)

    def setup(self):
        """Set up datasets for training and validation."""
        # We need to set up datasets for all stages
        self.dataset_train = CocoDetection(
            img_folder=self.image_directory,
            ann_file=self.training_annotation_file,
            transforms=self.training_transforms,
            test_limit=None,
        )
        self.dataset_val = CocoDetection(
            img_folder=self.image_directory,
            ann_file=self.validation_annotation_file,
            transforms=self.validation_transforms,
            test_limit=None,
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
