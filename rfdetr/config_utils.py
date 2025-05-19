# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pathlib import Path
from typing import Literal, Optional, Union

import torch
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from rfdetr.util.error_handling import ConfigurationError
from rfdetr.util.logging_config import get_logger

logger = get_logger(__name__)

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class ModelConfig(BaseModel):
    """Configuration for the RF-DETR model."""

    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"]
    out_feature_indexes: list[int]
    dec_layers: int = Field(default=3, ge=1)
    projector_scale: list[Literal["P3", "P4", "P5", "P6"]]
    hidden_dim: int = Field(default=256, gt=0)
    sa_nheads: int = Field(default=8, gt=0)
    ca_nheads: int = Field(default=16, gt=0)
    dec_n_points: int = Field(default=2, gt=0)
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    num_classes: int = Field(default=90, gt=0)
    pretrain_weights: Optional[str] = None
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    resolution: int = Field(default=560, gt=0)
    group_detr: int = Field(default=13, gt=0)
    gradient_checkpointing: bool = False
    num_queries: int = Field(default=300, gt=0)
    num_select: int = Field(default=300, gt=0)

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v):
        """Validate that resolution is divisible by 14 for DINOv2."""
        if v % 14 != 0:
            error_msg = f"Resolution {v} must be divisible by 14 for DINOv2"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        return v


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""

    lr: float = Field(default=1e-4, gt=0)
    lr_encoder: float = Field(default=1.5e-4, gt=0)
    batch_size: int = Field(default=4, gt=0)
    grad_accum_steps: int = Field(default=4, gt=0)
    epochs: int = Field(default=100, gt=0)
    ema_decay: float = Field(default=0.993, ge=0, le=1)
    ema_tau: int = Field(default=100, ge=0)
    lr_drop: int = Field(default=100, ge=0)
    checkpoint_interval: int = Field(default=10, ge=1)
    warmup_epochs: float = Field(default=0, ge=0)
    lr_vit_layer_decay: float = Field(default=0.8, ge=0, le=1)
    lr_component_decay: float = Field(default=0.7, ge=0, le=1)
    drop_path: float = Field(default=0.0, ge=0, le=1)
    group_detr: int = Field(default=13, ge=0)
    ia_bce_loss: bool = True
    cls_loss_coef: float = Field(default=1.0, ge=0)
    bbox_loss_coef: float = Field(default=5.0, ge=0)
    giou_loss_coef: float = Field(default=2.0, ge=0)
    num_select: int = Field(default=300, gt=0)
    dataset_file: Literal["coco", "o365", "roboflow"] = "coco"
    square_resize_div_64: bool = True
    output_dir: str = "output"
    multi_scale: bool = True
    expanded_scales: Union[bool, list[int]] = True
    use_ema: bool = True
    num_workers: int = Field(default=2, ge=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    early_stopping: bool = False
    early_stopping_patience: int = Field(default=10, ge=1)
    early_stopping_min_delta: float = Field(default=0.001, ge=0)
    early_stopping_use_ema: bool = False
    tensorboard: bool = True
    wandb: bool = False
    project: Optional[str] = None
    run: Optional[str] = None


class DatasetConfig(BaseModel):
    """Configuration for dataset parameters."""

    coco_path: Optional[str] = ""
    coco_train: Optional[str] = ""
    coco_val: Optional[str] = ""
    coco_img_path: Optional[str] = ""


class MaskConfig(BaseModel):
    """Configuration for mask parameters."""

    enabled: bool = True
    loss_mask_coef: float = Field(default=1.0, ge=0)
    loss_dice_coef: float = Field(default=1.0, ge=0)


class OtherConfig(BaseModel):
    """Other configuration parameters."""

    seed: int = 42
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    world_size: int = Field(default=1, ge=1)
    dist_url: str = "env://"
    clip_max_norm: float = Field(default=0.5, ge=0)
    steps_per_validation: int = Field(default=0, ge=0)


class RFDETRConfig(BaseModel):
    """Main configuration class for RF-DETR."""

    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig
    mask: MaskConfig
    other: OtherConfig

    @model_validator(mode="after")
    @classmethod
    def validate_config(cls, values):
        """Validate that configuration parameters are consistent."""
        if values.model.num_select != values.training.num_select:
            error_msg = f"Model num_select ({values.model.num_select}) must match training num_select ({values.training.num_select})"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)

        if values.model.group_detr != values.training.group_detr:
            error_msg = f"Model group_detr ({values.model.group_detr}) must match training group_detr ({values.training.group_detr})"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)

        return values

    def to_args(self):
        """Convert the configuration to an argparse namespace for backward compatibility."""
        from rfdetr.main import populate_args

        # Start with basic model settings
        args_dict = {
            # Model parameters
            "encoder": self.model.encoder,
            "out_feature_indexes": self.model.out_feature_indexes,
            "dec_layers": self.model.dec_layers,
            "projector_scale": self.model.projector_scale,
            "hidden_dim": self.model.hidden_dim,
            "sa_nheads": self.model.sa_nheads,
            "ca_nheads": self.model.ca_nheads,
            "dec_n_points": self.model.dec_n_points,
            "bbox_reparam": self.model.bbox_reparam,
            "lite_refpoint_refine": self.model.lite_refpoint_refine,
            "layer_norm": self.model.layer_norm,
            "amp": self.model.amp,
            "num_classes": self.model.num_classes,
            "pretrain_weights": self.model.pretrain_weights,
            "device": self.model.device,
            "resolution": self.model.resolution,
            "group_detr": self.model.group_detr,
            "gradient_checkpointing": self.model.gradient_checkpointing,
            "num_queries": self.model.num_queries,
            "num_select": self.model.num_select,
            # Training parameters
            "lr": self.training.lr,
            "lr_encoder": self.training.lr_encoder,
            "batch_size": self.training.batch_size,
            "grad_accum_steps": self.training.grad_accum_steps,
            "epochs": self.training.epochs,
            "ema_decay": self.training.ema_decay,
            "ema_tau": self.training.ema_tau,
            "lr_drop": self.training.lr_drop,
            "checkpoint_interval": self.training.checkpoint_interval,
            "warmup_epochs": self.training.warmup_epochs,
            "lr_vit_layer_decay": self.training.lr_vit_layer_decay,
            "lr_component_decay": self.training.lr_component_decay,
            "drop_path": self.training.drop_path,
            "ia_bce_loss": self.training.ia_bce_loss,
            "cls_loss_coef": self.training.cls_loss_coef,
            "bbox_loss_coef": self.training.bbox_loss_coef,
            "giou_loss_coef": self.training.giou_loss_coef,
            "dataset_file": self.training.dataset_file,
            "square_resize_div_64": self.training.square_resize_div_64,
            "output_dir": self.training.output_dir,
            "multi_scale": self.training.multi_scale,
            "expanded_scales": self.training.expanded_scales,
            "use_ema": self.training.use_ema,
            "num_workers": self.training.num_workers,
            "weight_decay": self.training.weight_decay,
            "early_stopping": self.training.early_stopping,
            "early_stopping_patience": self.training.early_stopping_patience,
            "early_stopping_min_delta": self.training.early_stopping_min_delta,
            "early_stopping_use_ema": self.training.early_stopping_use_ema,
            # Dataset parameters
            "coco_path": self.dataset.coco_path,
            "coco_train": self.dataset.coco_train,
            "coco_val": self.dataset.coco_val,
            "coco_img_path": self.dataset.coco_img_path,
            # Mask parameters
            "masks": self.mask.enabled,
            "loss_mask_coef": self.mask.loss_mask_coef,
            "loss_dice_coef": self.mask.loss_dice_coef,
            # Other parameters
            "seed": self.other.seed,
            "world_size": self.other.world_size,
            "dist_url": self.other.dist_url,
            "clip_max_norm": self.other.clip_max_norm,
            "steps_per_validation": self.other.steps_per_validation,
        }

        # Use the populate_args function to fill in missing values with defaults
        args = populate_args(**args_dict)
        return args

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "RFDETRConfig":
        """
        Load configuration from a YAML file.
        Args:
            yaml_path: Path to the YAML configuration file
        Returns:
            Instantiated RFDETRConfig object
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            error_msg = f"Config file not found: {yaml_path}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)

        try:
            with open(yaml_path) as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML in configuration file {yaml_path}"
            logger.error(f"{error_msg}: {e}")
            raise ConfigurationError(error_msg) from e
        except Exception as e:
            error_msg = f"Error reading configuration file {yaml_path}"
            logger.error(f"{error_msg}: {e}")
            raise ConfigurationError(error_msg) from e

        try:
            return cls.model_validate(config_dict)
        except Exception as e:
            error_msg = f"Invalid configuration data in {yaml_path}"
            logger.error(f"{error_msg}: {e}")
            raise ConfigurationError(f"{error_msg}: {e}") from e


def load_config(config_path: Union[str, Path]) -> RFDETRConfig:
    """
    Load configuration from a YAML file.
    Args:
        config_path: Path to the YAML configuration file
    Returns:
        Instantiated RFDETRConfig object
    """
    return RFDETRConfig.from_yaml(config_path)
