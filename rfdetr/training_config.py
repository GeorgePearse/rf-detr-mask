# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Configuration classes for RF-DETR training.
"""

from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Iteration-based training configuration."""

    # Basic training parameters
    max_steps: int = Field(2000, description="Maximum number of training steps")
    batch_size: int = Field(2, description="Batch size per device")
    grad_accum_steps: int = Field(1, description="Number of gradient accumulation steps")

    # Training scheduling
    val_frequency: int = Field(200, description="Validation frequency in steps")
    checkpoint_frequency: int = Field(500, description="Checkpoint saving frequency in steps")
    early_stopping: bool = Field(False, description="Whether to use early stopping")
    early_stopping_patience: int = Field(
        5, description="Number of validations with no improvement before stopping"
    )
    early_stopping_min_delta: float = Field(
        0.001, description="Minimum change to count as improvement"
    )

    # Learning rate and optimization
    lr: float = Field(1e-4, description="Base learning rate")
    lr_encoder: float = Field(1.5e-4, description="Learning rate for encoder")
    lr_projector: float = Field(1e-5, description="Learning rate for projector")
    lr_scheduler: Literal["cosine", "step"] = Field("cosine", description="LR scheduler type")
    warmup_ratio: float = Field(0.1, description="Fraction of steps for warmup")
    lr_min_factor: float = Field(0.0, description="Minimum LR factor at end of schedule")
    weight_decay: float = Field(1e-4, description="Weight decay for optimizer")

    # Mixed precision
    amp: bool = Field(False, description="Use automatic mixed precision")
    fp16_eval: bool = Field(False, description="Use FP16 precision for evaluation")

    # Training regularization
    clip_max_norm: float = Field(0.1, description="Gradient clipping max norm")
    dropout: float = Field(0.0, description="Dropout rate")
    drop_path: float = Field(0.0, description="Drop path rate")
    drop_mode: Literal["standard", "early", "late"] = Field("standard", description="Drop mode")
    drop_schedule: Literal["constant", "linear"] = Field(
        "constant", description="Drop schedule type"
    )
    cutoff_epoch: int = Field(0, description="Cutoff epoch for dropout")

    # Data parameters
    dataset_file: Literal["coco", "o365", "roboflow"] = Field("coco", description="Dataset format")
    coco_path: str = Field("", description="Path to COCO annotations directory")
    coco_train: str = Field(
        "2025-05-15_12:38:23.077836_train_ordered.json", description="Training annotation file name"
    )
    coco_val: str = Field(
        "2025-05-15_12:38:38.270134_val_ordered.json", description="Validation annotation file name"
    )
    coco_img_path: str = Field("", description="Path to images directory")
    num_workers: int = Field(2, description="Number of dataloader workers")
    resolution: int = Field(640, description="Input resolution for the model")
    square_resize: bool = Field(True, description="Resize images to square")
    square_resize_div_64: bool = Field(False, description="Resize to dimensions divisible by 64")
    multi_scale: bool = Field(False, description="Use multi-scale training")
    expanded_scales: bool = Field(False, description="Use expanded scale options")
    test_limit: Optional[int] = Field(None, description="Limit dataset size for testing")

    # Model parameters
    num_classes: int = Field(2, description="Number of classes")
    encoder: str = Field("dinov2_windowed_small", description="Encoder type")
    pretrain_weights: Optional[str] = Field(None, description="Path to pretrained weights")
    freeze_encoder: bool = Field(False, description="Freeze encoder weights")
    layer_norm: bool = Field(True, description="Use layer normalization")
    rms_norm: bool = Field(False, description="Use RMS normalization")
    backbone_lora: bool = Field(False, description="Use LoRA for backbone")

    # Transformer parameters
    hidden_dim: int = Field(256, description="Hidden dimension size")
    dec_layers: int = Field(3, description="Number of decoder layers")
    vit_encoder_num_layers: int = Field(12, description="Number of ViT encoder layers")
    dim_feedforward: int = Field(1024, description="Feedforward dimension")
    sa_nheads: int = Field(8, description="Self-attention heads")
    ca_nheads: int = Field(8, description="Cross-attention heads")
    group_detr: int = Field(1, description="Group DETR parameter")
    lite_refpoint_refine: bool = Field(True, description="Use lite reference point refinement")
    num_queries: int = Field(300, description="Number of query slots")
    bbox_reparam: bool = Field(True, description="Use bbox reparameterization")

    # Feature extraction parameters
    out_feature_indexes: list[int] = Field(
        [9, 10, 11], description="Feature extraction layers from backbone"
    )
    projector_scale: list[Literal["P3", "P4", "P5", "P6"]] = Field(
        ["P3", "P4", "P5"], description="Feature pyramid levels"
    )
    window_block_indexes: Optional[list[int]] = Field(
        None, description="Window block indexes for windowed attention"
    )
    use_cls_token: bool = Field(False, description="Use classification token from ViT")
    position_embedding: Literal["sine", "learned"] = Field(
        "sine", description="Type of position embedding"
    )

    # Loss parameters
    cls_loss_coef: float = Field(2.0, description="Classification loss coefficient")
    bbox_loss_coef: float = Field(5.0, description="Bbox loss coefficient")
    giou_loss_coef: float = Field(2.0, description="GIoU loss coefficient")
    focal_alpha: float = Field(0.25, description="Focal loss alpha")
    mask_loss_coef: float = Field(1.0, description="Mask loss coefficient")
    dice_loss_coef: float = Field(1.0, description="Dice loss coefficient")
    aux_loss: bool = Field(True, description="Use auxiliary losses")
    sum_group_losses: bool = Field(False, description="Sum group losses")
    use_varifocal_loss: bool = Field(False, description="Use varifocal loss")
    use_position_supervised_loss: bool = Field(False, description="Use position supervised loss")
    ia_bce_loss: bool = Field(False, description="Use IA BCE loss")

    # Matcher parameters
    set_cost_class: float = Field(2.0, description="Class coefficient in matching cost")
    set_cost_bbox: float = Field(5.0, description="L1 box coefficient in matching cost")
    set_cost_giou: float = Field(2.0, description="GIoU coefficient in matching cost")

    # EMA parameters
    use_ema: bool = Field(True, description="Use EMA model")
    ema_decay: float = Field(0.9997, description="EMA decay factor")
    ema_tau: float = Field(0, description="EMA tau value")

    # Output parameters
    output_dir: str = Field("output_iter_training", description="Output directory")
    dont_save_weights: bool = Field(False, description="Don't save weights")
    device: Literal["cuda", "cpu"] = Field("cuda", description="Device to use")
    seed: int = Field(42, description="Random seed")

    # Logging parameters
    tensorboard: bool = Field(True, description="Use TensorBoard logging")
    wandb: bool = Field(False, description="Use Weights & Biases logging")
    project: Optional[str] = Field(None, description="Project name for logging")
    run: Optional[str] = Field(None, description="Run name for logging")

    # ONNX export
    export_onnx: bool = Field(False, description="Export to ONNX format")
    export_torch: bool = Field(True, description="Export PyTorch model")
    simplify_onnx: bool = Field(False, description="Simplify ONNX model")
    opset_version: int = Field(17, description="ONNX opset version")

    # Distributed training parameters
    world_size: int = Field(1, description="Number of distributed processes")
    dist_url: str = Field("env://", description="URL used to set up distributed training")
    sync_bn: bool = Field(True, description="Use synchronized batch normalization")

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path]) -> "TrainingConfig":
        """Load configuration from a YAML file."""
        yaml_file = Path(yaml_file)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_file}")

        with open(yaml_file) as f:
            config_dict = yaml.safe_load(f)

        # Handle nested structure
        if "model" in config_dict:
            # Flatten the structure, prioritizing model attributes over others
            flattened_dict = {}
            flattened_dict.update(config_dict.get("other", {}))
            flattened_dict.update(config_dict.get("training", {}))
            flattened_dict.update(config_dict.get("dataset", {}))
            if "mask" in config_dict:
                # Add mask prefix to mask parameters
                for k, v in config_dict["mask"].items():
                    if k == "enabled":
                        # Special case for mask.enabled
                        flattened_dict["mask_enabled"] = v
                    else:
                        flattened_dict[k] = v
            # Model attributes have highest priority
            flattened_dict.update(config_dict.get("model", {}))

            # Output directory from training section should override model section
            if "output_dir" in config_dict.get("training", {}):
                flattened_dict["output_dir"] = config_dict["training"]["output_dir"]

            config_dict = flattened_dict

        return cls(**config_dict)

    def to_yaml(self, yaml_file: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        yaml_file = Path(yaml_file)
        yaml_file.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_file, "w") as f:
            yaml.dump(self.dict(), f, sort_keys=False, indent=2)

    def to_args_dict(self) -> dict[str, Any]:
        """Convert to a dictionary compatible with the legacy argparse approach."""
        return self.dict()
