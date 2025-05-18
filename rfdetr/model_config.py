# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Comprehensive Pydantic model configuration for RF-DETR models.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from pydantic import BaseModel, Field, field_validator

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class ModelConfig(BaseModel):
    """
    Comprehensive configuration for RF-DETR model construction and training.
    This replaces the use of dictionary arguments or attribute access objects.
    """
    # Core model parameters
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base", "dinov2_small", "dinov2_base"]
    out_feature_indexes: List[int]
    dec_layers: int = Field(default=3, ge=1)
    two_stage: bool = True
    projector_scale: List[Literal["P3", "P4", "P5", "P6"]]
    hidden_dim: int = Field(default=256, gt=0)
    
    # Attention & transformer parameters
    sa_nheads: int = Field(default=8, gt=0)
    ca_nheads: int = Field(default=16, gt=0)
    dec_n_points: int = Field(default=2, gt=0)
    dim_feedforward: int = Field(default=1024, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Architecture options
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    rms_norm: bool = False
    backbone_lora: bool = False
    
    # Training parameters
    amp: bool = True
    fp16_eval: bool = False
    group_detr: int = Field(default=13, gt=0)
    gradient_checkpointing: bool = False
    freeze_encoder: bool = False
    
    # Learning rate parameters
    lr: float = Field(default=1e-4, ge=0.0)
    lr_encoder: float = Field(default=1e-5, ge=0.0)
    lr_projector: float = Field(default=1e-5, ge=0.0)
    lr_scheduler: Literal["cosine", "step"] = "cosine"
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    lr_min_factor: float = Field(default=0.0, ge=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    lr_vit_layer_decay: float = Field(default=0.7, ge=0.0, le=1.0)
    lr_component_decay: float = Field(default=1.0, ge=0.0)
    
    # Dataset and preprocessing
    num_classes: int = Field(default=90, gt=0)
    resolution: int = Field(default=560, gt=0)
    shape: Optional[Tuple[int, int]] = None
    dataset_file: Literal["coco", "o365", "roboflow"] = "coco"
    coco_path: str = ""
    coco_train: str = ""
    coco_val: str = ""
    coco_img_path: str = ""
    multi_scale: bool = False
    expanded_scales: bool = False
    square_resize: bool = True
    square_resize_div_64: bool = False
    test_limit: Optional[int] = None
    
    # Initialization and weights
    pretrain_weights: Optional[str] = None
    force_no_pretrain: bool = False
    
    # Query parameters
    num_queries: int = Field(default=300, gt=0)
    num_select: int = Field(default=300, gt=0)
    
    # Position embedding
    position_embedding: Literal["sine", "learned"] = "sine"
    use_cls_token: bool = False
    
    # Hardware
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    
    # Backbone parameters - vit settings
    vit_encoder_num_layers: int = Field(default=12, gt=0)
    pretrained_encoder: bool = True
    window_block_indexes: Optional[List[int]] = None
    drop_path: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Loss parameters
    aux_loss: bool = True
    cls_loss_coef: float = Field(default=1.0, ge=0.0)
    bbox_loss_coef: float = Field(default=5.0, ge=0.0)
    giou_loss_coef: float = Field(default=2.0, ge=0.0)
    focal_alpha: float = Field(default=0.25, ge=0.0, le=1.0)
    sum_group_losses: bool = False
    use_varifocal_loss: bool = False
    use_position_supervised_loss: bool = False
    ia_bce_loss: bool = False
    
    # Matcher costs
    set_cost_class: float = Field(default=2.0, ge=0.0)
    set_cost_bbox: float = Field(default=5.0, ge=0.0) 
    set_cost_giou: float = Field(default=2.0, ge=0.0)
    
    @field_validator("resolution")
    def validate_resolution(cls, v):
        """Validate that resolution is divisible by 14 for DINOv2."""
        if v % 14 != 0:
            raise ValueError(f"Resolution {v} must be divisible by 14 for DINOv2")
        return v
    
    def dict_for_model_build(self) -> Dict[str, Any]:
        """
        Convert this Pydantic model to a dictionary for backward compatibility
        with code that expects a dict-like object.
        """
        return self.model_dump()
    
    def to_args_dict(self) -> Dict[str, Any]:
        """Alias for dict_for_model_build."""
        return self.dict_for_model_build()