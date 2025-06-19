"""Configuration management for RF-DETR training.

This module provides a unified configuration system that handles
model, training, and data configuration.
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Model type
    model_type: str = "rf-detr-base"
    encoder: str = "dinov2_windowed_small"
    
    # Architecture parameters
    num_queries: int = 300
    hidden_dim: int = 256
    num_classes: int = 80
    
    # Transformer parameters
    num_decoder_layers: int = 6
    num_encoder_layers: int = 12
    dim_feedforward: int = 2048
    dropout: float = 0.0
    drop_path: float = 0.1
    
    # Head parameters
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    
    # Feature extraction
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    
    # Mask head parameters (if segmentation is enabled)
    masks: bool = False
    mask_in_features: int = 256
    mask_out_dim: int = 256
    
    # Other model options
    aux_loss: bool = True
    two_stage: bool = True
    position_embedding: str = "sine"
    backbone_feature_layers: List[str] = field(
        default_factory=lambda: ["res2", "res3", "res4", "res5"]
    )


@dataclass
class DataConfig:
    """Data loading and augmentation configuration."""
    
    # Dataset
    dataset: str = "coco"
    dataset_file: str = "coco"
    
    # Paths
    coco_path: str = ""
    coco_train: str = "train2017"
    coco_val: str = "val2017"
    coco_img_path: str = ""
    
    # Data loading
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Image size and augmentation
    resolution: int = 448
    square_resize: bool = True
    square_resize_div_64: bool = False
    rectangular: bool = False
    rect_width: int = 832
    rect_height: int = 640
    
    # Multi-scale training
    multi_scale: bool = False
    expanded_scales: List[int] = field(
        default_factory=lambda: [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    )
    
    # Augmentation
    use_albumentations: bool = False
    albumentations_config: Optional[str] = None
    
    # Validation
    test_limit: Optional[int] = None
    steps_per_validation: int = 100


@dataclass
class TrainConfig:
    """Training configuration."""
    
    # Basic training
    epochs: int = 100
    seed: int = 42
    device: str = "cuda"
    
    # Learning rates
    lr: float = 1e-4
    lr_encoder: float = 1e-5
    lr_projector: float = 1e-5
    lr_vit_layer_decay: float = 1.0
    lr_component_decay: float = 0.9
    lr_drop: int = 50
    
    # Optimization
    weight_decay: float = 1e-4
    clip_max_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Loss coefficients
    loss_class_coef: float = 4.5
    loss_bbox_coef: float = 2.0
    loss_giou_coef: float = 1.0
    loss_mask_coef: float = 1.0
    loss_dice_coef: float = 1.0
    
    # Matching costs
    set_cost_class: float = 5.0
    set_cost_bbox: float = 2.0
    set_cost_giou: float = 1.0
    
    # Training options
    amp: bool = False
    use_fp16: bool = False
    sync_bn: bool = False
    
    # Checkpointing
    checkpoint_frequency: int = 10
    val_frequency: int = 1
    output_dir: str = "output"
    resume: Optional[str] = None
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Logging
    print_per_class_metrics: bool = False
    log_frequency: int = 50
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    dist_url: str = "env://"


@dataclass
class Config:
    """Complete configuration combining all sub-configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        
        config_dict = self.to_dict()
        
        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix == ".json":
            with open(path, "r") as f:
                config_dict = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Create sub-configs
        model_config = ModelConfig(**config_dict.get("model", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        train_config = TrainConfig(**config_dict.get("train", {}))
        
        return cls(model=model_config, data=data_config, train=train_config)


class ConfigurationManager:
    """Manages configuration loading, validation, and merging."""
    
    def __init__(self, base_config: Optional[Config] = None):
        """Initialize configuration manager.
        
        Args:
            base_config: Base configuration to use
        """
        self.config = base_config or Config()
    
    def load_from_file(self, path: Union[str, Path]) -> None:
        """Load configuration from file.
        
        Args:
            path: Path to configuration file
        """
        self.config = Config.load(path)
    
    def update_from_args(self, args: Any) -> None:
        """Update configuration from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        # Update model config
        if hasattr(args, "encoder"):
            self.config.model.encoder = args.encoder
        if hasattr(args, "num_classes"):
            self.config.model.num_classes = args.num_classes
        if hasattr(args, "masks"):
            self.config.model.masks = args.masks
        
        # Update data config
        if hasattr(args, "dataset"):
            self.config.data.dataset = args.dataset
        if hasattr(args, "batch_size"):
            self.config.data.batch_size = args.batch_size
        if hasattr(args, "num_workers"):
            self.config.data.num_workers = args.num_workers
        if hasattr(args, "resolution"):
            self.config.data.resolution = args.resolution
        
        # Update train config
        if hasattr(args, "epochs"):
            self.config.train.epochs = args.epochs
        if hasattr(args, "lr"):
            self.config.train.lr = args.lr
        if hasattr(args, "output_dir"):
            self.config.train.output_dir = args.output_dir
        
        # Update all other attributes
        for key, value in vars(args).items():
            # Check each sub-config
            if hasattr(self.config.model, key):
                setattr(self.config.model, key, value)
            elif hasattr(self.config.data, key):
                setattr(self.config.data, key, value)
            elif hasattr(self.config.train, key):
                setattr(self.config.train, key, value)
    
    def validate(self) -> List[str]:
        """Validate configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate model config
        if self.config.model.num_classes < 1:
            errors.append("num_classes must be at least 1")
        
        if self.config.model.hidden_dim % 8 != 0:
            errors.append("hidden_dim should be divisible by 8")
        
        # Validate data config
        if self.config.data.batch_size < 1:
            errors.append("batch_size must be at least 1")
        
        if self.config.data.resolution % 14 != 0 and "dinov2" in self.config.model.encoder:
            errors.append("resolution must be divisible by 14 for DINOv2 models")
        
        # Validate train config
        if self.config.train.epochs < 1:
            errors.append("epochs must be at least 1")
        
        if self.config.train.lr <= 0:
            errors.append("learning rate must be positive")
        
        return errors
    
    def get_config(self) -> Config:
        """Get the current configuration.
        
        Returns:
            Current configuration
        """
        return self.config
    
    def to_namespace(self) -> Any:
        """Convert configuration to namespace for compatibility.
        
        Returns:
            Namespace object with all configuration attributes
        """
        from argparse import Namespace
        
        # Flatten all configs into a single namespace
        namespace = Namespace()
        
        # Add model config
        for key, value in asdict(self.config.model).items():
            setattr(namespace, key, value)
        
        # Add data config
        for key, value in asdict(self.config.data).items():
            setattr(namespace, key, value)
        
        # Add train config
        for key, value in asdict(self.config.train).items():
            setattr(namespace, key, value)
        
        return namespace