# CLAUDE.md for Adapters Directory

## Purpose

The `adapters` directory contains compatibility layer modules that bridge the RF-DETR-Mask core functionality with modern training frameworks like PyTorch Lightning. This allows the architecture to utilize higher-level abstractions without modifying the core model implementation.

## Adapter Modules

1. **data_module.py** - Contains `RFDETRDataModule`, a PyTorch Lightning DataModule that:
   - Handles dataset initialization, configuration, and loading
   - Manages dataloaders with proper sampling strategies
   - Provides consistent data pre-processing for training and validation

2. **rfdetr_lightning.py** - Contains `RFDETRLightningModule`, a PyTorch Lightning Module that:
   - Wraps the RF-DETR model with Lightning training infrastructure
   - Handles forward pass, loss computation, and optimization
   - Provides evaluation functionality with COCO metrics
   - Supports Model EMA (Exponential Moving Average) for better generalization
   - Manages mixed precision training with autocast

3. **training_config.py** - Contains `TrainingConfig`, a Pydantic model that:
   - Provides structured configuration for training parameters
   - Supports YAML-based configuration with validation
   - Handles nested configuration structures
   - Provides type safety and default values

## Usage Patterns

1. **Configuration:**
   ```python
   from adapters.training_config import TrainingConfig
   
   # Load from YAML
   config = TrainingConfig.from_yaml("configs/mask_enabled.yaml")
   
   # Or construct programmatically
   config = TrainingConfig(
       batch_size=4,
       max_steps=10000,
       # other parameters...
   )
   ```

2. **Training Setup:**
   ```python
   from adapters.data_module import RFDETRDataModule
   from adapters.rfdetr_lightning import RFDETRLightningModule
   from lightning.pytorch import Trainer
   
   # Initialize data module and model
   data_module = RFDETRDataModule(config)
   model = RFDETRLightningModule(config)
   
   # Create trainer and fit
   trainer = Trainer(
       max_steps=config.max_steps,
       # other trainer options...
   )
   trainer.fit(model, data_module)
   ```

## Integration with Core RF-DETR-Mask

The adapter modules maintain clear interfaces with the core RF-DETR-Mask implementation:

- Data module uses `rfdetr.datasets.build_dataset` to construct datasets
- Lightning module uses `rfdetr.models.build_model` to create the model architecture
- Configuration structures are compatible with core model requirements
- Error handling follows project conventions

## Best Practices

1. When using these adapters:
   - Prefer YAML configurations over programmatic initialization
   - Use the adapter classes as-is without subclassing when possible
   - Follow the Lightning lifecycle (setup, train/val steps, epoch callbacks)

2. When extending the adapters:
   - Maintain backward compatibility with existing configuration files
   - Keep clear separation between adapter logic and core model functionality
   - Add comprehensive type hints for better IDE support
   - Follow the error handling patterns established in the codebase