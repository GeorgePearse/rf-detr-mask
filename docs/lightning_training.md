# PyTorch Lightning Integration for RF-DETR-Mask

This document describes the integration of PyTorch Lightning into the RF-DETR-Mask codebase to simplify training, enable multi-GPU training, and improve code organization.

## Implementation Overview

The Lightning integration consists of the following components:

1. **Lightning Module (`RFDETRLightningModule`)**: Encapsulates the RF-DETR model, loss functions, optimization logic, and training/validation steps.

2. **Lightning DataModule (`RFDETRDataModule`)**: Handles dataset creation, data loading, and batching.

3. **Training Script (`train_lightning.py`)**: Integrates with YAML configuration and manages the Lightning Trainer setup.

## Features

- **Configuration-driven training**: Uses YAML config files to control all aspects of training
- **Mixed precision training**: Automatically uses the most efficient precision format (bfloat16 when available, float16 otherwise)
- **Multi-GPU training**: Built-in support for distributed training using PyTorch Lightning's strategies
- **Experiment tracking**: Integrated with TensorBoard and optional Weights & Biases support
- **Checkpoint management**: Automatically saves best models and enables easy training resumption
- **Early stopping**: Optional early stopping to prevent overfitting
- **Gradient accumulation**: Supports training with large effective batch sizes on memory-constrained hardware
- **Learning rate monitoring**: Tracks learning rates for all parameter groups

## Usage

### Basic Training

To start training with the default configuration:

```bash
./train_lightning_fixed_size.sh
```

### Quick Testing

To verify that the setup works correctly with a minimal training run:

```bash
./test_lightning.sh
```

### Custom Configuration

To train with a custom configuration:

```bash
python scripts/train_lightning.py --config path/to/your/config.yaml --output_dir output_custom
```

### Resuming Training

To resume from a checkpoint:

```bash
python scripts/train_lightning.py --config path/to/your/config.yaml --resume path/to/checkpoint.ckpt
```

### Evaluation Only

To evaluate a trained model:

```bash
python scripts/train_lightning.py --config path/to/your/config.yaml --resume path/to/checkpoint.ckpt --eval
```

## Configuration

Configuration is managed through YAML files. See `configs/fixed_size_config.yaml` for an example.

The configuration is organized into the following sections:

- **model**: Model architecture parameters
- **training**: Training hyperparameters
- **dataset**: Dataset paths and settings
- **mask**: Instance segmentation parameters
- **other**: Miscellaneous settings

## Benefits Over Previous Training Setup

1. **Code organization**: Separates model, data, and training logic into clean, maintainable components
2. **Reduced boilerplate**: Removes the need for custom distributed training setup
3. **Simplified logging**: Consistent logging interface for metrics
4. **Better checkpointing**: More robust saving and loading of model states
5. **Multi-GPU support**: Simplified distributed training setup
6. **Extensibility**: Easier to add new features through Lightning's callback system