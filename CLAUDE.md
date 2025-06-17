# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Before writing any code, make a plan and check it with any other available models (ChatGPT and Gemini for instance), after you complete any code check the implementation with ChatGPT and Gemini. Try to make it a conversation, e.g. the model should be presented with a question, and asked to review your work, but you can make it a multi-turn interaction, e.g. explain your own reasoning afterwards if necessary.

After any implementation is complete, run:
`python scripts/train.py --steps_per_validation 20 --test_limit 20`

And check that it still succeeds until class-level metrics are displayed.

Use agent parallelism where possible, e.g. spin up multiple claudes, and other LLMs to breakdown a task and build smaller components. Communicate via files on disk where possible.

## Recent Updates and Key Changes

### Architecture Simplifications
- Removed separate `segmentation.py` file - mask head is now integrated directly into LWDETR
- Consolidated model architecture for cleaner code organization
- Fixed circular dependency issues in the codebase

### ONNX Export Improvements
- Enhanced ONNX optimizer with better operator fusion
- Added proper handling for PyTorch version compatibility
- Improved symbolic shape inference for dynamic batch sizes

### Training Enhancements
- Added Albumentations support with configurable transforms
- Implemented per-class metrics tracking and visualization
- Added support for custom annotation formats
- Improved memory efficiency with better gradient accumulation

### Testing and Quality
- Comprehensive test suite covering detection and segmentation
- Integration tests for DINOv2 hub models
- Pre-commit hooks for code quality (ruff, mypy)
- Quick training tests for rapid validation

## Build/Testing Commands

### Installation
- Install in dev mode: `pip install -e ".[dev]"`
- Install ONNX export dependencies: `pip install ".[onnxexport]"`
- Install metrics dependencies: `pip install ".[metrics]"`
- Install build dependencies: `pip install ".[build]"`

### Testing
- Run all tests: `python -m pytest tests/ -v --cov=rfdetr --cov-report=xml --cov-report=term`
- Run specific test: `python tests/test_minimal_segmentation.py`
- Run unit tests: `python -m unittest discover tests`
- Run mypy type checking: `mypy rfdetr/ --ignore-missing-imports`
- Run linting: `ruff check rfdetr/` or `python -m ruff check rfdetr/`

### Training
- CLI training: `rfdetr-mask` (entry point defined in pyproject.toml)
- Script training for segmentation: `python scripts/train_coco_segmentation.py`
- Script training for detection: `python scripts/train.py`
- Multi-GPU training: `python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> --use_env main.py`

### Test Dataset Paths
When testing instance segmentation:
- Train annotations: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json`
- Val annotations: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json`
- Images directory: `/home/georgepearse/data/images`

## Code Architecture

### Core Model Architecture
RF-DETR-MASK extends RF-DETR (Real-time Fully End-to-End Detection with Transformers) with instance segmentation capabilities, achieving >60 AP on COCO with real-time performance.

1. **Base Classes** (`rfdetr/detr.py`):
   - `RFDETR`: Base class for all models, handles weight loading and configuration
   - `RFDETRBase`: Base model variant (29M params)
   - `RFDETRLarge`: Large model variant (128M params)
   - Models use DINOv2 backbones with pretrained weights from Meta AI

2. **Model Components** (`rfdetr/models/`):
   - `LWDETR`: Core detection model class that combines backbone, transformer, and heads
   - `transformer.py`: Transformer-based detection head with deformable attention
   - `backbone/`: DINOv2-based backbone implementations with optional windowed attention
   - `matcher.py`: Hungarian matcher for training target assignment
   - `ops/`: Custom CUDA operations for deformable attention
   - **Note**: Segmentation functionality is now integrated directly into LWDETR model

3. **Training Pipeline** (`rfdetr/engine.py`):
   Recently refactored into modular functions:
   - `train_one_epoch`: Main training loop with gradient accumulation support
   - `get_autocast_args`: Handles PyTorch version compatibility for AMP (prefers bfloat16)
   - `update_dropout_schedules`: Dynamic dropout/drop path rate adjustment
   - `compute_losses`: Batch loss computation with criterion
   - `process_gradient_accumulation_batch`: Memory-efficient gradient accumulation
   - `evaluate`: Unified evaluation for both detection and segmentation tasks

4. **Dataset Handling** (`rfdetr/datasets/`):
   - COCO dataset support with segmentation masks
   - O365 dataset support for large-scale pretraining
   - Custom transforms for data augmentation (LSJ, RandomSelect, etc.)
   - COCO evaluator for mAP computation (bbox and segm)

### Key Design Patterns
- Models inherit from `nn.Module` following PyTorch conventions
- Configuration through dataclasses (`RFDETRBaseConfig`, `RFDETRLargeConfig`)
- Pretrained weights auto-download from hosted URLs
- Support for both detection-only and detection+segmentation modes via `return_masks` parameter
- Mixed precision training with automatic dtype selection (bfloat16 preferred when available)
- Gradient accumulation for large batch training on limited GPU memory
- EMA (Exponential Moving Average) model support for improved stability

### Segmentation Extensions
The segmentation functionality is now integrated directly into the LWDETR model:
- Convolutional mask prediction head (MaskHeadSmallConv) that takes transformer outputs and multi-scale features
- FPN-style multi-scale feature fusion from backbone layers (fpn_channels configuration)
- Mask loss computation integrated into the criterion with configurable loss weights
- Modified postprocessing to return both boxes and masks
- Support for COCO-style polygon annotations
- Mask predictions are resized to match input image dimensions during inference
- Uses sigmoid activation for mask outputs with binary cross-entropy loss

### Training Features
- Gradient accumulation with configurable steps
- Mixed precision training with GradScaler
- Distributed training support (DDP)
- Callback system for extensibility
- Dynamic learning rate scheduling (warmup, multi-step, cosine)
- Early stopping support with patience configuration
- TensorBoard and Weights & Biases logging integration
- Per-class AP metrics tracking and visualization
- Albumentations integration for advanced augmentations
- Support for training with custom annotations

## Code Style Guidelines

- **Imports**: Standard lib first, third-party next (torch, numpy), project imports last
- **Documentation**: Triple double-quotes docstrings explaining purpose/parameters
- **Naming**: snake_case for variables/functions, CamelCase for classes, ALL_CAPS for constants
- **Types**: Use type hints for function signatures and complex data structures
- **Indentation**: 4 spaces
- **Error handling**: Use specific exceptions with descriptive messages
- **File headers**: Include copyright/license information
- **Design patterns**: Follow PyTorch conventions, use builder pattern for complex objects
- **Inheritance**: Extend appropriate PyTorch classes (nn.Module)
- **Performance**: Use torch.no_grad() for evaluation code

## Development Workflow

1. Use absolute imports within the package (e.g., `from rfdetr.models import build_model`)
2. Always check number of classes in annotation files - model architecture must match
3. Run pre-commit hooks after significant changes: `git add . && git commit`
4. Use mypy for type checking during development
5. Test on small dataset first before full training runs
6. When debugging training issues, check gradient accumulation settings and loss scaling
7. For segmentation tasks, ensure masks are properly loaded in the dataset
8. When modifying ONNX export, ensure compatibility with different PyTorch versions
9. Use Albumentations for advanced data augmentation (see `configs/transforms/`)
10. Monitor per-class metrics during training for balanced performance
