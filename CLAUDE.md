# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Before writing any code, make a plan and check it with any other available models (ChatGPT and Gemini for instance), after you complete any code check the implementation with ChatGPT and Gemini.

After any implementation is complete, run:
`python scripts/train.py --steps_per_validation 20 --test_limit 20`

And check that it still succeeds until class-level metrics are displayed.

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
   - `segmentation.py`: Mask head implementation (MaskHeadSmallConv) for instance segmentation
   - `transformer.py`: Transformer-based detection head with deformable attention
   - `backbone/`: DINOv2-based backbone implementations with optional windowed attention
   - `matcher.py`: Hungarian matcher for training target assignment
   - `ops/`: Custom CUDA operations for deformable attention

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
The segmentation head adds:
- Convolutional mask prediction head that takes transformer outputs and multi-scale features
- FPN-style multi-scale feature fusion from backbone layers
- Mask loss computation integrated into the criterion
- Modified postprocessing to return both boxes and masks
- Support for COCO-style polygon annotations

### Training Features
- Gradient accumulation with configurable steps
- Mixed precision training with GradScaler
- Distributed training support (DDP)
- Callback system for extensibility
- Dynamic learning rate scheduling
- Early stopping support
- TensorBoard and Weights & Biases logging integration

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
