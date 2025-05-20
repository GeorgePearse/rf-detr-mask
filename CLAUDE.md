# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RF-DETR-MASK is an instance segmentation extension of the RF-DETR architecture, adding pixel-precise object delineation capabilities to the original object detection model. It extends the base RF-DETR with a mask prediction head for detailed instance segmentation while maintaining real-time performance characteristics.

The architecture consists of:
- A DINOv2 backbone (various sizes)
- A DETR-like transformer for object detection
- A mask prediction head for instance segmentation

## Core Concepts

- The project follows a modular architecture with clear separation between model components, training logic, and data handling
- Configuration is managed through YAML files and Pydantic models
- PyTorch Lightning is used for training orchestration
- Mask prediction adds instance segmentation capabilities to the base detection model
- Models are trained on COCO-format datasets with support for segmentation masks

## Build/Testing Commands

- Install in dev mode: `pip install -e ".[dev]"`
- Run all tests: `python -m unittest discover tests`
- Run specific test: `python tests/test_minimal_segmentation.py`
- Run segmentation-specific test: `python tests/test_segmentation_integration.py`
- Test mask shape handling: `python tests/test_mask_shape_handling.py`
- Install ONNX export dependencies: `pip install ".[onnxexport]"`
- Install metrics dependencies: `pip install ".[metrics]"`
- Install build dependencies: `pip install ".[build]"`
- Train with default config: `python scripts/train.py`
- Train with mask enabled: `python scripts/train.py configs/mask_enabled.yaml`
- Evaluate a model: `EVAL_ONLY=1 RESUME_CHECKPOINT=path/to/checkpoint python scripts/train.py`
- Run quality checks: `ruff check .` and `mypy .`

## Training Commands

- Basic training: `python scripts/train.py configs/default.yaml`
- Train with masks: `python scripts/train.py configs/mask_enabled.yaml`
- Resume training: `RESUME_CHECKPOINT=path/to/checkpoint.pth python scripts/train.py configs/default.yaml`
- Run quick test training: `python scripts/train.py configs/test_mode.yaml`

## Memory Management

- Run `bash kill.sh` between consecutive runs of training to ensure GPU memory is completely flushed
- When testing the train.py script, use the GPU (as specified in training configs)
- To manually clean GPU memory: `torch.cuda.empty_cache()`

## Configuration Guidelines

- Always default to a pydantic class and YAML config file instead of using argparse
- Config files are located in the `configs/` directory
- Default config is in `configs/default.yaml`
- Mask-enabled config is in `configs/mask_enabled.yaml`
- All configuration settings should have proper defaults in the Pydantic models

## Testing Guidelines

- When testing instance segmentation, use the specified annotation files:
  - `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json`
  - `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json`
- Images are available at `/home/georgepearse/data/images`
- Always check the number of classes in annotations, as the model architecture must match
- Write tests for any new functionality you add
- Verify script execution before suggesting running commands

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

## Important Coding Patterns

- Use fully typed Python code to leverage mypy for faster feedback
- NEVER use `hasattr` as it is considered an anti-pattern in this repo
  - Replace with Protocol classes and isinstance checks
  - Use proper default values in Pydantic configs
  - Use try/except for runtime attribute checks where appropriate
- `getattr` is an anti-pattern, the underlying class should have the default value
- Never use the python typing module, Python is new enough that all required type annotations are available
- Never fix an import error with sys.path.append
- Write fully typed code for better mypy error checking
- Always verify script execution before suggesting running commands

## External Resources

- Use Context7 when implementing something with a specific package to retrieve the latest docs
- Amp is a command line tool for writing code made by sourcegraph

## Common Workflow Patterns

- Model and training configuration is managed through Pydantic classes and YAML files
- Training is handled through PyTorch Lightning, with a custom LightningModule
- Dataset loading follows the PyTorch Lightning DataModule pattern
- Segmentation is implemented using a mask head with attention mechanism
- All model variants use a DINOv2 backbone with different configurations
- Early stopping and model checkpointing are implemented through Lightning callbacks

## Task Strategies

- When working with the codebase, first understand the architecture and data flow
- For fixing bugs, identify the relevant module before making changes
- Use pre-commit hooks to ensure code quality before committing
- Always add tests for new functionality
- Run mypy on modified code to catch type issues early
- Use correct channel order for images (RGB vs BGR) when doing inference
- Follow the proper input resolution requirements (must be divisible by 56)

## Anti-Patterns to Avoid

- NEVER use `hasattr` - see the hasattr_replacement documents in docs/
- Don't add arbitrary checks for config attributes - use Pydantic defaults
- Don't manipulate Python's system path - fix the underlying imports
- Don't use `getattr` with fallback values - the class should define defaults
- Don't leave TODOs or incomplete implementations
- Never use relative imports, use absolute imports instead

IMPORTANT: Always follow the proper patterns for configuration handling, use Protocol classes instead of hasattr checks, and ensure proper type annotations throughout the codebase.