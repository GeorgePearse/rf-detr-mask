# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Never use the python typing module, we'll be using a new enough version of python that ever type annotation
we need is already avaiable.

Never fix an import error with sys.path.append

## Build/Testing Commands

- Install in dev mode: `pip install -e ".[dev]"`
- Run all tests: `python -m unittest discover tests`
- Run specific test: `python tests/test_minimal_segmentation.py`
- Install ONNX export dependencies: `pip install ".[onnxexport]"`
- Install metrics dependencies: `pip install ".[metrics]"`
- Install build dependencies: `pip install ".[build]"`

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
