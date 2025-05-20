# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Never use the python typing module, we'll be using a new enough version of python that ever type annotation
we need is already avaiable.

Never fix an import error with sys.path.append

Make sure to run `bash kill.sh` between consecutive runs of training to make sure that the GPU memory is completely flushed
When testing the train.py script make sure that you do use the GPU

Always default to a pydantic class, and a YAML config file, instead of using argparse.

Whenever you suggest that I run a script in a certain way, can you make sure that you've already tried to run that script.

Use "hasattr" very sparingly. It is often over-used by LLMs

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


## Important 
- Use Context7 when used to implement something with a specific package, it helps you retrieve the latest docs.
- Amp is a command line tool for writing code made by sourcegraph.
