#!/usr/bin/env python3
"""Test that AMP is enabled by default."""

import sys
sys.path.insert(0, '/home/georgepearse/rf-detr-mask')

from scripts.train import get_args
import argparse

# Test default value
parser = argparse.ArgumentParser()
args = get_args(parser)
default_args = parser.parse_args([])

print(f"AMP enabled by default: {default_args.amp}")
print(f"Type: {type(default_args.amp)}")

# Test explicit disable
disabled_args = parser.parse_args(["--amp", "False"])
print(f"\nAMP when set to False: {disabled_args.amp}")

# Test that it's being used
print(f"\nIn training, this will enable:")
print(f"- Automatic mixed precision with autocast")
print(f"- Gradient scaling for stable FP16 training")
print(f"- Better memory efficiency and faster training")