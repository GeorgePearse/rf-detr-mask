#!/usr/bin/env python3
"""Test script to verify evaluation with fp16_eval works after the dtype fix"""

import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rfdetr.models import build_model, build_criterion_and_postprocessors
from rfdetr.util.misc import NestedTensor
from rfdetr.main import populate_args


def create_test_args():
    """Create minimal args for testing"""
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    
    # Set minimal required args
    args.dataset = 'coco'
    args.dataset_file = 'coco'
    args.coco_path = '/home/georgepearse/data/cmr'
    args.masks = True
    args.num_classes = 2  # Simple 2-class test
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.amp = True
    args.fp16_eval = True  # This is the key setting to test
    args.encoder = 'dinov2_small'  # Set a valid encoder
    
    # Use populate_args to fill in all defaults
    args_dict = vars(args)
    args = populate_args(**args_dict)
    
    return args


def test_evaluation_with_fp16():
    """Test that evaluation with fp16_eval works properly"""
    args = create_test_args()
    device = torch.device(args.device)
    
    # Build model and criterion
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    criterion.to(device)
    
    print("Testing evaluation with fp16_eval enabled...")
    
    # Set model to eval mode
    model.eval()
    
    # Test with regular precision first
    print("\n1. Testing with float32 precision...")
    batch_size = 2
    channels = 3
    height = 640
    width = 640
    
    images = torch.rand(batch_size, channels, height, width).to(device)
    mask = torch.zeros(batch_size, height, width, dtype=torch.bool).to(device)
    samples = NestedTensor(images, mask)
    
    with torch.no_grad():
        outputs = model(samples)
    print("✓ Float32 evaluation works")
    
    # Test with half precision (fp16_eval enabled)
    print("\n2. Testing with fp16_eval (half precision)...")
    model.half()  # Convert model to half precision
    samples.tensors = samples.tensors.half()  # Convert inputs to half precision
    
    with torch.no_grad():
        outputs = model(samples)
    print("✓ Half precision evaluation works")
    
    print("\n✅ SUCCESS: The dtype fix works correctly!")
    print("The model can now properly handle evaluation with fp16_eval enabled.")


if __name__ == "__main__":
    test_evaluation_with_fp16()