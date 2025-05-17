#!/usr/bin/env python3
"""Simple test script to verify the dtype fix for evaluation"""

import argparse
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from rfdetr.util.misc import NestedTensor
from scripts.train_coco_segmentation import get_args_parser
from rfdetr.models.segmentation import build_model
from rfdetr.engine import evaluate


def test_dtype_fix():
    """Test that the dtype fix works properly during evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create minimal args for model building
    parser = get_args_parser()
    args = parser.parse_args([
        '--masks',
        '--num_classes', '2',  # Minimal test
        '--coco_path', '/home/georgepearse/data/cmr',
        '--train_anno', '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json',
        '--val_anno', '/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json',
    ])
    
    # Build model
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    criterion.to(device)
    
    print("Testing model evaluation with fp16...")
    
    # Test in regular float32 mode first
    model.eval()
    batch_size = 2
    channels = 3
    height = 320
    width = 320
    
    images = torch.rand(batch_size, channels, height, width).to(device)
    mask = torch.zeros(batch_size, height, width, dtype=torch.bool).to(device)
    samples = NestedTensor(images, mask)
    
    # Test float32 forward pass
    print("Testing float32 forward pass...")
    with torch.no_grad():
        outputs = model(samples)
    print("✓ Float32 evaluation works")
    
    # Test half precision forward pass
    print("\nTesting half precision forward pass...")
    model.half()
    samples.tensors = samples.tensors.half()
    
    with torch.no_grad():
        outputs = model(samples)
    print("✓ Half precision evaluation works")
    
    print("\n✅ Dtype fix test passed successfully!")


if __name__ == "__main__":
    test_dtype_fix()