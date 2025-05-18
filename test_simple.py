#!/usr/bin/env python
# Simple test script to verify the configuration works

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from rfdetr.config_utils import load_config
from rfdetr.models import build_model

def main():
    # Load configuration
    config_path = os.path.join("configs", "test_config.yaml")
    config = load_config(config_path)
    
    # Convert to args for backward compatibility
    args = config.to_args()
    
    # Adjust some parameters for simple test
    args.batch_size = 1
    args.resolution = 560  # Standard resolution for DINOv2
    args.amp = False
    
    # Print args for debug
    print("Using args:")
    print(f"  encoder: {args.encoder}")
    print(f"  resolution: {args.resolution}")
    print(f"  amp: {args.amp}")
    print(f"  num_classes: {args.num_classes}")
    
    # Create a dummy input
    dummy_input = torch.randn(1, 3, args.resolution, args.resolution).to(args.device)
    dummy_targets = [
        {
            "boxes": torch.tensor([[100, 100, 200, 200]], device=args.device),
            "labels": torch.tensor([1], device=args.device),
            "area": torch.tensor([10000.0], device=args.device),
            "iscrowd": torch.tensor([0], device=args.device),
            "masks": torch.ones((1, args.resolution, args.resolution), device=args.device),
        }
    ]
    
    # Build the model
    print("Building model...")
    model = build_model(args)
    model.to(args.device)
    print(f"Model built with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    
    # Try a forward pass
    print("Testing forward pass...")
    try:
        from rfdetr.util.misc import NestedTensor
        samples = NestedTensor(dummy_input, torch.ones_like(dummy_input[:,0,:,:], dtype=torch.bool))
        outputs = model(samples, dummy_targets)
        print("Forward pass successful!")
        print(f"Output keys: {outputs.keys()}")
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("Test complete")

if __name__ == "__main__":
    main()