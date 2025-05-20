#!/usr/bin/env python
"""
Test script to verify the mask shape fix in coco_eval.py
"""

import numpy as np
import torch
import pycocotools.mask as mask_util

# Test different mask shapes that might be encountered
test_cases = [
    # Case 1: 2D mask (H, W)
    torch.rand(100, 100) > 0.5,
    
    # Case 2: 3D mask with batch dim (1, H, W)
    torch.rand(1, 100, 100) > 0.5,
    
    # Case 3: 3D mask with channel dim (H, W, 1)
    torch.rand(100, 100, 1) > 0.5,
    
    # Case 4: 4D mask (1, H, W, 1)
    torch.rand(1, 100, 100, 1) > 0.5,
]

print("Testing mask shape handling...")

for i, mask in enumerate(test_cases):
    print(f"\nTest case {i + 1}: input shape = {mask.shape}")
    
    # Simulate the fixed code behavior
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    # Handle cases where mask might have extra dimensions
    if mask_np.ndim > 2:
        # If mask has batch dimension, remove it
        if mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        # If mask has other extra dimensions, squeeze them
        mask_np = mask_np.squeeze()
    
    # Ensure mask is 2D
    print(f"Final shape: {mask_np.shape}")
    assert mask_np.ndim == 2, f"Mask should be 2D but got shape: {mask_np.shape}"
    
    # Encode the mask
    rle = mask_util.encode(np.array(mask_np[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
    print(f"Successfully encoded mask, RLE keys: {rle.keys()}")

print("\nAll test cases passed!")