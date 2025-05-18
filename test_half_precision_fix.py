#!/usr/bin/env python
"""Test script to verify the half precision fix for cdist in matcher"""

import torch
from rfdetr.models.matcher import HungarianMatcher

# Create matcher
matcher = HungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0)

# Create test data in half precision
bs, num_queries, num_classes = 2, 100, 10
outputs = {
    "pred_logits": torch.rand(bs, num_queries, num_classes).half().cuda(),
    "pred_boxes": torch.rand(bs, num_queries, 4).half().cuda()
}

targets = [
    {
        "labels": torch.tensor([0, 1]).cuda(),
        "boxes": torch.rand(2, 4).half().cuda()
    },
    {
        "labels": torch.tensor([1, 2]).cuda(), 
        "boxes": torch.rand(2, 4).half().cuda()
    }
]

# Test the matcher - this should now work without errors
try:
    indices = matcher(outputs, targets)
    print("SUCCESS: Half precision matching works correctly!")
    print(f"Indices: {indices}")
except RuntimeError as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()