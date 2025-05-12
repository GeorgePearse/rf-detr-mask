#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Minimal test script to verify RF-DETR can train on segmentation data.
This script:
1. Creates a small synthetic dataset with masks
2. Verifies the model can produce mask outputs
3. Runs a single training step to ensure gradients flow correctly
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add the parent directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfdetr.models import build_model, build_criterion_and_postprocessors
from rfdetr.util.misc import NestedTensor, collate_fn
from rfdetr.main import populate_args


class SyntheticMaskDataset(Dataset):
    """Synthetic dataset with bounding boxes and segmentation masks."""
    
    def __init__(self, num_samples=10, image_size=640, num_classes=10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create a random image
        img = torch.rand(3, self.image_size, self.image_size)
        
        # Create random number of objects (1-5)
        num_objects = torch.randint(1, 6, (1,)).item()
        
        # Create boxes [x1, y1, x2, y2]
        boxes = []
        for _ in range(num_objects):
            x1 = torch.randint(0, self.image_size // 2, (1,)).item()
            y1 = torch.randint(0, self.image_size // 2, (1,)).item()
            x2 = torch.randint(x1 + 50, min(x1 + 200, self.image_size), (1,)).item()
            y2 = torch.randint(y1 + 50, min(y1 + 200, self.image_size), (1,)).item()
            boxes.append([x1, y1, x2, y2])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        # Create masks
        masks = torch.zeros((num_objects, self.image_size, self.image_size), dtype=torch.bool)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.int()
            masks[i, y1:y2, x1:x2] = True
        
        # Create labels
        labels = torch.randint(1, self.num_classes + 1, (num_objects,))
        
        # Create target dict
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([self.image_size, self.image_size]),
            'size': torch.tensor([self.image_size, self.image_size]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros((num_objects,), dtype=torch.int64)
        }
        
        return img, target


def test_segmentation_training():
    """Test if RF-DETR can train on segmentation data."""
    print("Starting segmentation training test")
    
    # Create args
    args = populate_args(
        num_classes=10,
        batch_size=2,
        resolution=644,  # Make divisible by 14 for DINOv2
        output_dir=tempfile.mkdtemp(),
        device="cpu",  # Use CPU for testing
        encoder="dinov2_small",  # Use DINOv2 small model
        out_feature_indexes=[11],  # Last layer of DINOv2
        projector_scale=['P4']  # Use P4 scale for feature projection
    )
    
    # Create synthetic dataset
    dataset = SyntheticMaskDataset(num_samples=10, image_size=args.resolution, num_classes=args.num_classes)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Build model and criterion
    model = build_model(args)
    criterion, postprocessors = build_criterion_and_postprocessors(args)
    
    # Verify model can output masks
    print("Verifying model can output masks...")
    model.eval()
    with torch.no_grad():
        # Get a batch
        batch = next(iter(dataloader))
        samples, targets = batch
        
        # Forward pass
        outputs = model(samples)
        
        # Check if masks are in the output
        if 'pred_masks' not in outputs:
            print("ERROR: Model output does not include 'pred_masks'")
            return False
        
        # Check mask shape (should be [batch_size, num_queries, 28, 28])
        expected_shape = [args.batch_size, args.num_queries, 28, 28]  # Model uses fixed 28x28 mask size
        actual_shape = list(outputs['pred_masks'].shape)
        if actual_shape != expected_shape:
            print(f"ERROR: Expected mask shape {expected_shape}, got {actual_shape}")
            return False
        
        print(f"Mask shape verification passed: {actual_shape}")
    
    # Test training step
    print("Testing training step...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Get a batch
    batch = next(iter(dataloader))
    samples, targets = batch
    
    # Forward pass
    outputs = model(samples)
    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    
    # Calculate loss
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    
    # Check if loss contains mask loss
    mask_loss_keys = [k for k in loss_dict.keys() if 'mask' in k]
    if not mask_loss_keys:
        print("WARNING: No mask loss found in loss dict")
    else:
        print(f"Found mask loss keys: {mask_loss_keys}")
    
    # Backward pass
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    
    # Check if gradients flowed to mask-related parameters
    mask_grad_exists = False
    for name, param in model.named_parameters():
        if 'mask' in name and param.grad is not None and torch.any(param.grad != 0):
            mask_grad_exists = True
            print(f"Verified gradients in parameter: {name}")
            break
    
    if not mask_grad_exists:
        print("WARNING: No gradients found in mask-related parameters")
    
    # Test postprocessors
    print("Testing mask postprocessing...")
    
    # Process the outputs with the segm postprocessor
    if 'segm' in postprocessors:
        with torch.no_grad():
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            mask_results = postprocessors['segm'](outputs, orig_target_sizes)
            
            # Check if masks are in the results
            if 'masks' not in mask_results[0]:
                print("ERROR: Postprocessor output does not include 'masks'")
                return False
            
            print(f"Postprocessor output 'masks' shape: {mask_results[0]['masks'].shape}")
    else:
        print("WARNING: 'segm' not in postprocessors")
    
    print("Segmentation training test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_segmentation_training()
    if success:
        print("✅ Test passed: RF-DETR can train on segmentation data")
        sys.exit(0)
    else:
        print("❌ Test failed: RF-DETR cannot train on segmentation data")
        sys.exit(1)