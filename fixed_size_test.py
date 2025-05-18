#!/usr/bin/env python
# Test script that uses fixed-size inputs

import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rfdetr.config_utils import load_config
from rfdetr.models import build_model, build_criterion_and_postprocessors
from rfdetr.util.misc import NestedTensor
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.logging_config import get_logger

logger = get_logger(__name__)

# Create a dummy dataset with fixed size images
class FixedSizeDataset(Dataset):
    def __init__(self, num_samples=10, resolution=560, num_classes=69):
        self.num_samples = num_samples
        self.resolution = resolution
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Create a fixed-size tensor
        image = torch.randn(3, self.resolution, self.resolution)
        mask = torch.ones(self.resolution, self.resolution, dtype=torch.bool)
        
        # Create a target with a box and mask - use float32 for boxes
        target = {
            "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32),
            "labels": torch.tensor([1]),
            "area": torch.tensor([10000.0]),
            "iscrowd": torch.tensor([0]),
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([self.resolution, self.resolution]),
            "size": torch.tensor([self.resolution, self.resolution]),
            "masks": torch.ones((1, self.resolution, self.resolution)),
        }
        
        return image, mask, target

def collate_fn(batch):
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    
    # Create the NestedTensor directly
    batched_imgs = torch.stack(images)
    batched_masks = torch.stack(masks)
    
    return NestedTensor(batched_imgs, batched_masks), targets

def main():
    # Load configuration
    config_path = os.path.join("configs", "test_config.yaml")
    config = load_config(config_path)
    
    # Convert to args for backward compatibility
    args = config.to_args()
    
    # Adjust some parameters for the test
    args.batch_size = 1
    args.resolution = 560  # Standard resolution for DINOv2
    args.amp = False
    args.epochs = 1
    args.num_classes = 69
    
    # Set device
    device = torch.device(args.device)
    
    # Create the dummy dataset
    dataset = FixedSizeDataset(num_samples=5, resolution=args.resolution, num_classes=args.num_classes)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Build the model
    logger.info("Building model...")
    model = build_model(args)
    model.to(device)
    
    # Build criterion and postprocessors
    criterion, postprocessors = build_criterion_and_postprocessors(args)
    criterion.to(device)
    
    # Build optimizer
    param_dicts = get_param_dict(args, model)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    logger.info("Starting mini-test training...")
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for i, (samples, targets) in enumerate(dataloader):
            # Move to device
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            try:
                outputs = model(samples, targets)
                
                # Compute loss
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                
                logger.info(f"Batch {i+1}/{len(dataloader)}, Loss: {losses.item():.4f}")
                
                # Print some output properties
                if i == 0:
                    logger.info(f"Output keys: {outputs.keys()}")
                    logger.info(f"Loss keys: {loss_dict.keys()}")
            except Exception as e:
                logger.error(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}, Time: {epoch_time:.2f}s")
    
    logger.info("Test training complete!")

if __name__ == "__main__":
    main()