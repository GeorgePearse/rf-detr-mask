#!/usr/bin/env python3
"""Debug script to check model outputs"""

import torch
import numpy as np
from PIL import Image

# Import RF-DETR with mask support
from rfdetr import RFDETRBase


def debug_model_outputs():
    """Debug what the model actually outputs"""
    
    print("Debugging RF-DETR-MASK outputs...")
    
    # Create a dummy image
    image = np.ones((640, 640, 3), dtype=np.uint8) * 255
    image[100:300, 100:400] = [255, 0, 0]  # Red rectangle
    pil_image = Image.fromarray(image)
    
    # Initialize model on CPU
    print("Initializing model on CPU...")
    model = RFDETRBase(device='cpu')
    
    # Get raw model output
    print("\nChecking raw model output...")
    model.model.model.eval()
    
    # Process image manually to see raw outputs
    img_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    # Normalize
    means = torch.tensor(model.means).view(3, 1, 1)
    stds = torch.tensor(model.stds).view(3, 1, 1)
    img_tensor = (img_tensor - means) / stds
    
    # Resize to model resolution
    import torchvision.transforms.functional as F
    img_tensor = F.resize(img_tensor, (model.model.resolution, model.model.resolution))
    
    with torch.no_grad():
        outputs = model.model.model(img_tensor)
    
    print("\nModel output keys:", list(outputs.keys()))
    
    if 'pred_masks' in outputs:
        print("✓ pred_masks found in outputs")
        print(f"  Shape: {outputs['pred_masks'].shape}")
    else:
        print("✗ pred_masks NOT found in outputs")
    
    if 'pred_logits' in outputs:
        print("✓ pred_logits found in outputs")
        print(f"  Shape: {outputs['pred_logits'].shape}")
        
    if 'pred_boxes' in outputs:
        print("✓ pred_boxes found in outputs")
        print(f"  Shape: {outputs['pred_boxes'].shape}")
    
    # Check postprocessor output
    print("\nChecking postprocessor output...")
    target_sizes = torch.tensor([[640, 640]])
    results = model.model.postprocessors["bbox"](outputs, target_sizes=target_sizes)
    
    print(f"Number of results: {len(results)}")
    if results:
        result = results[0]
        print("Result keys:", list(result.keys()))
        
        if 'masks' in result:
            print("✓ masks found in postprocessor result")
            mask_shape = result['masks'].shape
            print(f"  Shape: {mask_shape}")
            print(f"  Masks dtype: {result['masks'].dtype}")
        else:
            print("✗ masks NOT found in postprocessor result")
        
        # Check scores
        scores = result['scores']
        print(f"\nScores shape: {scores.shape}")
        print(f"Max score: {scores.max().item():.4f}")
        print(f"Number of scores > 0.5: {(scores > 0.5).sum().item()}")
        print(f"Number of scores > 0.3: {(scores > 0.3).sum().item()}")
        print(f"Number of scores > 0.1: {(scores > 0.1).sum().item()}")
    
    # Test the actual predict function
    print("\nTesting actual predict function...")
    detections = model.predict(pil_image, threshold=0.1, return_masks=True)
    print(f"Detections found: {len(detections)}")
    print(f"Has mask attribute: {hasattr(detections, 'mask')}")
    if hasattr(detections, 'mask'):
        print(f"Mask is None: {detections.mask is None}")
        if detections.mask is not None:
            print(f"Mask shape: {detections.mask.shape}")
    
    print("\nDebug completed.")


if __name__ == "__main__":
    debug_model_outputs()