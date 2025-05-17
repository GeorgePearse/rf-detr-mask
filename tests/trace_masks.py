#!/usr/bin/env python3
"""Trace where masks are being lost"""

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

# Import RF-DETR with mask support
from rfdetr import RFDETRBase


def trace_mask_flow():
    """Trace the flow of masks through the prediction pipeline"""

    print("Tracing mask flow through RF-DETR prediction...")

    # Create a dummy image
    image = np.ones((640, 640, 3), dtype=np.uint8) * 255
    image[100:300, 100:400] = [255, 0, 0]  # Red rectangle
    pil_image = Image.fromarray(image)

    # Initialize model on CPU
    print("Initializing model on CPU...")
    model = RFDETRBase(device="cpu")

    # Process image step by step
    print("\n1. Processing image...")
    img_tensor = F.to_tensor(pil_image)
    h, w = img_tensor.shape[1:]
    img_tensor = img_tensor.to(model.model.device)
    img_tensor = F.normalize(img_tensor, model.means, model.stds)
    img_tensor = F.resize(img_tensor, (model.model.resolution, model.model.resolution))
    batch_tensor = img_tensor.unsqueeze(0)

    print("\n2. Getting model predictions...")
    with torch.inference_mode():
        predictions = model.model.model(batch_tensor)
        print(f"   - pred_masks in predictions: {'pred_masks' in predictions}")
        if "pred_masks" in predictions:
            print(f"   - pred_masks shape: {predictions['pred_masks'].shape}")

    print("\n3. Running postprocessor...")
    target_sizes = torch.tensor([[h, w]], device=model.model.device)
    results = model.model.postprocessors["bbox"](predictions, target_sizes=target_sizes)
    print(f"   - Number of results: {len(results)}")
    if results:
        result = results[0]
        print(f"   - Result keys: {list(result.keys())}")
        if "masks" in result:
            print(f"   - masks shape: {result['masks'].shape}")

        # Check scores and filtering
        scores = result["scores"]
        print("\n4. Score filtering...")
        print(f"   - Total scores: {len(scores)}")
        print(f"   - Scores > 0.1: {(scores > 0.1).sum().item()}")
        print(f"   - Scores > 0.3: {(scores > 0.3).sum().item()}")
        print(f"   - Scores > 0.5: {(scores > 0.5).sum().item()}")

        # Manual filtering simulation
        threshold = 0.1
        keep = scores > threshold
        print(f"\n5. Manual filtering (threshold={threshold})...")
        print(f"   - Keep mask: {keep.sum().item()} items")

        if "masks" in result:
            masks = result["masks"]
            filtered_masks = masks[keep]
            print(f"   - Filtered masks shape: {filtered_masks.shape}")
            print(f"   - Filtered masks dtype: {filtered_masks.dtype}")

            # Convert to numpy
            mask_array = filtered_masks.cpu().numpy()
            print(f"   - Numpy mask array shape: {mask_array.shape}")

    print("\n6. Running full predict method...")
    detections = model.predict(pil_image, threshold=0.1, return_masks=True)
    print(f"   - Detections type: {type(detections)}")
    print(f"   - Detections value: {detections}")
    if isinstance(detections, tuple):
        print(f"   - Tuple length: {len(detections)}")
        for i, item in enumerate(detections):
            print(f"   - Item {i} type: {type(item)}")
            print(f"   - Item {i} length: {len(item)}")
    elif isinstance(detections, list):
        print(f"   - List length: {len(detections)}")
        for i, item in enumerate(detections):
            print(f"   - Item {i} type: {type(item)}")
            if hasattr(item, "mask"):
                print(f"   - Item {i} has mask: {item.mask is not None}")
                if item.mask is not None:
                    print(f"   - Item {i} mask shape: {item.mask.shape}")
    else:
        print(f"   - Number of detections: {len(detections)}")
        print(f"   - Has mask attribute: {hasattr(detections, 'mask')}")
        if hasattr(detections, "mask"):
            print(f"   - Mask is None: {detections.mask is None}")
            if detections.mask is not None:
                print(f"   - Mask shape: {detections.mask.shape}")

    print("\nTrace completed.")


if __name__ == "__main__":
    trace_mask_flow()
