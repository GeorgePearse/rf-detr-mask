#!/usr/bin/env python3
"""Minimal test script to verify segmentation functionality"""

import cv2
import numpy as np
import torch
from PIL import Image

# Import RF-DETR with mask support
from rfdetr import RFDETRBase
from rfdetr.util.logging_config import get_logger

logger = get_logger(__name__)


def test_segmentation():
    """Test basic segmentation functionality"""

    logger.info("Testing RF-DETR-MASK segmentation...")

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create a dummy image
    image = np.ones((640, 640, 3), dtype=np.uint8) * 255
    image[100:300, 100:400] = [255, 0, 0]  # Red rectangle
    pil_image = Image.fromarray(image)

    # Initialize model on CPU to avoid memory issues
    print("Initializing model on CPU...")
    # Create a more minimal configuration to avoid parameter issues
    model = RFDETRBase(
        device="cpu",
        num_queries=100,
        num_select=100,
        pretrain_weights=None,
        backbone="vit_tiny",  # Explicitly set backbone to avoid config conflicts
    )

    # Test without masks
    print("\nTesting detection without masks...")
    detections_without_masks = model.predict(pil_image, threshold=0.5)
    print(f"Found {len(detections_without_masks)} objects")

    # Test with masks - use lower threshold
    print("\nTesting detection with masks...")
    detection_result = model.predict(pil_image, threshold=0.1, return_masks=True)

    # Handle both single Detections and list of Detections
    if isinstance(detection_result, list):
        detections_with_masks = detection_result[0] if detection_result else None
        print(f"Found list with {len(detection_result)} elements")
    else:
        detections_with_masks = detection_result

    if detections_with_masks is not None:
        print(f"Found {len(detections_with_masks)} objects")

        # Check if masks are present
        if hasattr(detections_with_masks, "mask") and detections_with_masks.mask is not None:
            print(f"✓ Masks present with shape: {detections_with_masks.mask.shape}")

            # Visualize if detections are found
            if len(detections_with_masks) > 0:
                print("\nCreating visualization...")
                annotated_image = image.copy()

                # Skip using supervision library for masks in test mode
                print("Skipping supervision library mask annotations in test mode")
                # Apply a simple visualization
                for i in range(len(detections_with_masks)):
                    # Use masks directly if available
                    if detections_with_masks.mask is not None:
                        # Convert mask to boolean if needed
                        if detections_with_masks.mask.dtype != bool:
                            try:
                                mask = detections_with_masks.mask[i].astype(bool)
                            except Exception as e:
                                logger.error(
                                    f"Could not convert mask to boolean: {detections_with_masks.mask.dtype}. Error: {e}"
                                )
                                continue
                        else:
                            mask = detections_with_masks.mask[i]

                        # Apply red tint to mask area
                        if mask.shape == annotated_image.shape[:2]:
                            annotated_image[mask, 2] = 255  # Set red channel to max

                # Draw a basic box for visualization
                if hasattr(detections_with_masks, "xyxy"):
                    for i in range(len(detections_with_masks)):
                        x1, y1, x2, y2 = detections_with_masks.xyxy[i].astype(int)
                        # Draw box (green)
                        cv_image = annotated_image.copy()
                        cv_image = cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        annotated_image = cv_image

                # Save the result
                result_image = Image.fromarray(annotated_image)
                result_image.save("test_segmentation_output.png")
                print("✓ Visualization saved to test_segmentation_output.png")
        else:
            print("✗ No masks found in detections")
    else:
        print("✗ No detections found")

    print("\nTest completed.")


if __name__ == "__main__":
    test_segmentation()
