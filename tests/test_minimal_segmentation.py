#!/usr/bin/env python3
"""Minimal test script to verify segmentation functionality"""

import torch
import numpy as np
from PIL import Image
import supervision as sv

# Import RF-DETR with mask support
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES


def test_segmentation():
    """Test basic segmentation functionality"""
    
    print("Testing RF-DETR-MASK segmentation...")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create a dummy image
    image = np.ones((640, 640, 3), dtype=np.uint8) * 255
    image[100:300, 100:400] = [255, 0, 0]  # Red rectangle
    pil_image = Image.fromarray(image)
    
    # Initialize model on CPU to avoid memory issues
    print("Initializing model on CPU...")
    model = RFDETRBase(device='cpu')
    
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
        if hasattr(detections_with_masks, 'mask') and detections_with_masks.mask is not None:
            print(f"✓ Masks present with shape: {detections_with_masks.mask.shape}")
            
            # Visualize if detections are found
            if len(detections_with_masks) > 0:
                print("\nCreating visualization...")
                annotated_image = image.copy()
                
                # Create annotators
                box_annotator = sv.BoxAnnotator()
                mask_annotator = sv.MaskAnnotator()
                label_annotator = sv.LabelAnnotator()
                
                # Create labels
                labels = [
                    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                    for class_id, confidence
                    in zip(detections_with_masks.class_id, detections_with_masks.confidence)
                ]
                
                # Apply annotations
                annotated_image = mask_annotator.annotate(annotated_image, detections_with_masks)
                annotated_image = box_annotator.annotate(annotated_image, detections_with_masks)
                annotated_image = label_annotator.annotate(annotated_image, detections_with_masks, labels)
                
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