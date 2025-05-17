#!/usr/bin/env python3
"""Test script to verify the segmentation head integration"""

import torch
import numpy as np
from PIL import Image
import supervision as sv
import gc

# Import RF-DETR with mask support
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES


def create_dummy_image(height: int = 640, width: int = 640, channels: int = 3) -> np.ndarray:
    """Create a dummy image for testing"""
    # Create a simple image with rectangles
    image = np.ones((height, width, channels), dtype=np.uint8) * 255
    
    # Add some colored rectangles
    image[100:200, 100:250] = [255, 0, 0]  # Red rectangle
    image[300:450, 400:500] = [0, 255, 0]  # Green rectangle
    image[200:350, 200:400] = [0, 0, 255]  # Blue rectangle
    
    return image


def test_basic_segmentation():
    """Test that segmentation head produces outputs"""
    
    print("Testing basic segmentation functionality...")
    
    # Create a dummy image
    dummy_image = create_dummy_image()
    pil_image = Image.fromarray(dummy_image)
    
    # Initialize model
    try:
        model = RFDETRBase()
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return False
    
    # Test detection without masks
    try:
        detections = model.predict(pil_image, threshold=0.5, return_masks=False)
        print(f"✓ Detection without masks successful. Found {len(detections)} objects")
    except Exception as e:
        print(f"✗ Failed to run detection without masks: {e}")
        return False
    
    # Test detection with masks
    try:
        detections_with_masks = model.predict(pil_image, threshold=0.5, return_masks=True)
        print(f"✓ Detection with masks successful. Found {len(detections_with_masks)} objects")
        
        # Check if masks are present
        if hasattr(detections_with_masks, 'mask') and detections_with_masks.mask is not None:
            print(f"✓ Masks are present in detections. Shape: {detections_with_masks.mask.shape}")
        else:
            print("✗ Masks not found in detections")
            return False
            
    except Exception as e:
        print(f"✗ Failed to run detection with masks: {e}")
        return False
    
    return True


def test_mask_dimensions():
    """Test that mask dimensions match the original image"""
    
    print("\nTesting mask dimensions...")
    
    # Create images of different sizes
    test_sizes = [(480, 640), (600, 800), (768, 1024)]
    
    try:
        model = RFDETRBase()
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return False
    
    for height, width in test_sizes:
        print(f"\nTesting image size: {height}x{width}")
        
        # Create dummy image
        dummy_image = create_dummy_image(height=height, width=width)
        pil_image = Image.fromarray(dummy_image)
        
        try:
            # Get predictions with masks
            detections = model.predict(pil_image, threshold=0.5, return_masks=True)
            
            if hasattr(detections, 'mask') and detections.mask is not None:
                mask_shape = detections.mask.shape
                print(f"✓ Mask shape: {mask_shape}")
                
                # Check if masks have correct dimensions
                # masks should be of shape (num_detections, height, width) 
                if len(mask_shape) == 3:
                    mask_height, mask_width = mask_shape[1], mask_shape[2]
                    if mask_height == height and mask_width == width:
                        print(f"✓ Mask dimensions match image size")
                    else:
                        print(f"✗ Mask dimensions ({mask_height}x{mask_width}) don't match image size ({height}x{width})")
                        return False
                else:
                    print(f"✗ Unexpected mask shape: {mask_shape}")
                    return False
            else:
                print("✗ No masks found in detections")
                return False
                
        except Exception as e:
            print(f"✗ Error processing image size {height}x{width}: {e}")
            return False
    
    return True


def test_batch_processing():
    """Test batch processing with segmentation"""
    
    print("\nTesting batch processing with segmentation...")
    
    # Create multiple dummy images  - smaller batch to avoid memory issues
    images = []
    for i in range(2):
        dummy_image = create_dummy_image()
        images.append(Image.fromarray(dummy_image))
    
    try:
        model = RFDETRBase()
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return False
    
    try:
        # Process batch without masks
        detections_list = model.predict(images, threshold=0.5, return_masks=False)
        print(f"✓ Batch detection without masks successful. Processed {len(detections_list)} images")
        
        # Process batch with masks - use torch.cuda.empty_cache() if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        detections_list_with_masks = model.predict(images, threshold=0.5, return_masks=True)
        print(f"✓ Batch detection with masks successful. Processed {len(detections_list_with_masks)} images")
        
        # Check masks for each detection
        for i, detections in enumerate(detections_list_with_masks):
            if hasattr(detections, 'mask') and detections.mask is not None:
                print(f"✓ Image {i}: Masks present with shape {detections.mask.shape}")
            else:
                print(f"✗ Image {i}: No masks found")
                return False
                
    except Exception as e:
        print(f"✗ Failed batch processing: {e}")
        return False
    
    return True


def test_visualization():
    """Test visualization with masks"""
    
    print("\nTesting visualization with masks...")
    
    # Create a dummy image
    dummy_image = create_dummy_image()
    pil_image = Image.fromarray(dummy_image)
    
    try:
        model = RFDETRBase()
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return False
    
    try:
        # Get predictions with masks
        detection_result = model.predict(pil_image, threshold=0.3, return_masks=True)  # Lower threshold for testing
        
        # Handle the detection result properly
        if isinstance(detection_result, list):
            detections = detection_result[0] if detection_result else None
        else:
            detections = detection_result
            
        if detections is None or len(detections) == 0:
            print("⚠ No detections found. Skipping visualization test.")
            return True
        
        # Create annotators
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Create labels
        labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
        
        # Annotate image
        annotated_image = dummy_image.copy()
        
        # Apply mask annotations if present
        if hasattr(detections, 'mask') and detections.mask is not None:
            annotated_image = mask_annotator.annotate(annotated_image, detections)
            print("✓ Mask annotation successful")
        else:
            print("⚠ No masks to annotate")
        
        # Add boxes and labels
        annotated_image = box_annotator.annotate(annotated_image, detections)
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)
        
        print("✓ Visualization successful")
        
    except Exception as e:
        print(f"✗ Failed visualization: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    
    print("RF-DETR Segmentation Integration Tests")
    print("=" * 40)
    
    # Clear GPU memory before starting tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    tests = [
        ("Basic Segmentation", test_basic_segmentation),
        ("Mask Dimensions", test_mask_dimensions),
        ("Batch Processing", test_batch_processing),
        ("Visualization", test_visualization),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        success = test_func()
        results.append((test_name, success))
        
        # Clean up GPU memory after each test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary")
    print("=" * 40)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)