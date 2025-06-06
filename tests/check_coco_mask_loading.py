#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Script to check if masks are being loaded correctly from COCO annotations.
This is a debugging script to help diagnose issues with mask loading.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

# Add the parent directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfdetr.datasets.coco import CocoDetection, ConvertCoco
import rfdetr.datasets.transforms as T


def parse_args():
    parser = argparse.ArgumentParser('Check COCO Mask Loading')
    parser.add_argument('--coco_path', type=str, required=True,
                        help='Path to COCO dataset')
    parser.add_argument('--output_dir', type=str, default='mask_check_output',
                        help='Directory to save output visualizations')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of images to check')
    return parser.parse_args()


def visualize_masks(image, target, output_path):
    """Visualize the image with overlaid masks."""
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
        # Scale to 0-255 range if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create figure for visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image with bounding boxes
    ax[0].imshow(image)
    ax[0].set_title('Original Image with Boxes')
    
    # Get boxes
    if 'boxes' in target:
        boxes = target['boxes']
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.numpy()
        
        # Draw boxes
        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, 
                                 fill=False, edgecolor='red', linewidth=1)
            ax[0].add_patch(rect)
    
    # Image with masks
    ax[1].imshow(image)
    ax[1].set_title('Image with Masks')
    
    # Get masks
    if 'masks' in target:
        masks = target['masks']
        if isinstance(masks, torch.Tensor):
            masks = masks.numpy()
        
        # Draw masks
        for i, mask in enumerate(masks):
            # Create a colored mask
            color_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            color = np.random.rand(3)
            color_mask[mask] = [*color, 0.5]  # RGBA with 0.5 alpha
            ax[1].imshow(color_mask)
    else:
        ax[1].text(10, 30, "No masks found", color='red', fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def examine_original_annotations(coco_api, img_id, output_dir):
    """Examine the original annotations for a given image ID."""
    # Get image info
    img_info = coco_api.loadImgs(img_id)[0]
    
    # Get annotations for the image
    ann_ids = coco_api.getAnnIds(imgIds=img_id)
    anns = coco_api.loadAnns(ann_ids)
    
    # Check for segmentation
    has_segmentation = False
    for ann in anns:
        if 'segmentation' in ann:
            has_segmentation = True
            break
    
    # Create output
    result = {
        'img_id': img_id,
        'file_name': img_info['file_name'],
        'has_segmentation': has_segmentation,
        'num_annotations': len(anns),
        'annotations': []
    }
    
    # Add each annotation
    for ann in anns:
        ann_result = {
            'category_id': ann['category_id'],
            'bbox': ann.get('bbox', None),
            'has_segmentation': 'segmentation' in ann,
        }
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):
                ann_result['segmentation_type'] = 'polygon'
                ann_result['polygon_count'] = len(ann['segmentation'])
            else:
                ann_result['segmentation_type'] = 'RLE'
        
        result['annotations'].append(ann_result)
    
    # Save the result
    output_path = os.path.join(output_dir, f'annotation_info_{img_id}.txt')
    with open(output_path, 'w') as f:
        f.write(f"Image ID: {result['img_id']}\n")
        f.write(f"File name: {result['file_name']}\n")
        f.write(f"Has segmentation: {result['has_segmentation']}\n")
        f.write(f"Number of annotations: {result['num_annotations']}\n\n")
        
        for i, ann in enumerate(result['annotations']):
            f.write(f"Annotation {i}:\n")
            f.write(f"  Category ID: {ann['category_id']}\n")
            if ann['bbox'] is not None:
                f.write(f"  Bounding box: {ann['bbox']}\n")
            f.write(f"  Has segmentation: {ann['has_segmentation']}\n")
            if ann['has_segmentation']:
                f.write(f"  Segmentation type: {ann.get('segmentation_type', 'Unknown')}\n")
                if 'polygon_count' in ann:
                    f.write(f"  Polygon count: {ann['polygon_count']}\n")
            f.write("\n")
    
    return result


def convert_coco_check(args):
    """Check if the ConvertCoco class handles masks correctly."""
    # Paths
    coco_path = Path(args.coco_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get paths for COCO validation set
    img_folder = coco_path / 'val2017'
    ann_file = coco_path / 'annotations' / 'instances_val2017.json'
    
    # Load COCO API for examining original annotations
    coco_api = COCO(ann_file)
    
    # Create transforms
    transforms = T.Compose([
        T.RandomResize([640], max_size=1333),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = CocoDetection(img_folder, ann_file, transforms)
    
    # Create a patched version of ConvertCoco to diagnose issues
    original_call = ConvertCoco.__call__
    
    def patched_call(self, image, target):
        # Print target info before conversion
        print(f"Original target keys: {list(target.keys())}")
        if 'annotations' in target:
            print(f"Number of annotations: {len(target['annotations'])}")
            for i, ann in enumerate(target['annotations']):
                print(f"Annotation {i} keys: {list(ann.keys())}")
                if 'segmentation' in ann:
                    print(f"  Segmentation type: {type(ann['segmentation'])}")
                    if isinstance(ann['segmentation'], list):
                        print(f"  Number of polygons: {len(ann['segmentation'])}")
        
        # Call original method
        image, converted_target = original_call(self, image, target)
        
        # Print target info after conversion
        print(f"Converted target keys: {list(converted_target.keys())}")
        if 'masks' in converted_target:
            print(f"Masks shape: {converted_target['masks'].shape}")
            print(f"Masks type: {converted_target['masks'].dtype}")
        else:
            print("No masks found in converted target")
        
        return image, converted_target
    
    # Use our patched method for diagnosis
    ConvertCoco.__call__ = patched_call
    
    # Check a few images
    for i in range(min(args.num_images, len(dataset))):
        print(f"\n--- Checking image {i} ---")
        try:
            # Get image ID
            img_id = dataset.ids[i]
            print(f"Image ID: {img_id}")
            
            # Examine original annotations
            ann_info = examine_original_annotations(coco_api, img_id, output_dir)
            print(f"Original annotation info saved to: {output_dir}/annotation_info_{img_id}.txt")
            
            # Get image and target
            image, target = dataset[i]
            
            # Visualize
            output_path = output_dir / f'image_{img_id}.png'
            visualize_masks(image, target, output_path)
            print(f"Visualization saved to: {output_path}")
            
            # Check mask statistics if available
            if 'masks' in target:
                masks = target['masks']
                print(f"Mask statistics:")
                print(f"  Number of masks: {len(masks)}")
                print(f"  Mask shape: {masks.shape}")
                print(f"  Mask dtype: {masks.dtype}")
                print(f"  Mask min value: {masks.min().item()}")
                print(f"  Mask max value: {masks.max().item()}")
                
                # Count non-empty masks
                non_empty = 0
                for mask in masks:
                    if mask.any():
                        non_empty += 1
                print(f"  Non-empty masks: {non_empty} out of {len(masks)}")
            else:
                print("No masks available in the target.")
        
        except Exception as e:
            print(f"Error processing image {i}: {e}")
    
    # Restore original method
    ConvertCoco.__call__ = original_call


def suggest_coco_fix():
    """Suggest a fix for ConvertCoco to handle masks properly."""
    print("\n--- Suggested Fix for ConvertCoco ---")
    
    suggested_code = """
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # Handle segmentation masks if available
        if anno and "segmentation" in anno[0]:
            masks = []
            for obj in anno:
                if 'segmentation' in obj:
                    if isinstance(obj['segmentation'], list):
                        # Polygon format
                        rles = mask_util.frPyObjects(obj['segmentation'], h, w)
                        rle = mask_util.merge(rles)
                    else:
                        # RLE format
                        rle = obj['segmentation']
                    mask = mask_util.decode(rle)
                    masks.append(mask)
                else:
                    # Create an empty mask if segmentation is missing
                    masks.append(np.zeros((h, w), dtype=np.uint8))
            
            masks = torch.as_tensor(np.stack(masks), dtype=torch.bool)
        else:
            masks = None

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if masks is not None:
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        
        # Add masks to target if available
        if masks is not None:
            target["masks"] = masks

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target
    """
    
    print(suggested_code)
    print("\nTo implement this fix:")
    print("1. Import pycocotools.mask in coco.py: 'import pycocotools.mask as mask_util'")
    print("2. Replace the current __call__ method in ConvertCoco with the above implementation")
    print("3. This will properly handle both polygon and RLE segmentation formats")


if __name__ == "__main__":
    args = parse_args()
    convert_coco_check(args)
    suggest_coco_fix()