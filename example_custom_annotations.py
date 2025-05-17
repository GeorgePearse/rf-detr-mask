#!/usr/bin/env python
"""
Example showing how to use custom COCO annotation files with RF-DETR
"""

import argparse
from pathlib import Path

def create_custom_args():
    """Create an example args object with custom COCO annotations"""
    
    parser = argparse.ArgumentParser()
    
    # Standard arguments
    parser.add_argument('--dataset', default='coco', type=str)
    parser.add_argument('--dataset_file', default='coco', type=str)
    
    # Key arguments for custom annotations
    parser.add_argument('--coco_path', default='/home/georgepearse/data/cmr/annotations',
                      help='Path to the directory containing annotation files')
    parser.add_argument('--coco_train', default='2025-05-15_12:38:23.077836_train_ordered.json',
                      help='Name of the training annotation file')
    parser.add_argument('--coco_val', default='2025-05-15_12:38:38.270134_val_ordered.json',
                      help='Name of the validation annotation file')
    parser.add_argument('--coco_img_path', default='/home/georgepearse/data/images',
                      help='Path to the images directory')
    
    # Other required arguments
    parser.add_argument('--multi_scale', default=False, type=bool)
    parser.add_argument('--expanded_scales', default=False, type=bool)
    
    return parser.parse_args()


def main():
    args = create_custom_args()
    
    print("Custom COCO annotation configuration:")
    print(f"Annotations directory: {args.coco_path}")
    print(f"Training annotations: {args.coco_train}")
    print(f"Validation annotations: {args.coco_val}")
    print(f"Images directory: {args.coco_img_path}")
    
    # Example of how the build function would use these
    print("\nHow the build function processes these:")
    
    # For training
    train_ann_file = Path(args.coco_path) / args.coco_train
    print(f"Training annotation path: {train_ann_file}")
    print(f"Training image path: {args.coco_img_path}")
    
    # For validation
    val_ann_file = Path(args.coco_path) / args.coco_val
    print(f"Validation annotation path: {val_ann_file}")
    print(f"Validation image path: {args.coco_img_path}")


if __name__ == '__main__':
    main()