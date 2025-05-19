#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Create modified version of validation annotations with randomly flipped classifications
to use as test predictions for evaluation.
"""

import argparse
import json
import random
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser("Create Test Annotations", add_help=True)
    parser.add_argument(
        "--coco_path", type=str, required=True, help="Path to the annotations directory"
    )
    parser.add_argument(
        "--coco_val", type=str, required=True, help="Validation annotation file name"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="test_predictions.json",
        help="Output file name for modified annotations",
    )
    parser.add_argument(
        "--flip_ratio",
        type=float,
        default=0.3,
        help="Percentage of annotations to flip classifications for (0.0-1.0)",
    )
    return parser


def main(args):
    # Load the validation annotations
    val_path = Path(args.coco_path) / args.coco_val
    print(f"Looking for validation file at: {val_path}")
    try:
        with open(val_path) as f:
            annotations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find validation file at {val_path}")
        # If the file doesn't exist but we're creating test predictions, create a sample file
        if not Path(val_path).exists():
            print(f"Creating a sample validation file at {val_path}")
            annotations = {
                "images": [{"id": 1, "file_name": "test_image.jpg", "width": 640, "height": 480}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [100, 100, 200, 150],
                        "area": 30000,
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 2,
                        "bbox": [300, 200, 100, 100],
                        "area": 10000,
                        "iscrowd": 0,
                    },
                ],
                "categories": [
                    {"id": 1, "name": "person", "supercategory": "none"},
                    {"id": 2, "name": "car", "supercategory": "none"},
                    {"id": 3, "name": "dog", "supercategory": "none"},
                ],
            }
            # Save the sample file
            val_path.parent.mkdir(parents=True, exist_ok=True)
            with open(val_path, "w") as f:
                json.dump(annotations, f, indent=2)

    # Extract category IDs for random assignment
    category_ids = [cat["id"] for cat in annotations.get("categories", [])]
    if not category_ids:
        raise ValueError("No categories found in the annotation file")

    # Create a copy of the annotations
    test_annotations = annotations.copy()

    # Randomly flip classifications for annotations
    for i, ann in enumerate(test_annotations.get("annotations", [])):
        if random.random() < args.flip_ratio:
            # Get a random category ID different from the current one
            available_cats = [cat_id for cat_id in category_ids if cat_id != ann["category_id"]]
            if available_cats:
                ann["category_id"] = random.choice(available_cats)

    # Save the modified annotations
    output_path = Path(args.coco_path) / args.output_file
    with open(output_path, "w") as f:
        json.dump(test_annotations, f)

    print(f"Created modified annotations file at {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
