#!/usr/bin/env python
"""Minimal test to verify per-class metrics work correctly."""

import torch
import numpy as np
from pathlib import Path
from rfdetr.util.per_class_metrics import (
    print_per_class_metrics,
    get_coco_category_names,
)
from rfdetr.datasets.coco_eval import CocoEvaluator
from pycocotools.coco import COCO


def create_minimal_test():
    """Create a minimal test case for per-class metrics."""

    print("Creating minimal test for per-class metrics...")
    print("=" * 80)

    # Load the validation annotations to get the actual categories
    coco_path = "/home/georgepearse/data/cmr/annotations"
    val_ann_file = Path(coco_path) / "2025-05-15_12:38:38.270134_val_ordered.json"

    print(f"Loading annotations from: {val_ann_file}")
    coco_gt = COCO(val_ann_file)

    # Get category names
    category_names = get_coco_category_names(coco_gt)
    print(f"\nFound {len(category_names)} categories")
    print(f"First 10 categories: {list(category_names.items())[:10]}")

    # Create a simple evaluator (bbox only for simplicity)
    print("\nCreating COCO evaluator...")
    evaluator = CocoEvaluator(coco_gt, ["bbox"])

    # Create some dummy predictions for a few images
    print("\nCreating dummy predictions...")
    img_ids = list(coco_gt.imgs.keys())[:10]  # Use first 10 images

    predictions = {}
    for img_id in img_ids:
        # Create some random predictions
        num_preds = np.random.randint(5, 15)
        scores = np.random.rand(num_preds)
        labels = np.random.randint(0, len(category_names), num_preds)
        boxes = np.random.rand(num_preds, 4) * 100

        # Ensure boxes are in correct format [x1, y1, w, h]
        boxes[:, 2:] = boxes[:, 2:] + 10  # width and height

        predictions[img_id] = {
            "scores": torch.tensor(scores),
            "labels": torch.tensor(labels),
            "boxes": torch.tensor(boxes),
        }

    print(f"Created predictions for {len(predictions)} images")

    # Update evaluator
    print("\nUpdating evaluator with predictions...")
    evaluator.update(predictions)

    # Synchronize and compute metrics
    print("\nComputing metrics...")
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    # Print per-class metrics
    print("\nPrinting per-class metrics...")

    # Print bbox metrics
    print_per_class_metrics(
        evaluator,
        iou_type="bbox",
        class_names=category_names,
        metric_name="AP",
        iou_threshold=0.5,  # AP@0.5
        max_dets=100,
        area_range="all",
    )

    print("\n✅ SUCCESS: Per-class metrics test completed!")
    print("=" * 80)


if __name__ == "__main__":
    create_minimal_test()
