#!/usr/bin/env python
"""Minimal test to verify per-class metrics work correctly."""

import pytest
import torch
import numpy as np
from pathlib import Path
from rfdetr.util.per_class_metrics import (
    print_per_class_metrics,
    get_coco_category_names,
)
from rfdetr.datasets.coco_eval import CocoEvaluator
from pycocotools.coco import COCO


@pytest.mark.integration
@pytest.mark.quick
def test_per_class_metrics():
    """Test per-class metrics computation with minimal synthetic data."""

    # Load the validation annotations to get the actual categories
    coco_path = "/home/georgepearse/data/cmr/annotations"
    val_ann_file = Path(coco_path) / "2025-05-15_12:38:38.270134_val_ordered.json"

    coco_gt = COCO(val_ann_file)

    # Get category names
    category_names = get_coco_category_names(coco_gt)
    assert len(category_names) > 0, "No categories found in annotations"

    # Create a simple evaluator (bbox only for simplicity)
    evaluator = CocoEvaluator(coco_gt, ["bbox"])

    # Create some dummy predictions for a few images
    img_ids = list(coco_gt.imgs.keys())[:10]  # Use first 10 images
    assert len(img_ids) > 0, "No images found in annotations"

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

    assert len(predictions) == len(img_ids)

    # Update evaluator
    evaluator.update(predictions)

    # Synchronize and compute metrics
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

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

    # Verify that metrics were computed without errors
    assert hasattr(evaluator.coco_eval["bbox"], "stats")
    # The output is printed to stdout, we just verify no exceptions occurred
