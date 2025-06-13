#!/usr/bin/env python3
"""
RF-DETR-MASK Inference Script for Validation Split

This script runs inference with RF-DETR-MASK on a validation dataset and computes
evaluation metrics. It supports both detection and segmentation outputs.

Usage:
    python inference_validation.py --checkpoint path/to/checkpoint.pth \
        --config rfdetr_base --num-classes 3
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import torch.utils.data
from tqdm import tqdm
import pycocotools.mask as mask_util
import numpy as np

from rfdetr import build_model, RFDETRBaseConfig, RFDETRLargeConfig
from rfdetr.datasets import build_dataset
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.models import build_criterion_and_postprocessors
from rfdetr.util.misc import nested_tensor_from_tensor_list


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RF-DETR-MASK Inference on Validation Dataset"
    )

    # Model configuration
    parser.add_argument(
        "--config",
        type=str,
        default="rfdetr_base",
        choices=["rfdetr_base", "rfdetr_large"],
        help="Model configuration to use",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=80,
        help="Number of object classes (excluding background)",
    )

    # Dataset configuration
    parser.add_argument(
        "--val-annotations",
        type=str,
        default="/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json",
        help="Path to validation COCO-format annotations file",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/home/georgepearse/data/images",
        help="Path to images directory",
    )

    # Inference configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--num-select",
        type=int,
        default=300,
        help="Number of top predictions to select per image",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference_results",
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions in COCO format",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization images",
    )
    parser.add_argument(
        "--eval-bbox",
        action="store_true",
        default=True,
        help="Evaluate bounding box predictions",
    )
    parser.add_argument(
        "--eval-segm",
        action="store_true",
        help="Evaluate segmentation predictions",
    )

    return parser.parse_args()


def load_model(args) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load model from checkpoint.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    # Build model configuration
    if args.config == "rfdetr_base":
        config = RFDETRBaseConfig()
    else:
        config = RFDETRLargeConfig()

    # Update number of classes
    config.num_classes = args.num_classes

    # Build model
    model = build_model(config, return_masks=args.eval_segm)
    model = model.to(args.device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove module. prefix if present (from DataParallel)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, checkpoint


def build_dataloader(args) -> torch.utils.data.DataLoader:
    """Build validation dataloader.

    Args:
        args: Command line arguments

    Returns:
        DataLoader for validation dataset
    """
    # Build dataset
    dataset_val = build_dataset(
        image_set="val",
        args=argparse.Namespace(
            coco_path=args.images_dir,
            coco_anno=args.val_annotations,
            dataset_file="coco",
            masks=args.eval_segm,
            resolution=640,  # Default resolution
            expanded_mask_scales=False,
        ),
    )

    # Build dataloader
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    return data_loader_val, dataset_val


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    postprocessors: Dict[str, torch.nn.Module],
    device: torch.device,
    args,
) -> List[Dict[str, Any]]:
    """Run inference on validation dataset.

    Args:
        model: Model to run inference with
        data_loader: Validation data loader
        postprocessors: Post-processing modules
        device: Device to run on
        args: Command line arguments

    Returns:
        List of predictions for all images
    """
    model.eval()

    all_predictions = []

    print("Running inference...")
    for samples, targets in tqdm(data_loader, desc="Processing batches"):
        # Move to device
        samples = nested_tensor_from_tensor_list(
            [sample.to(device) for sample in samples]
        )

        # Forward pass
        outputs = model(samples)

        # Get original image sizes
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(
            device
        )

        # Post-process outputs
        results = postprocessors["bbox"](outputs, orig_target_sizes)

        # Add image IDs to results
        for i, (result, target) in enumerate(zip(results, targets)):
            result["image_id"] = target["image_id"].item()
            all_predictions.append(result)

    return all_predictions


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    dataset,
    iou_types: List[str],
) -> Dict[str, float]:
    """Evaluate predictions using COCO metrics.

    Args:
        predictions: List of predictions
        dataset: Dataset with COCO annotations
        iou_types: Types of IoU to evaluate ("bbox", "segm")

    Returns:
        Dictionary of evaluation metrics
    """
    # Create COCO evaluator
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    # Format predictions for COCO evaluator
    for pred in predictions:
        image_id = pred["image_id"]
        coco_evaluator.update({image_id: pred})

    # Accumulate and summarize
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # Extract metrics
    metrics = {}
    for iou_type in iou_types:
        if iou_type in coco_evaluator.coco_eval:
            stats = coco_evaluator.coco_eval[iou_type].stats
            metrics[f"{iou_type}_AP"] = stats[0]
            metrics[f"{iou_type}_AP50"] = stats[1]
            metrics[f"{iou_type}_AP75"] = stats[2]
            metrics[f"{iou_type}_APs"] = stats[3]
            metrics[f"{iou_type}_APm"] = stats[4]
            metrics[f"{iou_type}_APl"] = stats[5]

    return metrics, coco_evaluator


def save_predictions(predictions: List[Dict[str, Any]], output_path: Path):
    """Save predictions in COCO format.

    Args:
        predictions: List of predictions
        output_path: Path to save predictions
    """
    coco_predictions = []

    for pred in predictions:
        image_id = pred["image_id"]
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()

        # Convert each detection
        for i in range(len(boxes)):
            # Convert from xyxy to xywh
            x1, y1, x2, y2 = boxes[i]
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            coco_pred = {
                "image_id": int(image_id),
                "category_id": int(labels[i]),
                "bbox": bbox,
                "score": float(scores[i]),
            }

            # Add segmentation if available
            if "masks" in pred:
                mask = pred["masks"][i].cpu().numpy()
                # Convert mask to RLE format
                rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                coco_pred["segmentation"] = rle

            coco_predictions.append(coco_pred)

    # Save to file
    with open(output_path, "w") as f:
        json.dump(coco_predictions, f)

    print(f"Saved {len(coco_predictions)} predictions to {output_path}")


def main():
    """Main inference function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    model, checkpoint = load_model(args)

    # Build dataloader
    data_loader, dataset = build_dataloader(args)

    # Build criterion and postprocessors (we only need postprocessors)
    _, postprocessors = build_criterion_and_postprocessors(
        args=argparse.Namespace(
            num_classes=args.num_classes,
            matcher="HungarianMatcher",
            set_cost_class=2,
            set_cost_bbox=5,
            set_cost_giou=2,
            bbox_loss_coef=5,
            giou_loss_coef=2,
            focal_alpha=0.25,
            sum_group_losses=True,
            eval_size=640,
            masks=args.eval_segm,
            mask_loss_coef=1.0,
            dice_loss_coef=1.0,
        )
    )

    # Update postprocessor num_select
    postprocessors["bbox"].num_select = args.num_select

    # Run inference
    device = torch.device(args.device)
    start_time = time.time()
    predictions = run_inference(model, data_loader, postprocessors, device, args)
    inference_time = time.time() - start_time

    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Average time per image: {inference_time / len(predictions):.3f} seconds")

    # Evaluate predictions
    iou_types = []
    if args.eval_bbox:
        iou_types.append("bbox")
    if args.eval_segm:
        iou_types.append("segm")

    if iou_types:
        print("\nEvaluating predictions...")
        metrics, evaluator = evaluate_predictions(predictions, dataset, iou_types)

        # Print metrics
        print("\nEvaluation Results:")
        print("-" * 50)
        for metric_name, value in metrics.items():
            print(f"{metric_name:20s}: {value:.3f}")

        # Save metrics
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to {metrics_path}")

    # Save predictions if requested
    if args.save_predictions:
        predictions_path = output_dir / "predictions.json"
        save_predictions(predictions, predictions_path)

    print("\nInference complete!")


if __name__ == "__main__":
    main()
