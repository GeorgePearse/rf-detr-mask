#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Evaluation script for RF-DETR model that calculates and returns per-class mAP50 metrics.
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import rfdetr.util.misc as utils
from rfdetr.config_utils import load_config
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.models import build_model, build_criterion_and_postprocessors
from rfdetr.util.logging_config import get_logger

logger = get_logger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser("RF-DETR Model Evaluation", add_help=True)
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    
    # Configuration options
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation results")
    
    # Dataset parameters
    parser.add_argument("--dataset_file", default="coco", type=str, help="Dataset format")
    parser.add_argument("--coco_path", type=str, help="Path to the annotations directory")
    parser.add_argument("--coco_val", type=str, help="Validation annotation file name")
    parser.add_argument("--coco_img_path", type=str, help="Path to the images directory")
    
    # Model parameters
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    parser.add_argument("--resolution", type=int, default=560, help="Input resolution for the model")
    
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", default="cuda", help="Device to run evaluation on (cuda/cpu)")
    parser.add_argument("--fp16_eval", action="store_true", help="Use FP16 precision for evaluation")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-class metrics and confidence thresholds")
    parser.add_argument("--test_limit", type=int, help="Limit the number of samples in the validation dataset")
    
    return parser


def evaluate_with_per_class_metrics(model, criterion, postprocessors, data_loader, base_ds, device, args=None):
    """
    Evaluate the model and return per-class metrics.
    This is an extended version of rfdetr.engine.evaluate that reports per-class results.
    """
    model.eval()
    if args and args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors)
    coco_evaluator = rfdetr.datasets.coco_eval.CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args and args.fp16_eval:
            samples.tensors = samples.tensors.half()

        # Run evaluation with autocast
        with torch.cuda.amp.autocast(enabled=args.fp16_eval if args else False):
            outputs = model(samples)

        # Convert outputs back to float if using FP16
        if args and args.fp16_eval:
            for key in outputs:
                if key == "enc_outputs":
                    for sub_key in outputs[key]:
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx]:
                            outputs[key][idx][sub_key] = outputs[key][idx][sub_key].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # Reduce losses for logging
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        # Process predictions and update evaluator
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # Gather stats from all processes in distributed setup
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # Accumulate predictions and calculate metrics
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # Extract statistics
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in postprocessors:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    # Calculate per-class metrics
    per_class_metrics = extract_per_class_metrics(coco_evaluator, base_ds)
    stats["per_class_metrics"] = per_class_metrics
    
    return stats, coco_evaluator


def extract_per_class_metrics(coco_evaluator, base_ds):
    """
    Extract per-class metrics from COCO evaluator.
    Returns a dictionary mapping class IDs to their metrics.
    """
    if coco_evaluator is None or "bbox" not in coco_evaluator.coco_eval:
        return {}
    
    # Get class names from the dataset
    categories = base_ds.cats if hasattr(base_ds, "cats") else {}
    class_names = {cat["id"]: cat["name"] for cat in categories.values()} if categories else {}
    
    # Get the evaluation results
    eval_results = coco_evaluator.coco_eval["bbox"]
    
    # Extract per-class precision at IoU=0.5 (index 0 is for IoU threshold 0.5)
    # The precision array has shape [T, R, K, A, M] where:
    # T: IoU thresholds (default: 10 thresholds from 0.5 to 0.95)
    # R: recall thresholds (default: 101 points from 0 to 1)
    # K: category (default: 80 COCO categories)
    # A: area range (default: 4 ranges - all, small, medium, large)
    # M: max detections (default: 3 values - 1, 10, 100)
    precision = eval_results.eval["precision"]
    
    # We're interested in IoU=0.5, all area ranges, all max detections
    # Therefore: t=0 (0.5 IoU), a=0 (all areas), m=2 (100 detections)
    precision_50 = precision[0, :, :, 0, 2]  # shape: [R, K]
    
    per_class_metrics = {}
    
    # The coco_eval object contains catIds which is a list of category IDs
    cat_ids = eval_results.params.catIds
    
    for idx, cat_id in enumerate(cat_ids):
        # Get precision at all recall levels for this class
        precision_per_class = precision_50[:, idx]
        
        # Calculate AP50 for this class (mean precision over recall levels)
        # We filter out -1 values which represent undefined precision
        valid_precision = precision_per_class[precision_per_class > -1]
        ap50 = np.mean(valid_precision) if len(valid_precision) > 0 else 0.0
        
        # Get class name if available
        class_name = class_names.get(cat_id, f"class_{cat_id}")
        
        per_class_metrics[cat_id] = {
            "id": cat_id,
            "name": class_name,
            "ap50": float(ap50)  # Convert to Python float for JSON serialization
        }
    
    return per_class_metrics


def main(args):
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract model configuration from checkpoint
    checkpoint_args = checkpoint.get("args", None)
    if checkpoint_args is None:
        if not args.config:
            raise ValueError("No configuration found in checkpoint and no config file provided.")
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        model_args = config.to_args()
    else:
        logger.info("Using configuration from checkpoint")
        model_args = checkpoint_args
    
    # Override model arguments with command line arguments
    if args.num_classes:
        model_args.num_classes = args.num_classes
    if args.resolution:
        model_args.resolution = args.resolution
    if args.coco_path:
        model_args.coco_path = args.coco_path
    if args.coco_val:
        model_args.coco_val = args.coco_val
    if args.coco_img_path:
        model_args.coco_img_path = args.coco_img_path
    
    # Set additional parameters for evaluation
    model_args.device = args.device
    model_args.batch_size = args.batch_size
    model_args.num_workers = args.num_workers
    model_args.fp16_eval = args.fp16_eval
    model_args.eval = True
    
    # Add test_limit parameter if provided
    if args.test_limit is not None:
        model_args.test_limit = args.test_limit
        logger.info(f"Limiting validation dataset to {args.test_limit} samples")
    
    # Build model
    logger.info("Building model...")
    model = build_model(model_args)
    model.to(device)
    
    # Load model weights
    model_state_dict = checkpoint["model"]
    model.load_state_dict(model_state_dict, strict=True)
    logger.info("Model loaded successfully")
    
    # Build criterion and postprocessors
    criterion, postprocessors = build_criterion_and_postprocessors(model_args)
    
    # Build validation dataset
    logger.info("Building validation dataset...")
    dataset_val = build_dataset(image_set="val", args=model_args, resolution=model_args.resolution)
    sampler_val = SequentialSampler(dataset_val)
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    
    # Get COCO API from dataset for evaluation
    base_ds = get_coco_api_from_dataset(dataset_val)
    
    # Run evaluation with per-class metrics
    logger.info("Running evaluation...")
    stats, coco_evaluator = evaluate_with_per_class_metrics(
        model, criterion, postprocessors, data_loader_val, base_ds, device, args
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract key metrics
    main_metrics = {
        "map": stats["coco_eval_bbox"][0] if "coco_eval_bbox" in stats else 0,
        "map50": stats["coco_eval_bbox"][1] if "coco_eval_bbox" in stats else 0,
        "map75": stats["coco_eval_bbox"][2] if "coco_eval_bbox" in stats else 0,
    }
    
    if "coco_eval_masks" in stats:
        main_metrics.update({
            "mask_map": stats["coco_eval_masks"][0],
            "mask_map50": stats["coco_eval_masks"][1],
            "mask_map75": stats["coco_eval_masks"][2],
        })
    
    # Save the per-class metrics to a JSON file
    per_class = stats.get("per_class_metrics", {})
    
    # Print the metrics
    logger.info("\n===== EVALUATION RESULTS =====")
    logger.info(f"Overall mAP: {main_metrics['map']:.4f}")
    logger.info(f"Overall mAP50: {main_metrics['map50']:.4f}")
    logger.info(f"Overall mAP75: {main_metrics['map75']:.4f}")
    
    if "mask_map" in main_metrics:
        logger.info(f"Overall Mask mAP: {main_metrics['mask_map']:.4f}")
        logger.info(f"Overall Mask mAP50: {main_metrics['mask_map50']:.4f}")
        logger.info(f"Overall Mask mAP75: {main_metrics['mask_map75']:.4f}")
    
    # Print per-class metrics
    logger.info("\n===== PER-CLASS AP50 METRICS =====")
    # Sort classes by AP50 performance
    sorted_classes = sorted(
        per_class.values(), 
        key=lambda x: x["ap50"], 
        reverse=True
    )
    
    for class_info in sorted_classes:
        logger.info(f"Class {class_info['id']} ({class_info['name']}): AP50 = {class_info['ap50']:.4f}")
    
    # Save results to file
    results = {
        "overall_metrics": main_metrics,
        "per_class_metrics": per_class
    }
    
    output_file = output_dir / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    
    return main_metrics["map50"], per_class


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)