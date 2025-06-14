"""
Utilities for computing and displaying per-class COCO metrics.
"""

from typing import Dict, Optional, Any
import numpy as np
import pandas as pd


def print_per_class_metrics(
    coco_evaluator: Any,
    iou_type: str = "bbox",
    class_names: Optional[Dict[int, str]] = None,
    metric_name: str = "AP",
    iou_threshold: Optional[float] = None,
    max_dets: int = 100,
    area_range: str = "all",
) -> Dict[int, float]:
    """
    Print per-class COCO metrics (AP/AR) and return them as a dictionary.

    Args:
        coco_evaluator: CocoEvaluator instance after accumulate() and summarize()
        iou_type: Type of evaluation ('bbox' or 'segm')
        class_names: Optional dict mapping category IDs to names
        metric_name: 'AP' for Average Precision or 'AR' for Average Recall
        iou_threshold: IoU threshold (None for AP@[0.5:0.95], 0.5 for AP@0.5, etc.)
        max_dets: Maximum detections per image
        area_range: 'all', 'small', 'medium', or 'large'

    Returns:
        Dictionary mapping category IDs to metric values
    """
    if iou_type not in coco_evaluator.coco_eval:
        print(f"IoU type '{iou_type}' not available in evaluator")
        return {}

    per_class_metrics: Dict[int, float] = {}

    # Get the DataFrame representation
    df = get_per_class_metrics_dataframe(
        coco_evaluator,
        iou_type,
        class_names,
        metric_name,
        iou_threshold,
        max_dets,
        area_range,
    )

    if df.empty:
        print(f"No metrics available for {iou_type}")
        return per_class_metrics

    print(f"\n{'='*80}")
    print(
        f"Per-Class {metric_name} @ IoU={iou_threshold if iou_threshold else '[0.5:0.95]'} | area={area_range} | maxDets={max_dets}"
    )
    print(f"{'='*80}")

    # Convert DataFrame metrics to dictionary
    for _, row in df.iterrows():
        per_class_metrics[row["class_id"]] = row[metric_name]

    # Print the DataFrame with better formatting
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.3f}".format)

    print(df.to_string(index=False))

    # Print summary statistics
    if not df.empty:
        print(f"\n{'-'*80}")
        print("Summary Statistics:")
        print(f"{'Mean':30s}: {df[metric_name].mean():.3f}")
        print(f"{'Std':30s}: {df[metric_name].std():.3f}")
        print(f"{'Min':30s}: {df[metric_name].min():.3f}")
        print(f"{'Max':30s}: {df[metric_name].max():.3f}")
        print(f"{'Number of classes':30s}: {len(df)}")

    print(f"{'='*80}\n")

    return per_class_metrics


def get_coco_category_names(coco_gt) -> Dict[int, str]:
    """
    Extract category names from COCO ground truth object.

    Args:
        coco_gt: COCO ground truth object

    Returns:
        Dictionary mapping category IDs to names
    """
    if hasattr(coco_gt, "cats"):
        return {cat_id: cat_info["name"] for cat_id, cat_info in coco_gt.cats.items()}
    return {}


def get_per_class_metrics_dataframe(
    coco_evaluator: Any,
    iou_type: str = "bbox",
    class_names: Optional[Dict[int, str]] = None,
    metric_name: str = "AP",
    iou_threshold: Optional[float] = None,
    max_dets: int = 100,
    area_range: str = "all",
) -> pd.DataFrame:
    """
    Get per-class COCO metrics as a pandas DataFrame.

    Args:
        coco_evaluator: CocoEvaluator instance after accumulate() and summarize()
        iou_type: Type of evaluation ('bbox' or 'segm')
        class_names: Optional dict mapping category IDs to names
        metric_name: 'AP' for Average Precision or 'AR' for Average Recall
        iou_threshold: IoU threshold (None for AP@[0.5:0.95], 0.5 for AP@0.5, etc.)
        max_dets: Maximum detections per image
        area_range: 'all', 'small', 'medium', or 'large'

    Returns:
        DataFrame with columns: class_id, class_name, metric_value
    """
    if iou_type not in coco_evaluator.coco_eval:
        return pd.DataFrame()

    coco_eval = coco_evaluator.coco_eval[iou_type]
    params = coco_eval.params

    # Find the indices for the requested configuration
    if iou_threshold is None:
        iou_idx = slice(None)
    else:
        iou_thresholds = params.iouThrs
        iou_idx = None
        for i, thr in enumerate(iou_thresholds):
            if abs(thr - iou_threshold) < 1e-5:
                iou_idx = i
                break
        if iou_idx is None:
            return pd.DataFrame()

    # Find area range index
    area_labels = ["all", "small", "medium", "large"]
    try:
        area_idx = area_labels.index(area_range)
    except ValueError:
        return pd.DataFrame()

    # Find max detections index
    max_det_idx = None
    for i, md in enumerate(params.maxDets):
        if md == max_dets:
            max_det_idx = i
            break
    if max_det_idx is None:
        return pd.DataFrame()

    cat_ids = params.catIds
    precision = coco_eval.eval["precision"]
    recall = coco_eval.eval["recall"]

    data = []
    for cat_idx, cat_id in enumerate(cat_ids):
        if metric_name == "AP":
            if iou_threshold is None:
                prec = precision[:, :, cat_idx, area_idx, max_det_idx]
                prec = prec[prec > -1]
                metric_val = np.mean(prec) if prec.size > 0 else -1
            else:
                prec = precision[iou_idx, :, cat_idx, area_idx, max_det_idx]
                prec = prec[prec > -1]
                metric_val = np.mean(prec) if prec.size > 0 else -1
        else:  # AR
            if iou_threshold is None:
                rec = recall[:, cat_idx, area_idx, max_det_idx]
                rec = rec[rec > -1]
                metric_val = np.mean(rec) if rec.size > 0 else -1
            else:
                rec = recall[iou_idx, cat_idx, area_idx, max_det_idx]
                metric_val = rec if rec > -1 else -1

        if metric_val >= 0:
            class_name = (
                class_names.get(cat_id, f"class_{cat_id}")
                if class_names
                else f"class_{cat_id}"
            )
            data.append(
                {"class_id": cat_id, "class_name": class_name, metric_name: metric_val}
            )

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(metric_name, ascending=False)
    return df
