#!/usr/bin/env python3
"""
View RF-DETR-MASK predictions in FiftyOne.

This script loads model predictions (bounding boxes and instance masks)
and visualizes them using the FiftyOne app.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import fiftyone as fo
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from rfdetr.util.coco_classes import COCO_CLASSES


def load_model(
    model_path: str,
    model_type: str = "base",
    device: str = "cuda",
    num_classes: int = 91,
) -> torch.nn.Module:
    """Load RF-DETR-MASK model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        model_type: Model variant ("base" or "large")
        device: Device to load model on
        num_classes: Number of classes (including background)

    Returns:
        Loaded model in eval mode
    """
    # Load checkpoint first to get args
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Import model building function
    from rfdetr.models.lwdetr import build_model

    # Get args from checkpoint
    if "args" in checkpoint:
        args = checkpoint["args"]
        # Override device and eval mode
        args.device = device
        args.eval = True
        args.num_classes = num_classes - 1  # build_model adds 1

        # Build model with checkpoint args
        model = build_model(args)

        # Load state dict
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        return model
    else:
        raise ValueError("No args found in checkpoint")


def process_image(
    model: torch.nn.Module,
    image_path: str,
    device: str = "cuda",
    score_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """Process a single image and get predictions.

    Args:
        model: RF-DETR-MASK model
        image_path: Path to input image
        device: Device to run inference on
        score_threshold: Minimum score for predictions

    Returns:
        Tuple of (boxes, scores, labels, masks)
    """
    import torchvision.transforms.functional as F
    from rfdetr.util.box_ops import box_cxcywh_to_xyxy

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Convert to tensor and normalize
    image_tensor = F.to_tensor(image)
    image_tensor = F.normalize(
        image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )

    # Resize to model resolution (typically 560x560)
    image_tensor = F.resize(image_tensor, (560, 560))

    # Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)

    # Process outputs
    if "pred_logits" in outputs:
        # Direct model output format
        logits = outputs["pred_logits"][0]  # Remove batch dimension
        boxes = outputs["pred_boxes"][0]

        # Get probabilities
        prob = logits.sigmoid()
        scores, labels = prob.max(-1)

        # Filter by threshold
        keep = scores > score_threshold
        scores = scores[keep].cpu().numpy()
        labels = labels[keep].cpu().numpy()
        boxes = boxes[keep]

        # Convert boxes from cxcywh to xyxy
        boxes = box_cxcywh_to_xyxy(boxes)
        boxes = boxes * torch.tensor(
            [width, height, width, height], device=boxes.device
        )
        boxes = boxes.cpu().numpy()

        # Process masks if available
        masks = None
        if (
            "pred_masks" in outputs
            and outputs["pred_masks"] is not None
            and len(boxes) > 0
        ):
            pred_masks = outputs["pred_masks"][0][keep]

            if pred_masks.shape[0] > 0:  # Check if there are any masks
                # Resize masks to original image size
                pred_masks = torch.nn.functional.interpolate(
                    pred_masks.unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0]

                # Convert to binary masks
                masks = (pred_masks > 0.5).cpu().numpy()
    else:
        # Fallback for different output format
        boxes = np.array([])
        scores = np.array([])
        labels = np.array([])
        masks = None

    return boxes, scores, labels, masks


def create_fiftyone_dataset(
    image_dir: str,
    model: torch.nn.Module,
    dataset_name: str = "rf_detr_mask_predictions",
    max_samples: Optional[int] = None,
    score_threshold: float = 0.5,
    device: str = "cuda",
) -> fo.Dataset:
    """Create FiftyOne dataset with model predictions.

    Args:
        image_dir: Directory containing images
        model: RF-DETR-MASK model
        dataset_name: Name for FiftyOne dataset
        max_samples: Maximum number of samples to process
        score_threshold: Minimum score for predictions
        device: Device to run inference on

    Returns:
        FiftyOne dataset with predictions
    """
    # Create dataset
    dataset = fo.Dataset(name=dataset_name, overwrite=True)

    # Get image paths
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(Path(image_dir).glob(ext))

    if max_samples:
        image_paths = image_paths[:max_samples]

    # Process images
    samples = []
    for image_path in tqdm(image_paths, desc="Processing images"):
        # Get predictions
        boxes, scores, labels, masks = process_image(
            model, str(image_path), device, score_threshold
        )

        # Create sample
        sample = fo.Sample(filepath=str(image_path))

        # Add detections
        detections = []
        for i in range(len(boxes)):
            # Convert box format from [x1, y1, x2, y2] to [x, y, w, h] normalized
            img = Image.open(image_path)
            width, height = img.size

            x1, y1, x2, y2 = boxes[i]
            x = x1 / width
            y = y1 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            # Create detection
            class_id = int(labels[i])
            # COCO_CLASSES is a dict with 1-indexed keys
            if class_id in COCO_CLASSES:
                label = COCO_CLASSES[class_id]
            else:
                label = f"class_{class_id}"

            detection = fo.Detection(
                label=label, bounding_box=[x, y, w, h], confidence=float(scores[i])
            )

            # Add mask if available
            if len(masks) > 0:
                # Convert boolean mask to instance mask
                mask_array = masks[i].astype(np.uint8) * 255
                detection.mask = mask_array

            detections.append(detection)

        sample["predictions"] = fo.Detections(detections=detections)
        samples.append(sample)

    # Add samples to dataset
    dataset.add_samples(samples)

    return dataset


def create_fiftyone_dataset_from_coco(
    annotations_path: str,
    image_dir: str,
    predictions_path: Optional[str] = None,
    dataset_name: str = "rf_detr_mask_coco",
    max_samples: Optional[int] = None,
) -> fo.Dataset:
    """Create FiftyOne dataset from COCO annotations and optionally predictions.

    Args:
        annotations_path: Path to COCO annotations JSON
        image_dir: Directory containing images
        predictions_path: Optional path to COCO predictions JSON
        dataset_name: Name for FiftyOne dataset
        max_samples: Maximum number of samples to load

    Returns:
        FiftyOne dataset
    """
    # Load COCO dataset
    dataset = fo.Dataset(name=dataset_name, overwrite=True)

    # Add ground truth annotations
    dataset.add_coco_labels(
        labels_path=annotations_path,
        label_field="ground_truth",
        label_types=["detections", "segmentations"],
    )

    # Update image paths if needed
    for sample in dataset:
        filename = os.path.basename(sample.filepath)
        sample.filepath = os.path.join(image_dir, filename)
        sample.save()

    # Add predictions if provided
    if predictions_path:
        dataset.add_coco_labels(
            labels_path=predictions_path,
            label_field="predictions",
            label_types=["detections", "segmentations"],
        )

    # Limit samples if requested
    if max_samples:
        view = dataset.limit(max_samples)
        dataset = view.clone(dataset_name)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="View RF-DETR-MASK predictions in FiftyOne"
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "coco"],
        default="inference",
        help="Mode: 'inference' to run model on images, 'coco' to load COCO annotations/predictions",
    )

    # Common arguments
    parser.add_argument(
        "--image-dir", required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--dataset-name", default="rf_detr_mask", help="FiftyOne dataset name"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to process"
    )

    # Inference mode arguments
    parser.add_argument(
        "--model-path", help="Path to model checkpoint (for inference mode)"
    )
    parser.add_argument(
        "--model-type", choices=["base", "large"], default="base", help="Model variant"
    )
    parser.add_argument("--num-classes", type=int, default=91, help="Number of classes")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Score threshold for predictions",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )

    # COCO mode arguments
    parser.add_argument(
        "--annotations", help="Path to COCO annotations JSON (for coco mode)"
    )
    parser.add_argument(
        "--predictions", help="Path to COCO predictions JSON (optional, for coco mode)"
    )

    # FiftyOne arguments
    parser.add_argument("--port", type=int, default=5151, help="Port for FiftyOne app")
    parser.add_argument(
        "--remote", action="store_true", help="Run FiftyOne in remote mode"
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Don't launch FiftyOne app automatically",
    )

    args = parser.parse_args()

    # Create dataset based on mode
    if args.mode == "inference":
        if not args.model_path:
            parser.error("--model-path is required for inference mode")

        print(f"Loading model from {args.model_path}...")
        model = load_model(
            args.model_path, args.model_type, args.device, args.num_classes
        )

        print(f"Creating dataset from {args.image_dir}...")
        dataset = create_fiftyone_dataset(
            args.image_dir,
            model,
            args.dataset_name,
            args.max_samples,
            args.score_threshold,
            args.device,
        )

    else:  # coco mode
        if not args.annotations:
            parser.error("--annotations is required for coco mode")

        print(f"Loading COCO dataset from {args.annotations}...")
        dataset = create_fiftyone_dataset_from_coco(
            args.annotations,
            args.image_dir,
            args.predictions,
            args.dataset_name,
            args.max_samples,
        )

    # Print dataset info
    print(f"\nDataset '{args.dataset_name}' created with {len(dataset)} samples")

    # Launch FiftyOne app
    if not args.no_launch:
        session = fo.launch_app(dataset, port=args.port, remote=args.remote)

        print(f"\nFiftyOne app running at http://localhost:{args.port}")
        print("Press Ctrl+C to exit")

        # Keep the session alive
        try:
            session.wait()
        except KeyboardInterrupt:
            print("\nShutting down...")
    else:
        print(f"\nDataset '{args.dataset_name}' created. Launch FiftyOne to view it.")


if __name__ == "__main__":
    main()
