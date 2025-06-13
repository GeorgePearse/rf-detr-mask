#!/usr/bin/env python3
"""
View RF-DETR-MASK predictions in FiftyOne - Simplified version using model's predict method.
"""

import argparse
from pathlib import Path
from typing import Optional

import fiftyone as fo
import torch
from PIL import Image
from tqdm import tqdm

from rfdetr.detr import RFDETRBase, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES


def load_model(
    model_path: str,
    model_type: str = "base",
    device: str = "cuda",
    num_classes: int = 91,
):
    """Load RF-DETR-MASK model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        model_type: Model variant ("base" or "large")
        device: Device to load model on
        num_classes: Number of classes (including background)

    Returns:
        Loaded model
    """
    # Initialize model
    if model_type == "base":
        model = RFDETRBase(num_classes=num_classes, device=device)
    elif model_type == "large":
        model = RFDETRLarge(num_classes=num_classes, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint with weights_only=False for compatibility
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load state dict
    model.model.model.load_state_dict(state_dict, strict=False)

    # Reinitialize detection head if num_classes mismatch
    if num_classes != model.model_config.num_classes:
        print(f"Reinitializing detection head with {num_classes} classes")
        model.model.reinitialize_detection_head(num_classes)
        # Load state dict again but only matching keys
        model.model.model.load_state_dict(state_dict, strict=False)

    model.model.model.to(device)
    model.model.model.eval()

    return model


def create_fiftyone_dataset(
    image_dir: str,
    model,
    dataset_name: str = "rf_detr_mask_predictions",
    max_samples: Optional[int] = None,
    score_threshold: float = 0.5,
    return_masks: bool = True,
    class_names=None,
) -> fo.Dataset:
    """Create FiftyOne dataset with model predictions.

    Args:
        image_dir: Directory containing images
        model: RF-DETR-MASK model
        dataset_name: Name for FiftyOne dataset
        max_samples: Maximum number of samples to process
        score_threshold: Minimum score for predictions
        return_masks: Whether to return masks
        class_names: List of class names

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
        try:
            # Get predictions using model's predict method
            detections = model.predict(
                str(image_path), threshold=score_threshold, return_masks=return_masks
            )

            # Create sample
            sample = fo.Sample(filepath=str(image_path))

            # Convert supervision detections to FiftyOne format
            fo_detections = []

            if detections.xyxy is not None and len(detections.xyxy) > 0:
                # Load image to get dimensions
                img = Image.open(image_path)
                width, height = img.size

                for i in range(len(detections.xyxy)):
                    # Convert xyxy to relative xywh
                    x1, y1, x2, y2 = detections.xyxy[i]
                    x = x1 / width
                    y = y1 / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height

                    # Get class name
                    class_id = int(detections.class_id[i])
                    if class_names and class_id < len(class_names):
                        label = class_names[class_id]
                    elif class_id < len(COCO_CLASSES):
                        label = COCO_CLASSES[class_id]
                    else:
                        label = f"class_{class_id}"

                    # Create detection
                    detection = fo.Detection(
                        label=label,
                        bounding_box=[x, y, w, h],
                        confidence=float(detections.confidence[i]),
                    )

                    # Add mask if available
                    if return_masks and detections.mask is not None:
                        # Get the mask for this detection
                        mask = detections.mask[i]
                        detection.mask = mask.astype("uint8") * 255

                    fo_detections.append(detection)

            sample["predictions"] = fo.Detections(detections=fo_detections)
            samples.append(sample)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    # Add samples to dataset
    if samples:
        dataset.add_samples(samples)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="View RF-DETR-MASK predictions in FiftyOne"
    )

    # Common arguments
    parser.add_argument(
        "--image-dir", required=True, help="Directory containing images"
    )
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--dataset-name", default="rf_detr_mask", help="FiftyOne dataset name"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to process"
    )

    # Model arguments
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
    parser.add_argument("--no-masks", action="store_true", help="Don't return masks")

    # Class names
    parser.add_argument("--class-names", nargs="+", help="List of class names")

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

    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.model_type, args.device, args.num_classes)

    print(f"Creating dataset from {args.image_dir}...")
    dataset = create_fiftyone_dataset(
        args.image_dir,
        model,
        args.dataset_name,
        args.max_samples,
        args.score_threshold,
        return_masks=not args.no_masks,
        class_names=args.class_names,
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
