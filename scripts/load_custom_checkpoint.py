#!/usr/bin/env python3
"""
Load and visualize a custom RF-DETR-MASK checkpoint with FiftyOne.
"""

import argparse
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import fiftyone as fo
import torch
from PIL import Image
from tqdm import tqdm

from rfdetr.config import RFDETRBaseConfig
from rfdetr.model_utils import Model
from rfdetr.util.coco_classes import COCO_CLASSES


def load_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """Extract configuration from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})

    # Extract relevant parameters
    config_params = {}
    if hasattr(args, "__dict__"):
        args_dict = vars(args)
        # Map old encoder names to new ones
        encoder = args_dict.get("encoder", "dinov2_windowed_small")
        if encoder == "dinov2_small":
            encoder = "dinov2_windowed_small"
        elif encoder == "dinov2_base":
            encoder = "dinov2_windowed_base"

        config_params = {
            "encoder": encoder,
            "hidden_dim": args_dict.get("hidden_dim", 256),
            "sa_nheads": args_dict.get("sa_nheads", 8),
            "ca_nheads": args_dict.get("ca_nheads", 16),
            "dec_n_points": args_dict.get("dec_n_points", 2),
            "num_classes": args_dict.get("num_classes", 91),
            "num_queries": args_dict.get("num_queries", 300),
            "num_select": args_dict.get("num_select", 300),
            "projector_scale": args_dict.get("projector_scale", ["P4"]),
            "out_feature_indexes": args_dict.get("out_feature_indexes", [2, 5, 8, 11]),
            "dim_feedforward": args_dict.get("dim_feedforward", 2048),
        }

    return config_params, checkpoint


def load_custom_model(
    checkpoint_path: str,
    device: str = "cuda",
    override_num_classes: Optional[int] = None,
):
    """Load model with custom configuration from checkpoint."""
    # Get checkpoint info
    config_params, checkpoint = load_checkpoint_info(checkpoint_path)

    # Create config with checkpoint parameters
    config = RFDETRBaseConfig(
        encoder=config_params["encoder"],
        hidden_dim=config_params["hidden_dim"],
        sa_nheads=config_params["sa_nheads"],
        ca_nheads=config_params["ca_nheads"],
        dec_n_points=config_params["dec_n_points"],
        num_classes=override_num_classes or config_params["num_classes"],
        num_queries=config_params["num_queries"],
        num_select=config_params["num_select"],
        projector_scale=config_params["projector_scale"],
        out_feature_indexes=config_params["out_feature_indexes"],
        device=device,
        pretrain_weights=None,  # Don't load pretrained weights
    )

    # Create model
    model = Model(**config.dict())

    # Get state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load state dict
    model.model.load_state_dict(state_dict, strict=False)
    model.model.to(device)
    model.model.eval()

    print("Loaded model with config:")
    print(f"  - encoder: {config.encoder}")
    print(f"  - hidden_dim: {config.hidden_dim}")
    print(f"  - num_classes: {config.num_classes}")

    return model, config


def predict_with_masks(model, image_path: str, threshold: float = 0.5):
    """Run prediction on a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Convert to tensor
    import torchvision.transforms.functional as F

    img_tensor = F.to_tensor(image)

    # Normalize
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    img_tensor = F.normalize(img_tensor, means, stds)

    # Resize to model resolution
    img_tensor = F.resize(img_tensor, (model.resolution, model.resolution))

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(model.device)

    # Run inference
    with torch.no_grad():
        outputs = model.model(img_tensor)

    # Get original size for postprocessing
    target_sizes = torch.tensor([[height, width]], device=model.device)

    # Postprocess
    results = model.postprocessors["bbox"](outputs, target_sizes=target_sizes)

    # Extract predictions
    result = results[0]
    scores = result["scores"]
    labels = result["labels"]
    boxes = result["boxes"]
    masks = result.get("masks", None)

    # Filter by threshold
    keep = scores > threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]
    if masks is not None:
        masks = masks[keep]

    return boxes, scores, labels, masks


def create_fiftyone_dataset(
    image_dir: str,
    model,
    dataset_name: str = "rf_detr_mask_custom",
    max_samples: Optional[int] = None,
    score_threshold: float = 0.5,
    class_names=None,
) -> fo.Dataset:
    """Create FiftyOne dataset with model predictions."""
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
            # Get predictions
            boxes, scores, labels, masks = predict_with_masks(
                model, str(image_path), score_threshold
            )

            # Create sample
            sample = fo.Sample(filepath=str(image_path))

            # Load image to get dimensions
            img = Image.open(image_path)
            width, height = img.size

            # Convert to FiftyOne detections
            fo_detections = []

            for i in range(len(boxes)):
                # Convert xyxy to relative xywh
                x1, y1, x2, y2 = boxes[i].cpu().numpy()
                x = x1 / width
                y = y1 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height

                # Get class name
                class_id = int(labels[i].cpu().numpy())
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
                    confidence=float(scores[i].cpu().numpy()),
                )

                # Add mask if available
                if masks is not None:
                    mask = masks[i].cpu().numpy()
                    # Ensure mask is binary
                    mask = (mask > 0.5).astype("uint8") * 255
                    detection.mask = mask

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
    parser = argparse.ArgumentParser(description="Load custom RF-DETR-MASK checkpoint")

    # Required arguments
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument(
        "--image-dir", required=True, help="Directory containing images"
    )

    # Optional arguments
    parser.add_argument(
        "--dataset-name", default="rf_detr_mask_custom", help="FiftyOne dataset name"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.5, help="Score threshold"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--override-num-classes", type=int, help="Override number of classes"
    )
    parser.add_argument("--class-names", nargs="+", help="List of class names")

    # FiftyOne arguments
    parser.add_argument("--port", type=int, default=5151, help="Port for FiftyOne app")
    parser.add_argument(
        "--remote", action="store_true", help="Run FiftyOne in remote mode"
    )
    parser.add_argument(
        "--no-launch", action="store_true", help="Don't launch FiftyOne app"
    )

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, config = load_custom_model(
        args.checkpoint, args.device, args.override_num_classes
    )

    print(f"\nCreating dataset from {args.image_dir}...")
    dataset = create_fiftyone_dataset(
        args.image_dir,
        model,
        args.dataset_name,
        args.max_samples,
        args.score_threshold,
        args.class_names,
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
