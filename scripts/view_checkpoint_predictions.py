#!/usr/bin/env python3
"""
Direct checkpoint loading for RF-DETR-MASK visualization.
"""

import argparse
from pathlib import Path
from typing import Optional

import fiftyone as fo
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F

from rfdetr.models.lwdetr import build_lwdetr
from rfdetr.models.backbone import build_backbone
from rfdetr.models.position_encoding import build_position_encoding
from rfdetr.models.transformer import build_transformer
from rfdetr.util.box_ops import box_cxcywh_to_xyxy
from rfdetr.util.coco_classes import COCO_CLASSES


class Args:
    """Mock args object for model building."""

    def __init__(self, **kwargs):
        # Default values
        self.num_classes = 91
        self.hidden_dim = 256
        self.position_embedding = "sine"
        self.num_queries = 300
        self.num_select = 300
        self.dropout = 0.0
        self.bbox_reparam = True
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.focal_loss = True
        self.no_aux_loss = False
        self.aux_loss = True
        self.two_stage = False
        self.dec_layers = 3
        self.dim_feedforward = 2048
        self.sa_nheads = 8
        self.ca_nheads = 16
        self.dec_n_points = 2
        self.lite_refpoint_refine = True
        self.encoder = "dinov2_windowed_small"
        self.out_feature_indexes = [2, 5, 8, 11]
        self.projector_scale = ["P4"]
        self.gradient_checkpointing = False
        self.group_detr = 1
        self.num_patterns = 0
        self.k_one2many = 0
        self.lambda_one2many = 0.0
        self.kernel_size = 5
        self.num_feature_levels = 1
        self.decoder_norm = "LN"
        self.device = "cuda"

        # Update with provided values
        for k, v in kwargs.items():
            setattr(self, k, v)


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load model directly from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get args from checkpoint
    if "args" in checkpoint:
        checkpoint_args = checkpoint["args"]
        # Create args object with checkpoint values
        args = Args(
            num_classes=getattr(checkpoint_args, "num_classes", 91),
            hidden_dim=getattr(checkpoint_args, "hidden_dim", 256),
            num_queries=getattr(checkpoint_args, "num_queries", 300),
            num_select=getattr(checkpoint_args, "num_select", 300),
            dropout=getattr(checkpoint_args, "dropout", 0.0),
            dim_feedforward=getattr(checkpoint_args, "dim_feedforward", 2048),
            sa_nheads=getattr(checkpoint_args, "sa_nheads", 8),
            ca_nheads=getattr(checkpoint_args, "ca_nheads", 16),
            dec_n_points=getattr(checkpoint_args, "dec_n_points", 2),
            dec_layers=getattr(checkpoint_args, "dec_layers", 3),
            encoder=getattr(checkpoint_args, "encoder", "dinov2_windowed_small"),
            out_feature_indexes=getattr(
                checkpoint_args, "out_feature_indexes", [2, 5, 8, 11]
            ),
            projector_scale=getattr(checkpoint_args, "projector_scale", ["P4"]),
            focal_loss=getattr(checkpoint_args, "focal_loss", True),
            focal_alpha=getattr(checkpoint_args, "focal_alpha", 0.25),
            focal_gamma=getattr(checkpoint_args, "focal_gamma", 2.0),
            bbox_reparam=getattr(checkpoint_args, "bbox_reparam", True),
            lite_refpoint_refine=getattr(checkpoint_args, "lite_refpoint_refine", True),
            two_stage=getattr(checkpoint_args, "two_stage", False),
            gradient_checkpointing=getattr(
                checkpoint_args, "gradient_checkpointing", False
            ),
            group_detr=getattr(checkpoint_args, "group_detr", 1),
            device=device,
        )
    else:
        args = Args(device=device)

    # Map encoder name if needed
    if args.encoder == "dinov2_small":
        args.encoder = "dinov2_windowed_small"
    elif args.encoder == "dinov2_base":
        args.encoder = "dinov2_windowed_base"

    # Build model
    backbone = build_backbone(args)
    position_encoding = build_position_encoding(args)
    transformer = build_transformer(args)

    model = build_lwdetr(
        backbone=backbone,
        transformer=transformer,
        position_encoding=position_encoding,
        args=args,
    )

    # Load state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()

    return model, args


def predict_image(model, image_path: str, args, threshold: float = 0.5):
    """Run prediction on a single image."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Preprocess
    img_tensor = F.to_tensor(image)
    img_tensor = F.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_tensor = F.resize(img_tensor, (560, 560))  # Default resolution
    img_tensor = img_tensor.unsqueeze(0).to(args.device)

    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)

    # Process outputs
    out_logits = outputs["pred_logits"][0]  # Remove batch dimension
    out_bbox = outputs["pred_boxes"][0]

    # Get predictions
    prob = out_logits.sigmoid()
    scores, labels = prob.max(-1)

    # Filter by threshold
    keep = scores > threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = out_bbox[keep]

    # Convert to xyxy format
    boxes = box_cxcywh_to_xyxy(boxes)
    # Scale to image size
    boxes = boxes * torch.tensor(
        [width, height, width, height], dtype=torch.float32, device=boxes.device
    )

    # Get masks if available
    masks = None
    if "pred_masks" in outputs and outputs["pred_masks"] is not None:
        pred_masks = outputs["pred_masks"][0][keep]  # Remove batch dimension and filter
        # Resize masks to original image size
        masks = F.interpolate(
            pred_masks.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]
        masks = (masks > 0.5).cpu().numpy()

    return boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy(), masks


def create_fiftyone_dataset(
    image_dir: str,
    model,
    args,
    dataset_name: str = "rf_detr_mask_checkpoint",
    max_samples: Optional[int] = None,
    score_threshold: float = 0.5,
    class_names=None,
) -> fo.Dataset:
    """Create FiftyOne dataset with predictions."""
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
            boxes, scores, labels, masks = predict_image(
                model, str(image_path), args, score_threshold
            )

            # Create sample
            sample = fo.Sample(filepath=str(image_path))

            # Load image dimensions
            img = Image.open(image_path)
            width, height = img.size

            # Convert to FiftyOne detections
            fo_detections = []

            for i in range(len(boxes)):
                # Convert xyxy to relative xywh
                x1, y1, x2, y2 = boxes[i]
                x = x1 / width
                y = y1 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height

                # Get class name
                class_id = int(labels[i])
                if class_names and class_id < len(class_names):
                    label = class_names[class_id]
                elif class_id < len(COCO_CLASSES):
                    label = COCO_CLASSES[class_id]
                else:
                    label = f"class_{class_id}"

                # Create detection
                detection = fo.Detection(
                    label=label, bounding_box=[x, y, w, h], confidence=float(scores[i])
                )

                # Add mask if available
                if masks is not None and i < len(masks):
                    mask = masks[i]
                    detection.mask = (mask * 255).astype("uint8")

                fo_detections.append(detection)

            sample["predictions"] = fo.Detections(detections=fo_detections)
            samples.append(sample)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Add samples
    if samples:
        dataset.add_samples(samples)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="View RF-DETR-MASK checkpoint predictions"
    )

    # Required
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--image-dir", required=True, help="Directory with images")

    # Optional
    parser.add_argument(
        "--dataset-name", default="rf_detr_mask_checkpoint", help="Dataset name"
    )
    parser.add_argument("--max-samples", type=int, help="Max samples to process")
    parser.add_argument(
        "--score-threshold", type=float, default=0.5, help="Score threshold"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--class-names", nargs="+", help="Class names")

    # FiftyOne
    parser.add_argument("--port", type=int, default=5151, help="FiftyOne port")
    parser.add_argument("--remote", action="store_true", help="Remote mode")
    parser.add_argument("--no-launch", action="store_true", help="Don't launch app")

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, model_args = load_model_from_checkpoint(args.checkpoint, args.device)
    print(f"Model loaded with {model_args.num_classes} classes")

    print(f"\nProcessing images from {args.image_dir}...")
    dataset = create_fiftyone_dataset(
        args.image_dir,
        model,
        model_args,
        args.dataset_name,
        args.max_samples,
        args.score_threshold,
        args.class_names,
    )

    print(f"\nDataset '{args.dataset_name}' created with {len(dataset)} samples")

    # Launch app
    if not args.no_launch:
        session = fo.launch_app(dataset, port=args.port, remote=args.remote)

        print(f"\nFiftyOne app running at http://localhost:{args.port}")
        print("Press Ctrl+C to exit")

        try:
            session.wait()
        except KeyboardInterrupt:
            print("\nShutting down...")
    else:
        print("\nDataset created. Use FiftyOne to view it.")


if __name__ == "__main__":
    main()
