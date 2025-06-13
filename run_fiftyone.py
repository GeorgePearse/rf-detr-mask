#!/usr/bin/env python3
"""
Quick FiftyOne visualization for RF-DETR-MASK checkpoint.
"""

import torch
import fiftyone as fo
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F


# Simple args class
class Args:
    def __init__(self):
        self.num_classes = 69
        self.hidden_dim = 128
        self.position_embedding = "sine"
        self.num_queries = 100
        self.num_select = 300
        self.dropout = 0.0
        self.bbox_reparam = True
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.focal_loss = True
        self.aux_loss = True
        self.two_stage = False
        self.dec_layers = 3
        self.dim_feedforward = 512
        self.sa_nheads = 4
        self.ca_nheads = 4
        self.dec_n_points = 4
        self.lite_refpoint_refine = True
        self.encoder = "dinov2_windowed_small"
        self.out_feature_indexes = [2, 5, 8]
        self.projector_scale = ["P3", "P4", "P5"]
        self.gradient_checkpointing = False
        self.group_detr = 1
        self.num_patterns = 0
        self.k_one2many = 0
        self.lambda_one2many = 0.0
        self.num_feature_levels = 1
        self.decoder_norm = "LN"
        self.device = "cuda"
        self.resume = ""
        self.eval = True
        self.no_aux_loss = False
        self.vit_encoder_num_layers = 6
        self.finetune_early_layers = 6
        self.return_masks = True
        self.mask_head_type = "smallconv"
        self.layer_norm = True
        self.multi_scale = False
        self.frozen_weights = None
        self.backbone_feature_layers = ["res2", "res3", "res4", "res5"]
        self.encoder_sa_nheads = 8


def main():
    # Paths
    checkpoint_path = "checkpoints/model.pth"
    image_dir = "/home/georgepearse/data/images"
    max_samples = 10
    score_threshold = 0.3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Build model
    from rfdetr.models.lwdetr import build_model

    args = Args()
    model = build_model(args)

    # Load weights
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    print("Model loaded!")

    # Create dataset
    dataset = fo.Dataset(name="rf_detr_mask_cmr", overwrite=True)

    # Get images
    image_paths = list(Path(image_dir).glob("*.jpg"))[:max_samples]

    # Process images
    samples = []
    for img_path in tqdm(image_paths, desc="Processing"):
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            width, height = image.size

            # Preprocess
            img_tensor = F.to_tensor(image)
            img_tensor = F.normalize(
                img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
            img_tensor = F.resize(img_tensor, (560, 560))
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(img_tensor)

            # Process outputs
            logits = outputs["pred_logits"][0]
            boxes = outputs["pred_boxes"][0]

            prob = logits.sigmoid()
            scores, labels = prob.max(-1)

            keep = scores > score_threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            # Convert boxes
            from rfdetr.util.box_ops import box_cxcywh_to_xyxy

            boxes = box_cxcywh_to_xyxy(boxes)
            boxes = boxes * torch.tensor(
                [width, height, width, height], device=boxes.device
            )

            # Create FiftyOne sample
            sample = fo.Sample(filepath=str(img_path))

            detections = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].cpu().numpy()

                detection = fo.Detection(
                    label=f"class_{int(labels[i])}",
                    bounding_box=[
                        x1 / width,
                        y1 / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    ],
                    confidence=float(scores[i]),
                )
                detections.append(detection)

            sample["predictions"] = fo.Detections(detections=detections)
            samples.append(sample)

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Add samples
    dataset.add_samples(samples)
    print(f"\nCreated dataset with {len(dataset)} samples")

    # Launch app
    session = fo.launch_app(dataset, port=5151)
    print("\nFiftyOne app running at http://localhost:5151")
    print("Press Ctrl+C to exit")

    try:
        session.wait()
    except KeyboardInterrupt:
        print("\nDone!")


if __name__ == "__main__":
    main()
