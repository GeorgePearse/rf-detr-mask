# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Model utilities to avoid circular imports"""

import os
from logging import getLogger
import torch
from rfdetr.util.files import download_file

logger = getLogger(__name__)

HOSTED_MODELS = {
    "rf-detr-base.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    # below is a less converged model that may be better for finetuning but worse for inference
    "rf-detr-base-2.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "rf-detr-large.pth": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
}


def download_pretrain_weights(pretrain_weights: str, redownload=False):
    if pretrain_weights in HOSTED_MODELS:
        if redownload or not os.path.exists(pretrain_weights):
            logger.info(f"Downloading pretrained weights for {pretrain_weights}")
            download_file(
                HOSTED_MODELS[pretrain_weights],
                pretrain_weights,
            )


class Model:
    def __init__(self, **kwargs):
        from rfdetr.models import build_model, build_criterion_and_postprocessors

        args = populate_args(**kwargs)
        self.resolution = args.resolution
        self.model = build_model(args)
        self.device = torch.device(args.device)
        _, self.postprocessors = build_criterion_and_postprocessors(args)
        if args.pretrain_weights is not None:
            print("Loading pretrain weights")
            try:
                checkpoint = torch.load(
                    args.pretrain_weights, map_location="cpu", weights_only=False
                )
            except Exception as e:
                print(f"Failed to load pretrain weights: {e}")
                # re-download weights if they are corrupted
                print("Failed to load pretrain weights, re-downloading")
                download_pretrain_weights(args.pretrain_weights, redownload=True)
                checkpoint = torch.load(
                    args.pretrain_weights, map_location="cpu", weights_only=False
                )

            # Extract class_names from checkpoint if available
            if "args" in checkpoint and hasattr(checkpoint["args"], "class_names"):
                self.class_names = checkpoint["args"].class_names
            else:
                self.class_names = None

            # Handle model weights
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                # Direct state dict
                self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()


def populate_args(**kwargs):
    """Populate args with defaults for Model class"""
    import argparse

    parser = argparse.ArgumentParser()

    # Add necessary arguments with defaults
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--num_classes", type=int, default=91)
    parser.add_argument("--pretrain_weights", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="dinov2_small_with_registers")
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--return_masks", action="store_true", default=False)

    # Create args from kwargs
    args = parser.parse_args([])
    for key, value in kwargs.items():
        setattr(args, key, value)

    return args
