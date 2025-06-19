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
    
    # Model architecture arguments
    parser.add_argument("--encoder", type=str, default="dinov2_small_with_registers")
    parser.add_argument("--vit_encoder_num_layers", type=int, default=12)
    parser.add_argument("--pretrained_encoder", action="store_true", default=True)
    parser.add_argument("--window_block_indexes", nargs="+", type=int, default=[])
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--out_feature_indexes", nargs="+", type=int, default=[2, 5, 8, 11])
    parser.add_argument("--projector_scale", nargs="+", type=str, default=["P4"])
    parser.add_argument("--use_cls_token", action="store_true", default=True)
    parser.add_argument("--position_embedding", type=str, default="sine")
    parser.add_argument("--freeze_encoder", action="store_true", default=False)
    parser.add_argument("--layer_norm", action="store_true", default=False)
    parser.add_argument("--rms_norm", action="store_true", default=False)
    parser.add_argument("--backbone_lora", action="store_true", default=False)
    parser.add_argument("--force_no_pretrain", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    
    # Transformer arguments
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_points", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=6)
    parser.add_argument("--sa_nheads", type=int, default=8)
    parser.add_argument("--ca_nheads", type=int, default=16)
    parser.add_argument("--dec_n_points", type=int, default=4)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--bbox_embed_diff_each_layer", action="store_true", default=False)
    parser.add_argument("--class_embed_diff_each_layer", action="store_true", default=False)
    parser.add_argument("--label_emb_size", type=int, default=96)
    parser.add_argument("--look_forward_twice", action="store_true", default=True)
    parser.add_argument("--enc_ffn", action="store_true", default=False)
    parser.add_argument("--interm_loss", action="store_true", default=True)
    parser.add_argument("--interm_loss_weight", type=float, default=1.0)
    parser.add_argument("--decoder_activation", type=str, default="prelu")
    parser.add_argument("--dec_layer_share", action="store_true", default=False)
    parser.add_argument("--dec_layer_dropout_prob", nargs="+", type=float, default=None)
    parser.add_argument("--drop_cls", type=float, default=0.0)
    parser.add_argument("--drop_scale", type=float, default=0.0)
    parser.add_argument("--backbone_feature_layers", nargs="+", type=str, default=["res2", "res3", "res4", "res5"])
    parser.add_argument("--position_embedding_scale", type=float, default=None)
    
    # Special flags
    parser.add_argument("--encoder_only", action="store_true", default=False)
    parser.add_argument("--backbone_only", action="store_true", default=False)
    
    # Loss arguments (needed for criterion)
    parser.add_argument("--cls_loss_coef", type=float, default=1.0)
    parser.add_argument("--bbox_loss_coef", type=float, default=5.0)
    parser.add_argument("--giou_loss_coef", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--eos_coef", type=float, default=0.1)
    parser.add_argument("--set_cost_class", type=float, default=2.0)
    parser.add_argument("--set_cost_bbox", type=float, default=5.0)
    parser.add_argument("--set_cost_giou", type=float, default=2.0)
    
    # Mask-related arguments
    parser.add_argument("--mask_loss_coef", type=float, default=1.0)
    parser.add_argument("--dice_loss_coef", type=float, default=1.0)
    parser.add_argument("--set_cost_mask", type=float, default=1.0)
    parser.add_argument("--set_cost_dice", type=float, default=1.0)
    parser.add_argument("--mask_out_channels", type=int, default=256)
    parser.add_argument("--num_mask_layers", type=int, default=3)
    parser.add_argument("--mask_size", type=int, default=64)
    parser.add_argument("--fpn_channels", nargs="+", type=int, default=[64, 128, 256, 512])
    
    # Additional required arguments
    parser.add_argument("--decoder_norm", type=str, default="LN")
    parser.add_argument("--num_feature_levels", type=int, default=1)
    parser.add_argument("--aux_loss", action="store_true", default=True)
    parser.add_argument("--two_stage", action="store_true", default=True)
    parser.add_argument("--use_position_supervised_loss", action="store_true", default=False)
    parser.add_argument("--lite_refpoint_refine", action="store_true", default=True)
    parser.add_argument("--bbox_reparam", action="store_true", default=True)
    parser.add_argument("--no_interm_box_loss", action="store_true", default=False)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--num_select", type=int, default=300)
    parser.add_argument("--no_clip_max_norm", action="store_true", default=False)
    parser.add_argument("--ia_bce_loss", action="store_true", default=False)
    parser.add_argument("--use_varifocal_loss", action="store_true", default=False)
    parser.add_argument("--masks", action="store_true", default=False)
    parser.add_argument("--no_intermittent_layers", action="store_true", default=False)
    parser.add_argument("--group_detr", type=int, default=1)
    parser.add_argument("--group_bbox_cost", type=float, default=1.0)
    parser.add_argument("--load_dinov2_weights", action="store_true", default=True)

    # Create args from kwargs
    args = parser.parse_args([])
    for key, value in kwargs.items():
        setattr(args, key, value)

    return args
