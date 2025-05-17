#!/usr/bin/env python
"""Test RF-DETR-Mask training with a small subset of data"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset
from rfdetr.engine import train_one_epoch
from rfdetr.models import build_criterion_and_postprocessors, build_model
from rfdetr.util.get_param_dicts import get_param_dict


def get_args():
    parser = argparse.ArgumentParser("Test RF-DETR-Mask training", add_help=True)

    # Dataset parameters
    parser.add_argument("--dataset", default="coco", type=str)
    parser.add_argument("--dataset_file", default="coco", type=str)
    parser.add_argument("--coco_path", type=str, default="/home/georgepearse/data/cmr/annotations")
    parser.add_argument(
        "--coco_train", type=str, default="2025-05-15_12:38:23.077836_train_ordered.json"
    )
    parser.add_argument(
        "--coco_val", type=str, default="2025-05-15_12:38:38.270134_val_ordered.json"
    )
    parser.add_argument("--coco_img_path", type=str, default="/home/georgepearse/data/images")
    parser.add_argument("--output_dir", default="output_test_cmr", help="Path to save logs")

    # Training parameters (reduced for testing)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_encoder", default=1e-5, type=float)
    parser.add_argument("--lr_projector", default=1e-5, type=float)
    parser.add_argument("--lr_vit_layer_decay", default=1.0, type=float)
    parser.add_argument("--lr_component_decay", default=0.9, type=float)
    parser.add_argument("--lr_drop", default=50, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--clip_max_norm", default=0.1, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument(
        "--train_subset", default=50, type=int, help="Number of training samples to use"
    )

    # Model parameters
    parser.add_argument("--encoder", default="dinov2_base", type=str)
    parser.add_argument("--pretrain_weights", type=str, default=None)
    parser.add_argument("--resolution", default=644, type=int)
    parser.add_argument("--set_loss", default="lw_detr", type=str)
    parser.add_argument("--set_cost_class", default=5, type=float)
    parser.add_argument("--set_cost_bbox", default=2, type=float)
    parser.add_argument("--set_cost_giou", default=1, type=float)
    parser.add_argument("--loss_class_coef", default=4.5, type=float)
    parser.add_argument("--loss_bbox_coef", default=2.0, type=float)
    parser.add_argument("--loss_giou_coef", default=1, type=float)
    parser.add_argument("--num_classes", default=69, type=int)
    parser.add_argument("--masks", action="store_true", default=True)
    parser.add_argument("--loss_mask_coef", default=1.0, type=float)
    parser.add_argument("--loss_dice_coef", default=1.0, type=float)

    # Other parameters
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--bbox_reparam", default=True, type=bool)
    parser.add_argument("--group_detr", default=1, type=int)
    parser.add_argument("--two_stage", default=True, type=bool)
    parser.add_argument("--no_intermittent_layers", default=False, type=bool)
    parser.add_argument("--use_fp16", default=True, type=bool)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--square_resize", action="store_true", default=True)
    parser.add_argument("--distributed", action="store_true", default=False)
    return parser.parse_args()


def set_args_defaults(args):
    """Set default values for all required model parameters"""
    args.focal_loss = True
    args.focal_alpha = 0.25
    args.focal_gamma = 2.0
    args.num_queries = 900
    args.hidden_dim = 256
    args.position_embedding_scale = None
    args.backbone_feature_layers = ["res2", "res3", "res4", "res5"]
    args.vit_encoder_num_layers = 12
    args.num_decoder_layers = 6
    args.num_decoder_points = 4
    args.dec_layers = 6

    args.pretrained_encoder = True
    args.window_block_indexes = []
    args.drop_path = 0.1
    args.out_feature_indexes = [3, 7, 11]
    args.projector_scale = ["P3", "P4", "P5"]
    args.use_cls_token = True
    args.position_embedding = "sine"
    args.freeze_encoder = False
    args.layer_norm = True
    args.rms_norm = False
    args.backbone_lora = False
    args.force_no_pretrain = False
    args.gradient_checkpointing = False
    args.encoder_only = False
    args.backbone_only = False

    args.sa_nheads = 8
    args.ca_nheads = 8
    args.dim_feedforward = 2048
    args.num_feature_levels = 3
    args.dec_n_points = 4
    args.lite_refpoint_refine = False
    args.decoder_norm = "LN"

    args.aux_loss = True
    args.cls_loss_coef = args.loss_class_coef
    args.bbox_loss_coef = args.loss_bbox_coef
    args.giou_loss_coef = args.loss_giou_coef

    args.use_varifocal_loss = False
    args.mask_loss_coef = args.loss_mask_coef
    args.dice_loss_coef = args.loss_dice_coef
    args.use_position_supervised_loss = False
    args.ia_bce_loss = False
    args.sum_group_losses = False
    args.num_select = 300

    args.multi_scale = False
    args.expanded_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    args.square_resize_div_64 = False

    args.grad_accum_steps = args.gradient_accumulation_steps
    args.fp16_eval = args.use_fp16


def main():
    args = get_args()
    set_args_defaults(args)

    print(f"Training on {args.train_subset} samples")

    device = torch.device(args.device)

    # Fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build the model
    model = build_model(args)
    criterion, postprocessors = build_criterion_and_postprocessors(args)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_parameters/1e6:.1f}M")

    # Build optimizer
    param_dicts = get_param_dict(args, model)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # Build LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.lr_drop], gamma=0.1)

    # Build dataset
    dataset_train = build_dataset(image_set="train", args=args, resolution=args.resolution)

    # Create subset for quick testing
    indices = list(range(min(args.train_subset, len(dataset_train))))
    dataset_train = Subset(dataset_train, indices)

    print(f"Using {len(dataset_train)} training samples")

    # Build data loaders
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("Starting training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # Train for one epoch
        print(f"\nEpoch {epoch}")

        num_training_steps_per_epoch = len(data_loader_train)
        from collections import defaultdict

        callbacks = defaultdict(list)

        train_stats = train_one_epoch(
            model,
            criterion,
            lr_scheduler,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.batch_size,
            args.clip_max_norm,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            vit_encoder_num_layers=args.vit_encoder_num_layers,
            args=args,
            callbacks=callbacks,
        )

        lr_scheduler.step()

        # Log statistics
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")

    print("\nFinal training stats:")
    for k, v in train_stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
