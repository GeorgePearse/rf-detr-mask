#!/usr/bin/env python
"""Minimal test to debug RF-DETR-Mask training"""

import sys
from typing import ClassVar

import torch

sys.path.insert(0, ".")

from torch.utils.data import DataLoader

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset
from rfdetr.models import build_criterion_and_postprocessors, build_model


class Config:
    device = "cuda"
    num_classes = 69
    masks = True
    encoder = "dinov2_base"
    vit_encoder_num_layers = 12
    pretrained_encoder = True
    window_block_indexes: ClassVar[list] = []
    drop_path = 0.1
    out_feature_indexes: ClassVar[list[int]] = [3, 7, 11]
    projector_scale: ClassVar[list[str]] = ["P3", "P4", "P5"]
    use_cls_token = True
    position_embedding = "sine"
    freeze_encoder = False
    layer_norm = True
    rms_norm = False
    backbone_lora = False
    force_no_pretrain = False
    gradient_checkpointing = False
    encoder_only = False
    backbone_only = False
    hidden_dim = 256
    resolution = 644
    num_queries = 900
    aux_loss = True
    group_detr = 1
    two_stage = True
    lite_refpoint_refine = False
    bbox_reparam = True
    sa_nheads = 8
    ca_nheads = 8
    dim_feedforward = 2048
    num_feature_levels = 3
    dec_n_points = 4
    decoder_norm = "LN"
    pretrain_weights = None
    dropout = 0.0
    dec_layers = 6
    shape = (644, 644)
    set_loss = "lw_detr"
    set_cost_class = 5
    set_cost_bbox = 2
    set_cost_giou = 1
    cls_loss_coef = 4.5
    bbox_loss_coef = 2.0
    giou_loss_coef = 1.0
    use_varifocal_loss = False
    mask_loss_coef = 1.0
    dice_loss_coef = 1.0
    use_position_supervised_loss = False
    ia_bce_loss = False
    sum_group_losses = False
    num_select = 300
    focal_alpha = 0.25
    focal_gamma = 2.0
    coco_path = "/home/georgepearse/data/cmr/annotations"
    coco_train = "2025-05-15_12:38:23.077836_train_ordered.json"
    coco_img_path = "/home/georgepearse/data/images"
    dataset = "coco"
    dataset_file = "coco"
    square_resize = True
    square_resize_div_64 = False
    multi_scale = False
    expanded_scales: ClassVar[list[int]] = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]


def main():
    print("Building model...")
    args = Config()
    model = build_model(args).cuda()
    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    print("\nBuilding criterion...")
    criterion, postprocessors = build_criterion_and_postprocessors(args)
    criterion = criterion.cuda()

    print("\nBuilding dataset...")
    dataset = build_dataset(image_set="train", args=args, resolution=args.resolution)
    print(f"Dataset size: {len(dataset)}")

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn, num_workers=0
    )

    print("\nTesting data loading...")
    for i, (samples, targets) in enumerate(dataloader):
        print(f"\nBatch {i}")
        print(f"Samples shape: {samples.tensors.shape}")
        print(f"Number of targets: {len(targets)}")
        print(f"First target keys: {targets[0].keys()}")
        if "masks" in targets[0]:
            print(f"Target masks shape: {targets[0]['masks'].shape}")
        else:
            print("No masks in target")

        if i >= 2:  # Test just a few batches
            break

    print("\nTesting forward pass...")
    model.train()
    samples, targets = next(iter(dataloader))
    samples = samples.to("cuda")
    targets = [{k: v.to("cuda") for k, v in t.items()} for t in targets]

    with torch.cuda.amp.autocast(enabled=False):
        outputs = model(samples, targets)
        print(f"Model output keys: {outputs.keys()}")
        if "pred_masks" in outputs:
            print(f"pred_masks shape: {outputs['pred_masks'].shape}")

        loss_dict = criterion(outputs, targets)
        print(f"Loss dict keys: {loss_dict.keys()}")
        print(f"loss_mask: {loss_dict.get('loss_mask', 'Not found')}")


if __name__ == "__main__":
    main()
