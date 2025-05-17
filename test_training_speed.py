#!/usr/bin/env python
"""Test training speed to find bottlenecks"""

import sys
import time
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
    grad_accum_steps = 1
    batch_size = 1
    lr = 1e-4
    lr_encoder = 1e-5
    lr_projector = 1e-5
    lr_vit_layer_decay = 1.0
    lr_component_decay = 0.9
    weight_decay = 1e-4
    clip_max_norm = 0.1
    amp = False
    distributed = False


def main():
    print("Building model...")
    args = Config()
    model = build_model(args).cuda()
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    print("\nBuilding criterion...")
    criterion, postprocessors = build_criterion_and_postprocessors(args)
    criterion = criterion.cuda()

    print("\nBuilding dataset...")
    dataset = build_dataset(image_set="train", args=args, resolution=args.resolution)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn, num_workers=0
    )

    print("\nBuilding optimizer...")
    from rfdetr.util.get_param_dicts import get_param_dict

    param_dicts = get_param_dict(args, model)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    print("\nBuilding LR scheduler...")
    torch.optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.1)

    print("\nTesting individual components...")
    model.train()

    # Test a single forward pass
    samples, targets = next(iter(dataloader))
    samples = samples.to("cuda")
    targets = [{k: v.to("cuda") for k, v in t.items()} for t in targets]

    # Time forward pass
    start = time.time()
    with torch.cuda.amp.autocast(enabled=False):
        outputs = model(samples, targets)
    torch.cuda.synchronize()
    print(f"Forward pass time: {time.time() - start:.3f}s")

    # Time loss computation
    start = time.time()
    loss_dict = criterion(outputs, targets)
    torch.cuda.synchronize()
    print(f"Loss computation time: {time.time() - start:.3f}s")

    # Time backward pass
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    start = time.time()
    losses.backward()
    torch.cuda.synchronize()
    print(f"Backward pass time: {time.time() - start:.3f}s")

    # Time optimizer step
    start = time.time()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    print(f"Optimizer step time: {time.time() - start:.3f}s")

    print("\nTesting full training loop for 10 iterations...")
    total_start = time.time()
    for i in range(10):
        iter_start = time.time()
        samples, targets = next(iter(dataloader))
        samples = samples.to("cuda")
        targets = [{k: v.to("cuda") for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        losses.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Iteration {i}: {time.time() - iter_start:.3f}s")

    print(f"\nTotal time for 10 iterations: {time.time() - total_start:.3f}s")


if __name__ == "__main__":
    main()
