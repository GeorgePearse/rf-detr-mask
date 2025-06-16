# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
cleaned main file
"""

import argparse
import ast
import os
from logging import getLogger

import torch


if str(os.environ.get("USE_FILE_SYSTEM_SHARING", "False")).lower() in ["true", "1"]:
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")

logger = getLogger(__name__)

# Import utilities from model_utils to avoid circular dependencies


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--amp", default=False, type=bool)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_encoder", default=1.5e-4, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=12, type=int)
    parser.add_argument("--lr_drop", default=11, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument("--lr_vit_layer_decay", default=0.8, type=float)
    parser.add_argument("--lr_component_decay", default=1.0, type=float)
    parser.add_argument(
        "--do_benchmark", action="store_true", help="benchmark the model"
    )

    # drop args
    # dropout and stochastic depth drop rate; set at most one to non-zero
    parser.add_argument(
        "--dropout", type=float, default=0, help="Drop path rate (default: 0.0)"
    )
    parser.add_argument(
        "--drop_path", type=float, default=0, help="Drop path rate (default: 0.0)"
    )

    # early / late dropout and stochastic depth settings
    parser.add_argument(
        "--drop_mode",
        type=str,
        default="standard",
        choices=["standard", "early", "late"],
        help="drop mode",
    )
    parser.add_argument(
        "--drop_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear"],
        help="drop schedule for early dropout / s.d. only",
    )
    parser.add_argument(
        "--cutoff_epoch",
        type=int,
        default=0,
        help="if drop_mode is early / late, this is the epoch where dropout ends / starts",
    )

    # Model parameters
    parser.add_argument(
        "--pretrained_encoder",
        type=str,
        default=None,
        help="Path to the pretrained encoder.",
    )
    parser.add_argument(
        "--pretrain_weights",
        type=str,
        default=None,
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--pretrain_exclude_keys",
        type=str,
        default=None,
        nargs="+",
        help="Keys you do not want to load.",
    )
    parser.add_argument(
        "--pretrain_keys_modify_to_load",
        type=str,
        default=None,
        nargs="+",
        help="Keys you want to modify to load. Only used when loading objects365 pre-trained weights.",
    )

    # * Backbone
    parser.add_argument(
        "--encoder",
        default="vit_tiny",
        type=str,
        help="Name of the transformer or convolutional encoder to use",
    )
    parser.add_argument(
        "--vit_encoder_num_layers",
        default=12,
        type=int,
        help="Number of layers used in ViT encoder",
    )
    parser.add_argument("--window_block_indexes", default=None, type=int, nargs="+")
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--out_feature_indexes",
        default=[-1],
        type=int,
        nargs="+",
        help="only for vit now",
    )
    parser.add_argument("--freeze_encoder", action="store_true", dest="freeze_encoder")
    parser.add_argument("--layer_norm", action="store_true", dest="layer_norm")
    parser.add_argument("--rms_norm", action="store_true", dest="rms_norm")
    parser.add_argument("--backbone_lora", action="store_true", dest="backbone_lora")
    parser.add_argument(
        "--force_no_pretrain", action="store_true", dest="force_no_pretrain"
    )

    # * Transformer
    parser.add_argument(
        "--dec_layers",
        default=3,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--sa_nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's self-attentions",
    )
    parser.add_argument(
        "--ca_nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's cross-attentions",
    )
    parser.add_argument(
        "--num_queries", default=300, type=int, help="Number of query slots"
    )
    parser.add_argument(
        "--group_detr",
        default=13,
        type=int,
        help="Number of groups to speed up detr training",
    )
    parser.add_argument("--two_stage", action="store_true")
    parser.add_argument(
        "--projector_scale",
        default="P4",
        type=str,
        nargs="+",
        choices=("P3", "P4", "P5", "P6"),
    )
    parser.add_argument(
        "--lite_refpoint_refine",
        action="store_true",
        help="lite refpoint refine mode for speed-up",
    )
    parser.add_argument(
        "--num_select",
        default=100,
        type=int,
        help="the number of predictions selected for evaluation",
    )
    parser.add_argument(
        "--dec_n_points", default=4, type=int, help="the number of sampling points"
    )
    parser.add_argument("--decoder_norm", default="LN", type=str)
    parser.add_argument("--bbox_reparam", action="store_true")
    parser.add_argument("--freeze_batch_norm", action="store_true")
    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=2,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )

    # * Loss coefficients
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    parser.add_argument(
        "--sum_group_losses",
        action="store_true",
        help="To sum losses across groups or mean losses.",
    )
    parser.add_argument("--use_varifocal_loss", action="store_true")
    parser.add_argument("--use_position_supervised_loss", action="store_true")
    parser.add_argument("--ia_bce_loss", action="store_true")

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--square_resize_div_64", action="store_true")

    parser.add_argument(
        "--output_dir", default="output", help="path where to save, empty for no saving"
    )
    parser.add_argument("--dont_save_weights", action="store_true")
    parser.add_argument(
        "--checkpoint_interval",
        default=10,
        type=int,
        help="epoch interval to save checkpoint",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", default=0.9997, type=float)
    parser.add_argument("--ema_tau", default=0, type=float)

    parser.add_argument("--num_workers", default=2, type=int)

    # distributed training parameters
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--sync_bn",
        default=True,
        type=bool,
        help="setup synchronized BatchNorm for distributed training",
    )

    # fp16
    parser.add_argument(
        "--fp16_eval",
        default=False,
        action="store_true",
        help="evaluate in fp16 precision.",
    )

    # custom args
    parser.add_argument(
        "--encoder_only", action="store_true", help="Export and benchmark encoder only"
    )
    parser.add_argument(
        "--backbone_only",
        action="store_true",
        help="Export and benchmark backbone only",
    )
    parser.add_argument("--resolution", type=int, default=640, help="input resolution")
    parser.add_argument("--use_cls_token", action="store_true", help="use cls token")
    parser.add_argument("--multi_scale", action="store_true", help="use multi scale")
    parser.add_argument(
        "--expanded_scales", action="store_true", help="use expanded scales"
    )
    parser.add_argument(
        "--warmup_epochs",
        default=1,
        type=float,
        help="Number of warmup epochs for linear warmup before cosine annealing",
    )
    # Add scheduler type argument: 'step' or 'cosine'
    parser.add_argument(
        "--lr_scheduler",
        default="step",
        choices=["step", "cosine"],
        help="Type of learning rate scheduler to use: 'step' (default) or 'cosine'",
    )
    parser.add_argument(
        "--lr_min_factor",
        default=0.0,
        type=float,
        help="Minimum learning rate factor (as a fraction of initial lr) at the end of cosine annealing",
    )
    # Early stopping parameters
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping based on mAP improvement",
    )
    parser.add_argument(
        "--early_stopping_patience",
        default=10,
        type=int,
        help="Number of epochs with no improvement after which training will be stopped",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        default=0.001,
        type=float,
        help="Minimum change in mAP to qualify as an improvement",
    )
    parser.add_argument(
        "--early_stopping_use_ema",
        action="store_true",
        help="Use EMA model metrics for early stopping",
    )
    # subparsers
    subparsers = parser.add_subparsers(
        title="sub-commands",
        dest="subcommand",
        description="valid subcommands",
        help="additional help",
    )

    # subparser for export model
    parser_export = subparsers.add_parser("export_model", help="LWDETR model export")
    parser_export.add_argument("--infer_dir", type=str, default=None)
    parser_export.add_argument(
        "--verbose", type=ast.literal_eval, default=False, nargs="?", const=True
    )
    parser_export.add_argument("--opset_version", type=int, default=17)
    parser_export.add_argument(
        "--simplify", action="store_true", help="Simplify onnx model"
    )
    parser_export.add_argument(
        "--tensorrt",
        "--trtexec",
        "--trt",
        action="store_true",
        help="build tensorrt engine",
    )
    parser_export.add_argument(
        "--dry-run", "--test", "-t", action="store_true", help="just print command"
    )
    parser_export.add_argument(
        "--profile",
        action="store_true",
        help="Run nsys profiling during TensorRT export",
    )

    # Testing a size change here to see if it works
    parser_export.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=(640, 704),
        help="input shape (width, height)",
    )
    return parser


def populate_args(
    # Basic training parameters
    num_classes=2,
    grad_accum_steps=1,
    amp=False,
    lr=1e-4,
    lr_encoder=1.5e-4,
    batch_size=2,
    weight_decay=1e-4,
    epochs=12,
    lr_drop=11,
    clip_max_norm=0.1,
    lr_vit_layer_decay=0.8,
    lr_component_decay=1.0,
    do_benchmark=False,
    # Drop parameters
    dropout=0,
    drop_path=0,
    drop_mode="standard",
    drop_schedule="constant",
    cutoff_epoch=0,
    # Model parameters
    pretrained_encoder=None,
    pretrain_weights=None,
    pretrain_exclude_keys=None,
    pretrain_keys_modify_to_load=None,
    pretrained_distiller=None,
    # Backbone parameters
    encoder="vit_tiny",
    vit_encoder_num_layers=12,
    window_block_indexes=None,
    position_embedding="sine",
    out_feature_indexes=[-1],
    freeze_encoder=False,
    layer_norm=False,
    rms_norm=False,
    backbone_lora=False,
    force_no_pretrain=False,
    # Transformer parameters
    dec_layers=3,
    dim_feedforward=2048,
    hidden_dim=256,
    sa_nheads=8,
    ca_nheads=8,
    num_queries=300,
    group_detr=13,
    two_stage=False,
    projector_scale="P4",
    lite_refpoint_refine=False,
    num_select=100,
    dec_n_points=4,
    decoder_norm="LN",
    bbox_reparam=False,
    freeze_batch_norm=False,
    # Matcher parameters
    set_cost_class=2,
    set_cost_bbox=5,
    set_cost_giou=2,
    # Loss coefficients
    cls_loss_coef=2,
    bbox_loss_coef=5,
    giou_loss_coef=2,
    focal_alpha=0.25,
    aux_loss=True,
    sum_group_losses=False,
    use_varifocal_loss=False,
    use_position_supervised_loss=False,
    ia_bce_loss=False,
    # Dataset parameters
    dataset_file="coco",
    coco_path=None,
    dataset_dir=None,
    square_resize_div_64=False,
    # Output parameters
    output_dir="output",
    dont_save_weights=False,
    checkpoint_interval=10,
    seed=42,
    resume="",
    start_epoch=0,
    eval=False,
    use_ema=False,
    ema_decay=0.9997,
    ema_tau=0,
    num_workers=2,
    # Distributed training parameters
    device="cuda",
    world_size=1,
    dist_url="env://",
    sync_bn=True,
    # FP16
    fp16_eval=False,
    # Custom args
    encoder_only=False,
    backbone_only=False,
    resolution=640,
    use_cls_token=False,
    multi_scale=False,
    expanded_scales=False,
    warmup_epochs=1,
    lr_scheduler="step",
    lr_min_factor=0.0,
    # Early stopping parameters
    early_stopping=True,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    early_stopping_use_ema=False,
    gradient_checkpointing=False,
    # Additional
    subcommand=None,
    **extra_kwargs,  # To handle any unexpected arguments
):
    args = argparse.Namespace(
        num_classes=num_classes,
        grad_accum_steps=grad_accum_steps,
        amp=amp,
        lr=lr,
        lr_encoder=lr_encoder,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=epochs,
        lr_drop=lr_drop,
        clip_max_norm=clip_max_norm,
        lr_vit_layer_decay=lr_vit_layer_decay,
        lr_component_decay=lr_component_decay,
        do_benchmark=do_benchmark,
        dropout=dropout,
        drop_path=drop_path,
        drop_mode=drop_mode,
        drop_schedule=drop_schedule,
        cutoff_epoch=cutoff_epoch,
        pretrained_encoder=pretrained_encoder,
        pretrain_weights=pretrain_weights,
        pretrain_exclude_keys=pretrain_exclude_keys,
        pretrain_keys_modify_to_load=pretrain_keys_modify_to_load,
        pretrained_distiller=pretrained_distiller,
        encoder=encoder,
        vit_encoder_num_layers=vit_encoder_num_layers,
        window_block_indexes=window_block_indexes,
        position_embedding=position_embedding,
        out_feature_indexes=out_feature_indexes,
        freeze_encoder=freeze_encoder,
        layer_norm=layer_norm,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        force_no_pretrain=force_no_pretrain,
        dec_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        hidden_dim=hidden_dim,
        sa_nheads=sa_nheads,
        ca_nheads=ca_nheads,
        num_queries=num_queries,
        group_detr=group_detr,
        two_stage=two_stage,
        projector_scale=projector_scale,
        lite_refpoint_refine=lite_refpoint_refine,
        num_select=num_select,
        dec_n_points=dec_n_points,
        decoder_norm=decoder_norm,
        bbox_reparam=bbox_reparam,
        freeze_batch_norm=freeze_batch_norm,
        set_cost_class=set_cost_class,
        set_cost_bbox=set_cost_bbox,
        set_cost_giou=set_cost_giou,
        cls_loss_coef=cls_loss_coef,
        bbox_loss_coef=bbox_loss_coef,
        giou_loss_coef=giou_loss_coef,
        focal_alpha=focal_alpha,
        aux_loss=aux_loss,
        sum_group_losses=sum_group_losses,
        use_varifocal_loss=use_varifocal_loss,
        use_position_supervised_loss=use_position_supervised_loss,
        ia_bce_loss=ia_bce_loss,
        dataset_file=dataset_file,
        coco_path=coco_path,
        dataset_dir=dataset_dir,
        square_resize_div_64=square_resize_div_64,
        output_dir=output_dir,
        dont_save_weights=dont_save_weights,
        checkpoint_interval=checkpoint_interval,
        seed=seed,
        resume=resume,
        start_epoch=start_epoch,
        eval=eval,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_tau=ema_tau,
        num_workers=num_workers,
        device=device,
        world_size=world_size,
        dist_url=dist_url,
        sync_bn=sync_bn,
        fp16_eval=fp16_eval,
        encoder_only=encoder_only,
        backbone_only=backbone_only,
        resolution=resolution,
        use_cls_token=use_cls_token,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
        lr_min_factor=lr_min_factor,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_use_ema=early_stopping_use_ema,
        gradient_checkpointing=gradient_checkpointing,
        **extra_kwargs,
    )
    return args
