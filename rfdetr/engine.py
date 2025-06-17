# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""

import math
from typing import Iterable

import torch

import rfdetr.util.misc as utils
from rfdetr.datasets.coco_eval import CocoEvaluator

try:
    from torch.amp import autocast, GradScaler

    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

    DEPRECATED_AMP = True
from typing import DefaultDict, List, Callable, Dict, Any, Tuple
from rfdetr.util.misc import NestedTensor


def get_autocast_args(args):
    """Get autocast arguments based on PyTorch version and hardware support.

    Args:
        args: Namespace containing amp flag and other training arguments

    Returns:
        dict: Arguments to pass to autocast context manager
    """
    # Prefer bfloat16 if available, otherwise use float16
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    if DEPRECATED_AMP:
        return {"enabled": args.amp, "dtype": dtype}
    else:
        return {"device_type": "cuda", "enabled": args.amp, "dtype": dtype}


def update_dropout_schedules(
    model: torch.nn.Module,
    schedules: Dict[str, List[float]],
    iteration: int,
    is_distributed: bool,
    vit_encoder_num_layers: int,
) -> None:
    """Update dropout and drop path rates based on schedules.

    Args:
        model: The model to update
        schedules: Dictionary containing 'dp' and/or 'do' schedules
        iteration: Current training iteration
        is_distributed: Whether using distributed training
        vit_encoder_num_layers: Number of ViT encoder layers
    """
    if "dp" in schedules:
        if is_distributed:
            model.module.update_drop_path(
                schedules["dp"][iteration], vit_encoder_num_layers
            )
        else:
            model.update_drop_path(schedules["dp"][iteration], vit_encoder_num_layers)

    if "do" in schedules:
        if is_distributed:
            model.module.update_dropout(schedules["do"][iteration])
        else:
            model.update_dropout(schedules["do"][iteration])


def compute_losses(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    samples: NestedTensor,
    targets: List[Dict[str, Any]],
    device: torch.device,
    args: TrainingArgs,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute losses for a batch of samples.

    Args:
        model: The model to run inference
        criterion: Loss computation module
        samples: Input samples
        targets: Ground truth targets
        device: Device to run on
        args: Training arguments

    Returns:
        Tuple of (total_loss, loss_dict)
    """
    samples = samples.to(device)
    targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]

    with autocast(**get_autocast_args(args)):
        outputs = model(samples, targets_device)
        loss_dict = criterion(outputs, targets_device)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

    return losses, loss_dict


def process_gradient_accumulation_batch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    samples: NestedTensor,
    targets: List[Dict[str, Any]],
    device: torch.device,
    args: TrainingArgs,
    scaler: GradScaler,
    sub_batch_size: int,
    grad_accum_steps: int,
) -> Dict[str, torch.Tensor]:
    """Process a batch with gradient accumulation.

    Args:
        model: The model to train
        criterion: Loss computation module
        samples: Input samples
        targets: Ground truth targets
        device: Device to run on
        args: Training arguments
        scaler: Gradient scaler for AMP
        sub_batch_size: Size of each sub-batch
        grad_accum_steps: Number of gradient accumulation steps

    Returns:
        Dictionary of losses from the last sub-batch
    """
    loss_dict: Dict[str, torch.Tensor] = {}

    for i in range(grad_accum_steps):
        start_idx = i * sub_batch_size
        final_idx = start_idx + sub_batch_size
        new_samples_tensors = samples.tensors[start_idx:final_idx]
        new_samples = NestedTensor(
            new_samples_tensors, samples.mask[start_idx:final_idx]
        )
        new_targets = targets[start_idx:final_idx]

        losses, loss_dict = compute_losses(
            model, criterion, new_samples, new_targets, device, args
        )

        # Scale loss by gradient accumulation steps
        scaled_losses = (1 / grad_accum_steps) * losses
        scaler.scale(scaled_losses).backward()

    return loss_dict


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable[Tuple[NestedTensor, List[Dict[str, Any]]]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: Union[torch.nn.Module, None] = None,
    schedules: Dict[str, List[float]] = {},
    num_training_steps_per_epoch: Union[int, None] = None,
    vit_encoder_num_layers: Union[int, None] = None,
    args: Union[TrainingArgs, None] = None,
    callbacks: Union[DefaultDict[str, List[Callable[..., None]]], None] = None,
) -> Dict[str, float]:
    """Train model for one epoch.

    Args:
        model: Model to train
        criterion: Loss computation module
        lr_scheduler: Learning rate scheduler
        data_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        batch_size: Batch size
        max_norm: Maximum gradient norm for clipping
        ema_m: Exponential moving average model
        schedules: Dropout/drop path schedules
        num_training_steps_per_epoch: Number of training steps per epoch
        vit_encoder_num_layers: Number of ViT encoder layers
        args: Training arguments
        callbacks: Training callbacks

    Returns:
        Dictionary of training metrics
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    start_steps = epoch * num_training_steps_per_epoch

    print("Grad accum steps: ", args.grad_accum_steps)
    print("Total batch size: ", batch_size * utils.get_world_size())

    # Add gradient scaler for AMP
    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler("cuda", enabled=args.amp)

    optimizer.zero_grad()
    assert batch_size % args.grad_accum_steps == 0
    sub_batch_size = batch_size // args.grad_accum_steps
    print("LENGTH OF DATA LOADER:", len(data_loader))
    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        it = start_steps + data_iter_step
        # Execute batch start callbacks
        if callbacks is not None:
            callback_dict = {
                "step": it,
                "model": model,
                "epoch": epoch,
            }
            for callback in callbacks["on_train_batch_start"]:
                callback(callback_dict)

        # Update dropout schedules
        update_dropout_schedules(
            model, schedules, it, args.distributed, vit_encoder_num_layers
        )

        # Process batch with gradient accumulation
        loss_dict = process_gradient_accumulation_batch(
            model,
            criterion,
            samples,
            targets,
            device,
            args,
            scaler,
            sub_batch_size,
            args.grad_accum_steps,
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        weight_dict = criterion.weight_dict
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(loss_dict_reduced)
            raise ValueError("Loss is {}, stopping training".format(loss_value))

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def process_evaluation_outputs(
    outputs: Dict[str, torch.Tensor], fp16_eval: bool
) -> Dict[str, torch.Tensor]:
    """Convert FP16 outputs to FP32 for evaluation if needed.

    Args:
        outputs: Model outputs dictionary
        fp16_eval: Whether FP16 evaluation is enabled

    Returns:
        Processed outputs dictionary
    """
    if not fp16_eval:
        return outputs

    for key in outputs.keys():
        if key == "enc_outputs":
            for sub_key in outputs[key].keys():
                outputs[key][sub_key] = outputs[key][sub_key].float()
        elif key == "aux_outputs":
            for idx in range(len(outputs[key])):
                for sub_key in outputs[key][idx].keys():
                    outputs[key][idx][sub_key] = outputs[key][idx][sub_key].float()
        else:
            outputs[key] = outputs[key].float()

    return outputs


def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessors: Dict[str, torch.nn.Module],
    data_loader: Iterable[Tuple[NestedTensor, List[Dict[str, Any]]]],
    base_ds: Any,  # CocoDataset type, but keeping Any to avoid import
    device: torch.device,
    args: Union[TrainingArgs, None] = None
) -> Tuple[Dict[str, Any], CocoEvaluator]:
    """Evaluate model on validation dataset.

    Args:
        model: Model to evaluate
        criterion: Loss computation module
        postprocessors: Postprocessing modules
        data_loader: Validation data loader
        base_ds: Base dataset for COCO evaluation
        device: Device to run evaluation on
        args: Evaluation arguments

    Returns:
        Tuple of (stats_dict, coco_evaluator)
    """
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Add autocast for evaluation with no_grad
        with torch.no_grad():
            if args is not None and args.amp:
                with autocast(**get_autocast_args(args)):
                    outputs = model(samples)
            else:
                outputs = model(samples)

        # Convert FP16 outputs to FP32 if needed
        outputs = process_evaluation_outputs(
            outputs, args.amp if args is not None else False
        )

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator
