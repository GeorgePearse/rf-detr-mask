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
import contextlib
import copy
import datetime
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate, train_one_epoch
from rfdetr.models import build_criterion_and_postprocessors, build_model
from rfdetr.util.benchmark import benchmark
from rfdetr.util.drop_scheduler import drop_scheduler
from rfdetr.util.files import download_file
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.utils import BestMetricHolder, ModelEma, clean_state_dict

if str(os.environ.get("USE_FILE_SYSTEM_SHARING", "False")).lower() in ["true", "1"]:
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")

logger = getLogger(__name__)

HOSTED_MODELS = {
    "rf-detr-base.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    # below is a less converged model that may be better for finetuning but worse for inference
    "rf-detr-base-2.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "rf-detr-large.pth": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
}


def download_pretrain_weights(pretrain_weights: str, redownload=False):
    if pretrain_weights in HOSTED_MODELS and (redownload or not os.path.exists(pretrain_weights)):
        logger.info(f"Downloading pretrained weights for {pretrain_weights}")
        download_file(
            HOSTED_MODELS[pretrain_weights],
            pretrain_weights,
        )


class Model:
    def __init__(self, **kwargs):
        args = populate_args(**kwargs)
        self._initialize_model_structure(args)
        self._load_pretrained_weights(args)
        self._apply_lora_if_needed(args)
        self._finalize_model_setup(args)

    def _initialize_model_structure(self, args):
        """Initialize the base model structure."""
        self.resolution = args.resolution
        self.model = build_model(args)
        self.device = torch.device(args.device)
        _, self.postprocessors = build_criterion_and_postprocessors(args)
        self.stop_early = False

    def _load_pretrained_weights(self, args):
        """Load pretrained weights if specified."""
        if args.pretrain_weights is None:
            return

        print("Loading pretrain weights")
        checkpoint = self._load_checkpoint(args.pretrain_weights)

        # Extract class_names from checkpoint if available
        if "args" in checkpoint and hasattr(checkpoint["args"], "class_names"):
            self.class_names = checkpoint["args"].class_names

        # Handle class mismatch
        self._handle_class_mismatch(args, checkpoint)

        # Process exclude keys
        self._process_exclude_keys(args, checkpoint)

        # Process keys to modify
        self._process_keys_to_modify(args, checkpoint)

        # Update query parameters for group detr
        self._update_query_params(args, checkpoint)

        # Load state dict
        self.model.load_state_dict(checkpoint["model"], strict=False)

    def _load_checkpoint(self, weights_path):
        """Load checkpoint with retry on failure."""
        try:
            return torch.load(weights_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Failed to load pretrain weights: {e}")
            print("Failed to load pretrain weights, re-downloading")
            download_pretrain_weights(weights_path, redownload=True)
            return torch.load(weights_path, map_location="cpu", weights_only=False)

    def _handle_class_mismatch(self, args, checkpoint):
        """Handle mismatch between checkpoint classes and model classes."""
        checkpoint_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
        if checkpoint_num_classes != args.num_classes + 1:
            logger.warning(
                f"num_classes mismatch: pretrain weights has {checkpoint_num_classes - 1} classes, "
                f"but your model has {args.num_classes} classes\n"
                f"reinitializing detection head with {checkpoint_num_classes - 1} classes"
            )
            self.reinitialize_detection_head(checkpoint_num_classes)

    def _process_exclude_keys(self, args, checkpoint):
        """Process keys to exclude from checkpoint."""
        if args.pretrain_exclude_keys is not None:
            assert isinstance(args.pretrain_exclude_keys, list)
            for exclude_key in args.pretrain_exclude_keys:
                checkpoint["model"].pop(exclude_key)

    def _process_keys_to_modify(self, args, checkpoint):
        """Process keys that need modification before loading."""
        if args.pretrain_keys_modify_to_load is not None:
            from rfdetr.util.obj365_to_coco_model import get_coco_pretrain_from_obj365

            assert isinstance(args.pretrain_keys_modify_to_load, list)
            for modify_key_to_load in args.pretrain_keys_modify_to_load:
                try:
                    checkpoint["model"][modify_key_to_load] = get_coco_pretrain_from_obj365(
                        self.model.state_dict()[modify_key_to_load],
                        checkpoint["model"][modify_key_to_load],
                    )
                except Exception as e:
                    print(f"Failed to load {modify_key_to_load}, deleting from checkpoint: {e}")
                    checkpoint["model"].pop(modify_key_to_load)

    def _update_query_params(self, args, checkpoint):
        """Update query parameters for group detr."""
        num_desired_queries = args.num_queries * args.group_detr
        query_param_names = ["refpoint_embed.weight", "query_feat.weight"]
        for name, state in checkpoint["model"].items():
            if any(name.endswith(x) for x in query_param_names):
                checkpoint["model"][name] = state[:num_desired_queries]

    def _apply_lora_if_needed(self, args):
        """Apply LoRA to backbone if needed."""
        if not args.backbone_lora:
            return

        print("Applying LORA to backbone")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            use_dora=True,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",  # covers OWL-ViT
                "qkv",  # covers open_clip ie Siglip2
                "query",
                "key",
                "value",
                "cls_token",
                "register_tokens",  # covers Dinov2 with windowed attn
            ],
        )
        self.model.backbone[0].encoder = get_peft_model(self.model.backbone[0].encoder, lora_config)

    def _finalize_model_setup(self, args):
        """Final model setup steps."""
        self.model = self.model.to(self.device)
        self.criterion, self.postprocessors = build_criterion_and_postprocessors(args)

    def reinitialize_detection_head(self, num_classes):
        self.model.reinitialize_detection_head(num_classes)

    def request_early_stop(self):
        self.stop_early = True
        print("Early stopping requested, will complete current epoch and stop")

    def train(self, callbacks: defaultdict[str, list[Callable]], **kwargs):
        """Main training loop for the model."""
        # Validate callbacks and setup the environment
        self._validate_callbacks(callbacks)
        args = populate_args(**kwargs)
        model, model_without_ddp, device = self._setup_training_environment(args)

        # Setup data and optimization components
        criterion, postprocessors = build_criterion_and_postprocessors(args)
        optimizer, lr_scheduler = self._setup_optimizer(model_without_ddp, args)

        # Setup datasets and dataloaders
        data_loader_train, data_loader_val, base_ds, n_parameters = (
            self._setup_datasets_and_loaders(args, model_without_ddp, kwargs)
        )

        # Handle resuming from checkpoint
        if args.resume:
            self._resume_from_checkpoint(args, model_without_ddp, optimizer, lr_scheduler)

        # Handle evaluation mode
        if args.eval:
            self._run_evaluation(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args
            )
            return

        # Setup drop schedules
        schedules = self._setup_drop_schedules(args, data_loader_train)

        # Main training loop
        best_metrics = self._run_training_loop(
            args,
            model,
            model_without_ddp,
            criterion,
            optimizer,
            lr_scheduler,
            data_loader_train,
            data_loader_val,
            base_ds,
            postprocessors,
            device,
            n_parameters,
            schedules,
            callbacks,
        )

        # Save final results
        self._save_final_results(args, best_metrics)

        # Finalize model
        self._finalize_model(best_metrics)

        # Call final callbacks
        for callback in callbacks["on_train_end"]:
            callback()

    def _validate_callbacks(self, callbacks):
        """Validate that all callbacks are supported."""
        currently_supported_callbacks = ["on_fit_epoch_end", "on_train_batch_start", "on_train_end"]
        for key in callbacks:
            if key not in currently_supported_callbacks:
                raise ValueError(
                    f"Callback {key} is not currently supported, please file an issue if you need it!\n"
                    f"Currently supported callbacks: {currently_supported_callbacks}"
                )

    def _setup_training_environment(self, args):
        """Setup the training environment including distributed training."""
        utils.init_distributed_mode(args)
        print(f"git:\n  {utils.get_sha()}\n")
        print(args)
        device = torch.device(args.device)

        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Setup model
        model = self.model
        model.to(device)

        # Handle distributed training
        model_without_ddp = model
        if args.distributed:
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True
            )
            model_without_ddp = model.module

        return model, model_without_ddp, device

    def _setup_optimizer(self, model_without_ddp, args):
        """Setup optimizer and learning rate scheduler."""
        param_dicts = get_param_dict(args, model_without_ddp)
        param_dicts = [p for p in param_dicts if p["params"].requires_grad]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

        # For cosine annealing, create a lambda function for the scheduler
        lr_lambda = self._create_lr_lambda(args)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return optimizer, lr_scheduler

    def _create_lr_lambda(self, args):
        """Create a learning rate lambda function for the scheduler."""
        # These calculations are needed for the lr_lambda function
        dataset_train = build_dataset(image_set="train", args=args, resolution=args.resolution)
        total_batch_size_for_lr = args.batch_size * utils.get_world_size() * args.grad_accum_steps
        num_training_steps_per_epoch_lr = (
            len(dataset_train) + total_batch_size_for_lr - 1
        ) // total_batch_size_for_lr
        total_training_steps_lr = num_training_steps_per_epoch_lr * args.epochs
        warmup_steps_lr = num_training_steps_per_epoch_lr * args.warmup_epochs

        # Define the lambda function
        def lr_lambda(current_step: int):
            if current_step < warmup_steps_lr:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps_lr))
            else:
                # Cosine annealing from multiplier 1.0 down to lr_min_factor
                if args.lr_scheduler == "cosine":
                    progress = float(current_step - warmup_steps_lr) / float(
                        max(1, total_training_steps_lr - warmup_steps_lr)
                    )
                    return args.lr_min_factor + (1 - args.lr_min_factor) * 0.5 * (
                        1 + math.cos(math.pi * progress)
                    )
                elif args.lr_scheduler == "step":
                    if current_step < args.lr_drop * num_training_steps_per_epoch_lr:
                        return 1.0
                    else:
                        return 0.1

        return lr_lambda

    def _setup_datasets_and_loaders(self, args, model_without_ddp, kwargs):
        """Setup datasets and data loaders."""
        dataset_train = build_dataset(image_set="train", args=args, resolution=args.resolution)
        dataset_val = build_dataset(image_set="val", args=args, resolution=args.resolution)

        # Setup samplers
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        # Create data loaders
        effective_batch_size = args.batch_size * args.grad_accum_steps

        # Handle small datasets specially
        data_loader_train = self._create_train_loader(
            args, dataset_train, sampler_train, effective_batch_size, kwargs
        )

        # Create validation loader
        data_loader_val = DataLoader(
            dataset_val,
            args.batch_size,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )

        # Get COCO API for evaluation
        base_ds = get_coco_api_from_dataset(dataset_val)

        # Model EMA setup
        if args.use_ema:
            self.ema_m = ModelEma(model_without_ddp, decay=args.ema_decay, tau=args.ema_tau)
        else:
            self.ema_m = None

        # Run benchmark if requested
        output_dir = Path(args.output_dir)
        if utils.is_main_process() and args.do_benchmark:
            benchmark_model = copy.deepcopy(model_without_ddp)
            bm = benchmark(benchmark_model.float(), dataset_val, output_dir)
            print(json.dumps(bm, indent=2))
            del benchmark_model

        # Count parameters
        n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        print("number of params:", n_parameters)

        return data_loader_train, data_loader_val, base_ds, n_parameters

    def _create_train_loader(
        self, args, dataset_train, sampler_train, effective_batch_size, kwargs
    ):
        """Create the training data loader, handling small datasets specially."""
        min_batches = kwargs.get("min_batches", 5)
        if len(dataset_train) < effective_batch_size * min_batches:
            logger.info(
                f"Training with uniform sampler because dataset is too small: "
                f"{len(dataset_train)} < {effective_batch_size * min_batches}"
            )
            sampler = torch.utils.data.RandomSampler(
                dataset_train,
                replacement=True,
                num_samples=effective_batch_size * min_batches,
            )
            return DataLoader(
                dataset_train,
                batch_size=effective_batch_size,
                collate_fn=utils.collate_fn,
                num_workers=args.num_workers,
                sampler=sampler,
            )
        else:
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, effective_batch_size, drop_last=True
            )
            return DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=utils.collate_fn,
                num_workers=args.num_workers,
            )

    def _resume_from_checkpoint(self, args, model_without_ddp, optimizer, lr_scheduler):
        """Resume training from a checkpoint."""
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)

        # Handle EMA
        if args.use_ema:
            if "ema_model" in checkpoint:
                self.ema_m.module.load_state_dict(clean_state_dict(checkpoint["ema_model"]))
            else:
                del self.ema_m
                self.ema_m = ModelEma(model_without_ddp, decay=args.ema_decay, tau=args.ema_tau)

        # Resume optimizer and scheduler if not in eval mode
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    def _run_evaluation(
        self, model, criterion, postprocessors, data_loader_val, base_ds, device, args
    ):
        """Run evaluation only."""
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args
        )
        output_dir = Path(args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

    def _setup_drop_schedules(self, args, dataset_train):
        """Setup dropout and drop path schedules."""
        effective_batch_size = args.batch_size * args.grad_accum_steps
        total_batch_size = effective_batch_size * utils.get_world_size()
        num_training_steps_per_epoch = (
            len(dataset_train) + total_batch_size - 1
        ) // total_batch_size

        schedules = {}

        # Setup dropout schedule if needed
        if args.dropout > 0:
            schedules["do"] = drop_scheduler(
                args.dropout,
                args.epochs,
                num_training_steps_per_epoch,
                args.cutoff_epoch,
                args.drop_mode,
                args.drop_schedule,
            )
            print(
                "Min DO = {:.7f}, Max DO = {:.7f}".format(
                    min(schedules["do"]), max(schedules["do"])
                )
            )

        # Setup drop path schedule if needed
        if args.drop_path > 0:
            schedules["dp"] = drop_scheduler(
                args.drop_path,
                args.epochs,
                num_training_steps_per_epoch,
                args.cutoff_epoch,
                args.drop_mode,
                args.drop_schedule,
            )
            print(
                "Min DP = {:.7f}, Max DP = {:.7f}".format(
                    min(schedules["dp"]), max(schedules["dp"])
                )
            )

        return schedules, num_training_steps_per_epoch

    def _run_training_loop(
        self,
        args,
        model,
        model_without_ddp,
        criterion,
        optimizer,
        lr_scheduler,
        data_loader_train,
        data_loader_val,
        base_ds,
        postprocessors,
        device,
        n_parameters,
        schedules_tuple,
        callbacks,
    ):
        """Run the main training loop."""
        schedules, num_training_steps_per_epoch = schedules_tuple
        effective_batch_size = args.batch_size * args.grad_accum_steps
        output_dir = Path(args.output_dir)

        print("Start training")
        start_time = time.time()

        # Track best metrics
        best_metrics = {
            "map_holder": BestMetricHolder(use_ema=args.use_ema),
            "map_5095": 0,
            "map_50": 0,
            "map_ema_5095": 0,
            "map_ema_50": 0,
        }

        # Main epoch loop
        for epoch in range(args.start_epoch, args.epochs):
            if self.stop_early:
                print(f"Early stopping requested, stopping at epoch {epoch}")
                break

            # Process this epoch
            best_metrics = self._process_epoch(
                args,
                epoch,
                model,
                model_without_ddp,
                criterion,
                optimizer,
                lr_scheduler,
                data_loader_train,
                data_loader_val,
                base_ds,
                postprocessors,
                device,
                n_parameters,
                schedules,
                num_training_steps_per_epoch,
                best_metrics,
                output_dir,
                callbacks,
                effective_batch_size,
            )

        return best_metrics

    def _process_epoch(
        self,
        args,
        epoch,
        model,
        model_without_ddp,
        criterion,
        optimizer,
        lr_scheduler,
        data_loader_train,
        data_loader_val,
        base_ds,
        postprocessors,
        device,
        n_parameters,
        schedules,
        num_training_steps_per_epoch,
        best_metrics,
        output_dir,
        callbacks,
        effective_batch_size,
    ):
        """Process a single epoch of training."""
        epoch_start_time = time.time()

        # Set epoch for distributed training
        if args.distributed and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)

        # Set model and criterion to training mode
        model.train()
        criterion.train()

        # Get evaluation frequency
        eval_freq = getattr(args, "eval_freq_steps", 1000)

        # Train one epoch
        train_stats = train_one_epoch(
            model,
            criterion,
            lr_scheduler,
            data_loader_train,
            optimizer,
            device,
            epoch,
            effective_batch_size,
            args.clip_max_norm,
            ema_m=self.ema_m,
            schedules=schedules,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            vit_encoder_num_layers=args.vit_encoder_num_layers,
            args=args,
            callbacks=callbacks,
            eval_freq=eval_freq,
            val_data_loader=data_loader_val,
            base_ds=base_ds,
            postprocessors=postprocessors,
        )

        # Save checkpoints if needed
        self._save_epoch_checkpoints(
            args, epoch, model_without_ddp, optimizer, lr_scheduler, output_dir
        )

        # Evaluate model
        test_stats, coco_evaluator = self._evaluate_model(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args
        )

        # Update regular model metrics
        best_metrics = self._update_regular_model_metrics(
            args,
            epoch,
            model_without_ddp,
            optimizer,
            lr_scheduler,
            test_stats,
            best_metrics,
            output_dir,
        )

        # Process EMA model if enabled
        if args.use_ema:
            best_metrics = self._process_ema_model(
                args,
                epoch,
                optimizer,
                lr_scheduler,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                best_metrics,
                output_dir,
            )

        # Log results
        log_stats = self._create_log_stats(
            train_stats, test_stats, epoch, n_parameters, best_metrics, epoch_start_time
        )

        # Save evaluation logs
        self._save_evaluation_logs(args, epoch, coco_evaluator, output_dir, log_stats)

        # Call epoch end callbacks
        for callback in callbacks["on_fit_epoch_end"]:
            callback(log_stats)

        return best_metrics

    def _save_epoch_checkpoints(
        self, args, epoch, model_without_ddp, optimizer, lr_scheduler, output_dir
    ):
        """Save checkpoints at the end of an epoch."""
        if not args.output_dir:
            return

        checkpoint_paths = [output_dir / "checkpoint.pth"]

        # Save extra checkpoints before LR drop and at checkpoint intervals
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")

        # Prepare weights dictionary
        weights = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }

        # Add EMA model if using it
        if args.use_ema:
            weights.update({"ema_model": self.ema_m.module.state_dict()})

        # Save weights
        if not args.dont_save_weights:
            for checkpoint_path in checkpoint_paths:
                # Create checkpoint directory
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                utils.save_on_master(weights, checkpoint_path)

    def _evaluate_model(
        self, model, criterion, postprocessors, data_loader_val, base_ds, device, args
    ):
        """Evaluate the model on the validation set."""
        with torch.inference_mode():
            return evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args=args
            )

    def _update_regular_model_metrics(
        self,
        args,
        epoch,
        model_without_ddp,
        optimizer,
        lr_scheduler,
        test_stats,
        best_metrics,
        output_dir,
    ):
        """Update metrics for the regular (non-EMA) model."""
        map_regular = test_stats["coco_eval_bbox"][0]
        is_best = best_metrics["map_holder"].update(map_regular, epoch, is_ema=False)

        if is_best:
            best_metrics["map_5095"] = max(best_metrics["map_5095"], map_regular)
            best_metrics["map_50"] = max(best_metrics["map_50"], test_stats["coco_eval_bbox"][1])

            # Save best regular model checkpoint
            if not args.dont_save_weights and args.output_dir:
                checkpoint_path = output_dir / "checkpoint_best_regular.pth"
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        return best_metrics

    def _process_ema_model(
        self,
        args,
        epoch,
        optimizer,
        lr_scheduler,
        criterion,
        postprocessors,
        data_loader_val,
        base_ds,
        device,
        best_metrics,
        output_dir,
    ):
        """Process and evaluate the EMA model if enabled."""
        ema_test_stats, _ = evaluate(
            self.ema_m.module,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args=args,
        )

        map_ema = ema_test_stats["coco_eval_bbox"][0]
        best_metrics["map_ema_5095"] = max(best_metrics["map_ema_5095"], map_ema)

        is_best = best_metrics["map_holder"].update(map_ema, epoch, is_ema=True)
        if is_best:
            best_metrics["map_ema_50"] = max(
                best_metrics["map_ema_50"], ema_test_stats["coco_eval_bbox"][1]
            )

            # Save best EMA model checkpoint
            if not args.dont_save_weights and args.output_dir:
                checkpoint_path = output_dir / "checkpoint_best_ema.pth"
                utils.save_on_master(
                    {
                        "model": self.ema_m.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        return best_metrics, ema_test_stats

    def _create_log_stats(
        self, train_stats, test_stats, epoch, n_parameters, best_metrics, epoch_start_time
    ):
        """Create log statistics dictionary."""
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        # Add EMA stats if available
        if hasattr(self, "ema_test_stats"):
            log_stats.update({f"ema_test_{k}": v for k, v in self.ema_test_stats.items()})

        # Add best metric summary
        log_stats.update(best_metrics["map_holder"].summary())

        # Add timing information
        train_epoch_time = time.time() - epoch_start_time
        train_epoch_time_str = str(datetime.timedelta(seconds=int(train_epoch_time)))
        log_stats["train_epoch_time"] = train_epoch_time_str

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats["epoch_time"] = epoch_time_str

        # Add timestamp
        with contextlib.suppress(Exception):
            log_stats.update({"now_time": str(datetime.datetime.now())})

        return log_stats

    def _save_evaluation_logs(self, args, epoch, coco_evaluator, output_dir, log_stats):
        """Save evaluation logs."""
        if not (args.output_dir and utils.is_main_process()):
            return

        # Save to log file
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

        # Save evaluation results
        if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
            (output_dir / "eval").mkdir(exist_ok=True)
            filenames = ["latest.pth"]
            if epoch % 50 == 0:
                filenames.append(f"{epoch:03}.pth")
            for name in filenames:
                torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name)

    def _save_final_results(self, args, best_metrics):
        """Save final results and best model."""
        if not (args.output_dir and utils.is_main_process()):
            return

        output_dir = Path(args.output_dir)
        best_is_ema = best_metrics["map_ema_5095"] > best_metrics["map_5095"]

        # Copy best checkpoint
        if best_is_ema:
            shutil.copy2(
                output_dir / "checkpoint_best_ema.pth", output_dir / "checkpoint_best_total.pth"
            )
        else:
            shutil.copy2(
                output_dir / "checkpoint_best_regular.pth",
                output_dir / "checkpoint_best_total.pth",
            )

        # Strip checkpoint to reduce size
        utils.strip_checkpoint(output_dir / "checkpoint_best_total.pth")

        # Get best metrics
        best_map_5095 = max(best_metrics["map_5095"], best_metrics["map_ema_5095"])
        best_map_50 = max(best_metrics["map_50"], best_metrics["map_ema_50"])

        # Save results JSON
        results_json = {"map95": best_map_5095, "map50": best_map_50, "class": "all"}
        results = {"class_map": {"valid": [results_json]}}
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f)

        # Print final info
        total_time = time.time() - self.start_time if hasattr(self, "start_time") else 0
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")
        print("Results saved to {}".format(output_dir / "results.json"))

    def _finalize_model(self, best_metrics):
        """Finalize model state after training."""
        # Use EMA model if it's better
        best_is_ema = best_metrics["map_ema_5095"] > best_metrics["map_5095"]
        if best_is_ema and hasattr(self, "ema_m") and self.ema_m is not None:
            self.model = self.ema_m.module

        # Set model to eval mode
        self.model.eval()

    def export(
        self,
        output_dir="output",
        infer_dir=None,
        simplify=False,
        backbone_only=False,
        opset_version=17,
        verbose=True,
        force=False,
        shape=None,
        batch_size=1,
        **kwargs,
    ):
        """Export the trained model to ONNX format"""
        print("Exporting model to ONNX format")
        try:
            from rfdetr.deploy.export import export_onnx, make_infer_image, onnx_simplify
        except ImportError:
            print(
                "It seems some dependencies for ONNX export are missing. Please run `pip install rfdetr[onnxexport]` and try again."
            )
            raise

        device = self.device
        model = deepcopy(self.model.to("cpu"))
        model.to(device)

        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        if shape is None:
            shape = (self.resolution, self.resolution)
        else:
            if shape[0] % 14 != 0 or shape[1] % 14 != 0:
                raise ValueError("Shape must be divisible by 14")

        input_tensors = make_infer_image(infer_dir, shape, batch_size, device).to(device)
        input_names = ["input"]
        output_names = ["features"] if backbone_only else ["dets", "labels"]
        dynamic_axes = None
        self.model.eval()
        with torch.no_grad():
            if backbone_only:
                features = model(input_tensors)
                print(f"PyTorch inference output shape: {features.shape}")
            else:
                outputs = model(input_tensors)
                dets = outputs["pred_boxes"]
                labels = outputs["pred_logits"]
                print(
                    f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}"
                )
        model.cpu()
        input_tensors = input_tensors.cpu()

        # Export to ONNX
        output_file = export_onnx(
            output_dir=output_dir,
            model=model,
            input_names=input_names,
            input_tensors=input_tensors,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            backbone_only=backbone_only,
            verbose=verbose,
            opset_version=opset_version,
        )

        print(f"Successfully exported ONNX model to: {output_file}")

        if simplify:
            sim_output_file = onnx_simplify(
                onnx_dir=output_file,
                input_names=input_names,
                input_tensors=input_tensors,
                force=force,
            )
            print(f"Successfully simplified ONNX model to: {sim_output_file}")

        print("ONNX export completed successfully")
        self.model = self.model.to(device)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         "LWDETR training and evaluation script", parents=[get_args_parser()]
#     )
#     args = parser.parse_args()
#
#     if args.output_dir:
#         Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#
#     config = vars(args)  # Convert Namespace to dictionary
#
#     if args.subcommand == "distill":
#         distill(**config)
#     elif args.subcommand is None:
#         main(**config)
#     elif args.subcommand == "export_model":
#         filter_keys = [
#             "num_classes",
#             "grad_accum_steps",
#             "lr",
#             "lr_encoder",
#             "weight_decay",
#             "epochs",
#             "lr_drop",
#             "clip_max_norm",
#             "lr_vit_layer_decay",
#             "lr_component_decay",
#             "dropout",
#             "drop_path",
#             "drop_mode",
#             "drop_schedule",
#             "cutoff_epoch",
#             "pretrained_encoder",
#             "pretrain_weights",
#             "pretrain_exclude_keys",
#             "pretrain_keys_modify_to_load",
#             "freeze_florence",
#             "freeze_aimv2",
#             "decoder_norm",
#             "set_cost_class",
#             "set_cost_bbox",
#             "set_cost_giou",
#             "cls_loss_coef",
#             "bbox_loss_coef",
#             "giou_loss_coef",
#             "focal_alpha",
#             "aux_loss",
#             "sum_group_losses",
#             "use_varifocal_loss",
#             "use_position_supervised_loss",
#             "ia_bce_loss",
#             "dataset_file",
#             "coco_path",
#             "dataset_dir",
#             "square_resize_div_64",
#             "output_dir",
#             "checkpoint_interval",
#             "seed",
#             "resume",
#             "start_epoch",
#             "eval",
#             "use_ema",
#             "ema_decay",
#             "ema_tau",
#             "num_workers",
#             "device",
#             "world_size",
#             "dist_url",
#             "sync_bn",
#             "fp16_eval",
#             "infer_dir",
#             "verbose",
#             "opset_version",
#             "dry_run",
#             "shape",
#         ]
#         for key in filter_keys:
#             config.pop(key, None)  # Use pop with None to avoid KeyError
#
#         from deploy.export import main as export_main
#
#         if args.batch_size != 1:
#             config["batch_size"] = 1
#             print(
#                 f"Only batch_size 1 is supported for onnx export, \
#                  but got batchsize = {args.batch_size}. batch_size is forcibly set to 1."
#             )
#         export_main(**config)


def get_args_parser() -> argparse.ArgumentParser:
    """Get argument parser for command line interface."""
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
    parser.add_argument("--do_benchmark", action="store_true", help="benchmark the model")

    # drop args
    # dropout and stochastic depth drop rate; set at most one to non-zero
    parser.add_argument("--dropout", type=float, default=0, help="Drop path rate (default: 0.0)")
    parser.add_argument("--drop_path", type=float, default=0, help="Drop path rate (default: 0.0)")

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
        "--pretrained_encoder", type=str, default=None, help="Path to the pretrained encoder."
    )
    parser.add_argument(
        "--pretrain_weights", type=str, default=None, help="Path to the pretrained model."
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
        "--out_feature_indexes", default=[-1], type=int, nargs="+", help="only for vit now"
    )
    parser.add_argument("--freeze_encoder", action="store_true", dest="freeze_encoder")
    parser.add_argument("--layer_norm", action="store_true", dest="layer_norm")
    parser.add_argument("--rms_norm", action="store_true", dest="rms_norm")
    parser.add_argument("--backbone_lora", action="store_true", dest="backbone_lora")
    parser.add_argument("--force_no_pretrain", action="store_true", dest="force_no_pretrain")

    # * Transformer
    parser.add_argument(
        "--dec_layers", default=3, type=int, help="Number of decoding layers in the transformer"
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
    parser.add_argument("--num_queries", default=300, type=int, help="Number of query slots")
    parser.add_argument(
        "--group_detr", default=13, type=int, help="Number of groups to speed up detr training"
    )
    parser.add_argument(
        "--projector_scale", default="P4", type=str, nargs="+", choices=("P3", "P4", "P5", "P6")
    )
    parser.add_argument(
        "--lite_refpoint_refine", action="store_true", help="lite refpoint refine mode for speed-up"
    )
    parser.add_argument(
        "--num_select",
        default=100,
        type=int,
        help="the number of predictions selected for evaluation",
    )
    parser.add_argument("--dec_n_points", default=4, type=int, help="the number of sampling points")
    parser.add_argument("--decoder_norm", default="LN", type=str)
    parser.add_argument("--bbox_reparam", action="store_true")
    parser.add_argument("--freeze_batch_norm", action="store_true")
    # * Matcher
    parser.add_argument(
        "--set_cost_class", default=2, type=float, help="Class coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_bbox", default=5, type=float, help="L1 box coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_giou", default=2, type=float, help="giou box coefficient in the matching cost"
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
        "--checkpoint_interval", default=10, type=int, help="epoch interval to save checkpoint"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", default=0.9997, type=float)
    parser.add_argument("--ema_tau", default=0, type=float)

    parser.add_argument("--num_workers", default=2, type=int)

    # distributed training parameters
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
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
        "--fp16_eval", default=False, action="store_true", help="evaluate in fp16 precision."
    )

    # custom args
    parser.add_argument(
        "--encoder_only", action="store_true", help="Export and benchmark encoder only"
    )
    parser.add_argument(
        "--backbone_only", action="store_true", help="Export and benchmark backbone only"
    )
    parser.add_argument("--resolution", type=int, default=640, help="input resolution")
    parser.add_argument("--use_cls_token", action="store_true", help="use cls token")
    parser.add_argument("--multi_scale", action="store_true", help="use multi scale")
    parser.add_argument("--expanded_scales", action="store_true", help="use expanded scales")
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
    # In-training evaluation parameters
    parser.add_argument(
        "--eval_freq_steps",
        default=1000,
        type=int,
        help="Run COCO API evaluation every N steps during training",
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
    parser_export.add_argument("--simplify", action="store_true", help="Simplify onnx model")
    parser_export.add_argument(
        "--tensorrt", "--trtexec", "--trt", action="store_true", help="build tensorrt engine"
    )
    parser_export.add_argument(
        "--dry-run", "--test", "-t", action="store_true", help="just print command"
    )
    parser_export.add_argument(
        "--profile", action="store_true", help="Run nsys profiling during TensorRT export"
    )
    parser_export.add_argument(
        "--shape", type=int, nargs=2, default=(640, 640), help="input shape (width, height)"
    )
    return parser


def populate_args(**kwargs) -> argparse.Namespace:
    # Basic training parameters
    num_classes = kwargs.get("num_classes", 2)
    grad_accum_steps = kwargs.get("grad_accum_steps", 1)
    amp = kwargs.get("amp", False)
    lr = kwargs.get("lr", 1e-4)
    lr_encoder = kwargs.get("lr_encoder", 1.5e-4)
    batch_size = kwargs.get("batch_size", 2)
    weight_decay = kwargs.get("weight_decay", 1e-4)
    epochs = kwargs.get("epochs", 12)
    lr_drop = kwargs.get("lr_drop", 11)
    clip_max_norm = kwargs.get("clip_max_norm", 0.1)
    lr_vit_layer_decay = kwargs.get("lr_vit_layer_decay", 0.8)
    lr_component_decay = kwargs.get("lr_component_decay", 1.0)
    do_benchmark = kwargs.get("do_benchmark", False)

    # Drop parameters
    dropout = kwargs.get("dropout", 0)
    drop_path = kwargs.get("drop_path", 0)
    drop_mode = kwargs.get("drop_mode", "standard")
    drop_schedule = kwargs.get("drop_schedule", "constant")
    cutoff_epoch = kwargs.get("cutoff_epoch", 0)

    # Model parameters
    pretrained_encoder = kwargs.get("pretrained_encoder")
    pretrain_weights = kwargs.get("pretrain_weights")
    pretrain_exclude_keys = kwargs.get("pretrain_exclude_keys")
    pretrain_keys_modify_to_load = kwargs.get("pretrain_keys_modify_to_load")
    pretrained_distiller = kwargs.get("pretrained_distiller")

    # Backbone parameters
    encoder = kwargs.get("encoder", "vit_tiny")
    vit_encoder_num_layers = kwargs.get("vit_encoder_num_layers", 12)
    window_block_indexes = kwargs.get("window_block_indexes")
    position_embedding = kwargs.get("position_embedding", "sine")
    out_feature_indexes = kwargs.get("out_feature_indexes")
    freeze_encoder = kwargs.get("freeze_encoder", False)
    layer_norm = kwargs.get("layer_norm", False)
    rms_norm = kwargs.get("rms_norm", False)
    backbone_lora = kwargs.get("backbone_lora", False)
    force_no_pretrain = kwargs.get("force_no_pretrain", False)

    # Transformer parameters
    dec_layers = kwargs.get("dec_layers", 3)
    dim_feedforward = kwargs.get("dim_feedforward", 2048)
    hidden_dim = kwargs.get("hidden_dim", 256)
    sa_nheads = kwargs.get("sa_nheads", 8)
    ca_nheads = kwargs.get("ca_nheads", 8)
    num_queries = kwargs.get("num_queries", 300)
    group_detr = kwargs.get("group_detr", 13)
    projector_scale = kwargs.get("projector_scale", "P4")
    lite_refpoint_refine = kwargs.get("lite_refpoint_refine", False)
    num_select = kwargs.get("num_select", 100)
    dec_n_points = kwargs.get("dec_n_points", 4)
    decoder_norm = kwargs.get("decoder_norm", "LN")
    bbox_reparam = kwargs.get("bbox_reparam", False)
    freeze_batch_norm = kwargs.get("freeze_batch_norm", False)

    # Matcher parameters
    set_cost_class = kwargs.get("set_cost_class", 2)
    set_cost_bbox = kwargs.get("set_cost_bbox", 5)
    set_cost_giou = kwargs.get("set_cost_giou", 2)

    # Loss coefficients
    cls_loss_coef = kwargs.get("cls_loss_coef", 2)
    bbox_loss_coef = kwargs.get("bbox_loss_coef", 5)
    giou_loss_coef = kwargs.get("giou_loss_coef", 2)
    focal_alpha = kwargs.get("focal_alpha", 0.25)
    aux_loss = kwargs.get("aux_loss", True)
    sum_group_losses = kwargs.get("sum_group_losses", False)
    use_varifocal_loss = kwargs.get("use_varifocal_loss", False)
    use_position_supervised_loss = kwargs.get("use_position_supervised_loss", False)
    ia_bce_loss = kwargs.get("ia_bce_loss", False)

    # Dataset parameters
    dataset_file = kwargs.get("dataset_file", "coco")
    coco_path = kwargs.get("coco_path")
    dataset_dir = kwargs.get("dataset_dir")
    square_resize_div_64 = kwargs.get("square_resize_div_64", False)

    # Output parameters
    output_dir = kwargs.get("output_dir", "output")
    dont_save_weights = kwargs.get("dont_save_weights", False)
    checkpoint_interval = kwargs.get("checkpoint_interval", 10)
    seed = kwargs.get("seed", 42)
    resume = kwargs.get("resume", "")
    start_epoch = kwargs.get("start_epoch", 0)
    eval = kwargs.get("eval", False)
    use_ema = kwargs.get("use_ema", False)
    ema_decay = kwargs.get("ema_decay", 0.9997)
    ema_tau = kwargs.get("ema_tau", 0)
    num_workers = kwargs.get("num_workers", 2)

    # Distributed training parameters
    device = kwargs.get("device", "cuda")
    world_size = kwargs.get("world_size", 1)
    dist_url = kwargs.get("dist_url", "env://")
    sync_bn = kwargs.get("sync_bn", True)

    # FP16
    fp16_eval = kwargs.get("fp16_eval", False)

    # Custom args
    encoder_only = kwargs.get("encoder_only", False)
    backbone_only = kwargs.get("backbone_only", False)
    resolution = kwargs.get("resolution", 640)
    use_cls_token = kwargs.get("use_cls_token", False)
    multi_scale = kwargs.get("multi_scale", False)
    expanded_scales = kwargs.get("expanded_scales", False)
    warmup_epochs = kwargs.get("warmup_epochs", 1)
    lr_scheduler = kwargs.get("lr_scheduler", "step")
    lr_min_factor = kwargs.get("lr_min_factor", 0.0)

    # Early stopping parameters
    early_stopping = kwargs.get("early_stopping", True)
    early_stopping_patience = kwargs.get("early_stopping_patience", 10)
    early_stopping_min_delta = kwargs.get("early_stopping_min_delta", 0.001)
    early_stopping_use_ema = kwargs.get("early_stopping_use_ema", False)
    eval_freq_steps = kwargs.get("eval_freq_steps", 1000)
    gradient_checkpointing = kwargs.get("gradient_checkpointing", False)

    # Additional
    subcommand = kwargs.get("subcommand")

    # Handle extra kwargs - filter out keys that are explicitly declared above
    # to avoid duplicate arguments like 'encoder'
    all_locals = set(locals().keys())
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in all_locals}
    if out_feature_indexes is None:
        out_feature_indexes = [-1]
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
        eval_freq_steps=eval_freq_steps,
        gradient_checkpointing=gradient_checkpointing,
        **extra_kwargs,
    )
    return args
