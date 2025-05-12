#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Script to train RF-DETR on COCO segmentation data.
This is a longer-running test script that performs training and evaluation 
on the full COCO dataset with segmentation annotations.
"""

import argparse
import os
import sys
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate, train_one_epoch
from rfdetr.models import build_model, build_criterion_and_postprocessors
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.utils import ModelEma, BestMetricHolder


def get_args_parser():
    parser = argparse.ArgumentParser('Train RF-DETR on COCO segmentation', add_help=True)
    
    # Dataset parameters
    parser.add_argument('--coco_path', type=str, required=True,
                        help='Path to the COCO dataset')
    parser.add_argument('--output_dir', default='output_segmentation',
                        help='Path to save logs and checkpoints')
    
    # Training parameters
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=12, type=int,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--lr_drop', default=11, type=int,
                        help='Epoch at which to drop learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='Gradient clipping max norm')
    parser.add_argument('--grad_accum_steps', default=1, type=int,
                        help='Gradient accumulation steps')
    parser.add_argument('--amp', action='store_true',
                        help='Use Automatic Mixed Precision')
    
    # Model parameters
    parser.add_argument('--encoder', default='vit_base', type=str,
                        help='Name of the transformer backbone')
    parser.add_argument('--pretrain_weights', type=str, default=None,
                        help='Path to pretrained weights')
    parser.add_argument('--resolution', type=int, default=640,
                        help='Input resolution')
    parser.add_argument('--num_classes', default=91, type=int,  # COCO has 80 classes + 1 background
                        help='Number of classes')
    
    # Other parameters
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--resume', default='', 
                        help='Resume from checkpoint')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of workers for data loading')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')
    
    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--sync_bn', action='store_true',
                        help='Use synchronized batch normalization')
    
    return parser


def main(args):
    # Initialize distributed mode if needed
    utils.init_distributed_mode(args)
    
    # Print arguments
    print(args)
    
    # Set device
    device = torch.device(args.device)
    
    # Fix random seeds
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create model
    model = build_model(args)
    model.to(device)
    
    # Create criterion and postprocessors
    criterion, postprocessors = build_criterion_and_postprocessors(args)
    criterion.to(device)
    
    # Setup for distributed training
    model_without_ddp = model
    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # Print number of parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters:', n_parameters)
    
    # Create parameter groups for optimizer
    param_dicts = get_param_dict(args, model_without_ddp)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    # Create EMA model if needed
    ema_m = None
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint if resuming
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    
    # Create datasets
    dataset_train = build_dataset(image_set='train', args=args, resolution=args.resolution)
    dataset_val = build_dataset(image_set='val', args=args, resolution=args.resolution)
    
    # Create samplers for data loaders
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    # Create batch samplers and data loaders
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn, num_workers=args.num_workers
    )
    data_loader_val = DataLoader(
        dataset_val, args.batch_size, sampler=sampler_val,
        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers
    )
    
    # Create COCO API for evaluation
    base_ds = get_coco_api_from_dataset(dataset_val)
    
    # Evaluation only
    if args.eval_only:
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args)
        
        # Print evaluation results
        if utils.is_main_process():
            print("Test results:")
            for k, v in test_stats.items():
                print(f"{k}: {v}")
        return
    
    # Create metrics holders
    best_map_holder = BestMetricHolder(use_ema=False)
    
    # Training loop
    print("Starting training")
    start_epoch = getattr(args, 'start_epoch', 0)
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for sampler
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        # Train for one epoch
        train_stats = train_one_epoch(
            model, criterion, lr_scheduler, data_loader_train, optimizer, device, epoch,
            args.batch_size, args.clip_max_norm, ema_m=ema_m, 
            num_training_steps_per_epoch=len(data_loader_train),
            vit_encoder_num_layers=12, args=args, callbacks={"on_train_batch_start": []}
        )
        
        # Update LR scheduler
        lr_scheduler.step()
        
        # Save checkpoint
        if args.output_dir:
            checkpoint_path = output_dir / 'checkpoint.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        
        # Evaluate
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args)
        
        # Check if this is the best model so far
        map_regular = test_stats['coco_eval_bbox'][0]
        segm_map = test_stats.get('coco_eval_masks', [0])[0]
        
        # Update best metrics
        _is_best = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _is_best:
            checkpoint_path = output_dir / 'checkpoint_best.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        
        # Log stats
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        
        # Print current time
        log_stats['now_time'] = str(datetime.datetime.now())
        
        # Write stats to log file
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            # Save COCO evaluation results
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    torch.save(
                        coco_evaluator.coco_eval["bbox"].eval,
                        output_dir / "eval" / f"{epoch:03}.pth"
                    )
                if "segm" in coco_evaluator.coco_eval:
                    torch.save(
                        coco_evaluator.coco_eval["segm"].eval,
                        output_dir / "eval" / f"segm_{epoch:03}.pth"
                    )
    
    # Print total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Create full args with default values
    full_args = utils.namespace_to_dict(args)
    from rfdetr.main import populate_args
    args = populate_args(**full_args)
    
    main(args)