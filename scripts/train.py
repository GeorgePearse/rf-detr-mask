#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR-Mask Training Script for CMR Dataset
# ------------------------------------------------------------------------

"""
Script to train RF-DETR with mask head on CMR segmentation data.
Adapted from the original RF-DETR training script to work with CMR instance segmentation dataset.


TODO: This script and main.py overlap far too much
"""

import argparse
import datetime
import json
import os
import random
import sys
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
    parser = argparse.ArgumentParser('Train RF-DETR-Mask on CMR segmentation', add_help=True)
    
    # Dataset parameters - using CMR dataset paths
    parser.add_argument('--dataset', default='coco', type=str,
                        help='Dataset name')
    parser.add_argument('--dataset_file', default='coco', type=str,
                        help='Dataset file name')
    parser.add_argument('--coco_path', type=str,
                        default='/home/georgepearse/data/cmr/annotations',
                        help='Path to the annotations directory')
    parser.add_argument('--coco_train', type=str,
                        default='2025-05-15_12:38:23.077836_train_ordered.json',
                        help='Training annotation file name')
    parser.add_argument('--coco_val', type=str,
                        default='2025-05-15_12:38:38.270134_val_ordered.json',
                        help='Validation annotation file name')
    parser.add_argument('--coco_img_path', type=str,
                        default='/home/georgepearse/data/images',
                        help='Path to the images directory')
    parser.add_argument('--output_dir', default='output_cmr_segmentation',
                        help='Path to save logs and checkpoints')
    
    # Training parameters
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--lr_encoder', default=1e-5, type=float,
                        help='Learning rate of the encoder')
    parser.add_argument('--lr_projector', default=1e-5, type=float,
                        help='Learning rate of the projector')
    parser.add_argument('--lr_vit_layer_decay', default=1.0, type=float,
                        help='Layer-wise learning rate decay for ViT')
    parser.add_argument('--lr_component_decay', default=0.9, type=float,
                        help='Component-wise learning rate decay')
    parser.add_argument('--lr_drop', default=50, type=int, 
                        help='lr_drop')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per device')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs to train for')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='Gradient clipping max norm')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of gradient accumulation steps')
    
    # Model parameters
    parser.add_argument('--encoder', default='dinov2_base', type=str,
                        help='Name of the transformer backbone')
    parser.add_argument('--pretrain_weights', type=str, default=None,
                        help='Path to pretrained weights')
    parser.add_argument('--resolution', default=644, type=int,
                        help='Input resolution to the encoder (must be divisible by 14 for DINOv2)')
    parser.add_argument('--set_loss', default='lw_detr', type=str,
                        help='Type of loss for object detection matching')
    parser.add_argument('--set_cost_class', default=5, type=float,
                        help='Class coefficient in the matching cost')
    parser.add_argument('--set_cost_bbox', default=2, type=float,
                        help='Bounding box L1 coefficient in the matching cost')
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help='giou coefficient in the matching cost')
    parser.add_argument('--loss_class_coef', default=4.5, type=float,
                        help='coefficient for loss on classification')
    parser.add_argument('--loss_bbox_coef', default=2.0, type=float,
                        help='coefficient for loss on bounding box regression')
    parser.add_argument('--loss_giou_coef', default=1, type=float,
                        help='coefficient for loss on bounding box giou')
    parser.add_argument('--num_classes', default=69, type=int,  # CMR has 69 classes
                        help='Number of classes')
    parser.add_argument('--masks', action='store_true', default=True,
                        help='Train segmentation head for panoptic segmentation')
    
    # Loss parameters for masks
    parser.add_argument('--loss_mask_coef', default=1.0, type=float,
                        help='coefficient for loss on mask prediction')
    parser.add_argument('--loss_dice_coef', default=1.0, type=float,
                        help='coefficient for loss on dice loss')
    
    # Other parameters
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--eval', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers for data loading')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--sync_bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch native sync batchnorm.')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--dropout', default=0.0, type=float, 
                        help='dropout rate that applies to transformer backbone')
    parser.add_argument('--bbox_reparam', default=True, type=bool,
                        help='reparameterize bbox loss (CWH)')
    parser.add_argument('--group_detr', default=1, type=int,
                        help='Number of groups for group DETR')
    parser.add_argument('--two_stage', default=True, type=bool,
                        help='Use two-stage variant of DETR')
    parser.add_argument('--no_intermittent_layers', default=False, type=bool,
                        help='Avoid computing intermediate decodings')
    parser.add_argument('--use_fp16', default=True, type=bool,
                        help='Use FP16 models (half)')
    parser.add_argument('--amp', action='store_true',
                        help='use mixed precision')
    parser.add_argument('--square_resize', action='store_true',
                        help='use square resize for images')
    parser.add_argument('--square_resize_div_64', action='store_true',
                        help='use square resize with dimensions divisible by 64')
    
    return parser


def main(args):
    # Initialize distributed mode
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)
    
    device = torch.device(args.device)
    
    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Build the model
    model = build_model(args)
    
    # Build criterion and postprocessors
    criterion, postprocessors = build_criterion_and_postprocessors(args)
    model.to(device)
    
    # Wrap with Distributed Data Parallel if needed
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {n_parameters}')
    
    # Build optimizer
    param_dicts = get_param_dict(args, model_without_ddp)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    # Build learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.lr_drop], gamma=0.1)
    
    # Build datasets
    dataset_train = build_dataset(image_set='train', args=args, resolution=args.resolution)
    dataset_val = build_dataset(image_set='val', args=args, resolution=args.resolution)
    
    # Build data samplers
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    # Build data loaders
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, 
                                 num_workers=args.num_workers)
    
    # Create model EMA if needed
    ema_m = getattr(args, 'ema', None)
    if ema_m:
        ema = ModelEma(model_without_ddp, ema_m)
    else:
        ema = None
    
    # Get COCO API for evaluation
    base_ds = get_coco_api_from_dataset(dataset_val)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        if ema and 'ema' in checkpoint:
            ema.ema.load_state_dict(checkpoint['ema'])
    
    # Evaluation only mode
    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        print("\nEvaluation Results:")
        for k, v in test_stats.items():
            if isinstance(v, list):
                print(f"{k}: {v}")
        return
    
    # Create metrics holders
    best_map_holder = BestMetricHolder(use_ema=False)
    
    # Training loop
    print("Starting training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # Set epoch for sampler
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        # Calculate number of training steps per epoch
        num_training_steps_per_epoch = len(data_loader_train)
        
        # Create empty callbacks dictionary
        from collections import defaultdict
        callbacks = defaultdict(list)
        
        # Train for one epoch
        train_stats = train_one_epoch(
            model, criterion, lr_scheduler, data_loader_train, optimizer, device, epoch,
            args.batch_size, args.clip_max_norm, 
            num_training_steps_per_epoch=num_training_steps_per_epoch, 
            vit_encoder_num_layers=args.vit_encoder_num_layers,
            args=args,
            callbacks=callbacks)
        
        # Update LR scheduler
        lr_scheduler.step()
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            checkpoint_path = output_dir / f'checkpoint_{epoch:04d}.pth'
            checkpoint_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if ema:
                checkpoint_dict['ema'] = ema.ema.state_dict()
            utils.save_on_master(checkpoint_dict, checkpoint_path)
        
        # Evaluate
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
        
        # Check if this is the best model so far
        map_regular = test_stats['coco_eval_bbox'][0]
        segm_map = test_stats.get('coco_eval_masks', [0])[0] if args.masks else 0
        
        # Update best metrics
        _is_best = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _is_best:
            checkpoint_path = output_dir / 'checkpoint_best.pth'
            checkpoint_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if ema:
                checkpoint_dict['ema'] = ema.ema.state_dict()
            utils.save_on_master(checkpoint_dict, checkpoint_path)
        
        # Log statistics
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training completed in {total_time_str}')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Set some defaults for compatibility
    args.focal_loss = True
    args.focal_alpha = 0.25
    args.focal_gamma = 2.0
    args.num_queries = 900
    args.hidden_dim = 256
    args.position_embedding_scale = None
    args.backbone_feature_layers = ['res2', 'res3', 'res4', 'res5']
    args.vit_encoder_num_layers = 12
    args.num_decoder_layers = 6
    args.num_decoder_points = 4
    args.dec_layers = 6
    
    # Add missing attributes for build_model
    args.pretrained_encoder = True  # Use pretrained encoder by default
    args.window_block_indexes = []  # Empty list for window block indexes
    args.drop_path = 0.1  # Default drop path rate
    args.out_feature_indexes = [3, 7, 11]  # Default output features
    args.projector_scale = ['P3', 'P4', 'P5']  # Default projector scale levels, let's try without P6
    args.use_cls_token = True  # Use CLS token
    args.position_embedding = 'sine'  # Use sine position embedding
    args.freeze_encoder = False  # Don't freeze encoder by default
    args.layer_norm = True  # Use layer normalization
    args.rms_norm = False  # Don't use RMS normalization
    args.backbone_lora = False  # No LoRA for backbone
    args.force_no_pretrain = False  # Use pretrained weights
    args.gradient_checkpointing = False  # No gradient checkpointing by default
    args.encoder_only = False  # Use full DETR model, not just encoder
    args.backbone_only = False  # Use full model, not just backbone
    
    # Transformer parameters
    args.sa_nheads = 8  # Self-attention heads
    args.ca_nheads = 8  # Cross-attention heads
    args.dim_feedforward = 2048  # Feedforward dimension
    args.num_feature_levels = 3  # Number of feature levels for multi-scale (matching projector scale)
    args.dec_n_points = 4  # Number of attention points for decoder
    args.lite_refpoint_refine = False  # Lightweight reference point refinement
    args.decoder_norm = 'LN'  # Type of normalization in decoder (LN or Identity)
    
    # Additional model parameters
    args.aux_loss = True  # Use auxiliary loss in decoder layers
    
    # Map loss coefficient names to match build_criterion_and_postprocessors
    args.cls_loss_coef = args.loss_class_coef
    args.bbox_loss_coef = args.loss_bbox_coef
    args.giou_loss_coef = args.loss_giou_coef
    
    # Additional loss configuration
    args.use_varifocal_loss = False  # Not using varifocal loss
    args.mask_loss_coef = args.loss_mask_coef
    args.dice_loss_coef = args.loss_dice_coef
    args.use_position_supervised_loss = False  # Not using position supervised loss
    args.ia_bce_loss = False  # Not using instance-aware BCE loss
    args.sum_group_losses = False  # Don't sum group losses
    args.num_select = 300  # Number of top predictions to select in postprocessing
    
    # Data augmentation parameters
    args.multi_scale = False  # Don't use multi-scale training by default
    args.expanded_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]  # Multi-scale options
    args.square_resize = True  # Use square resize to ensure compatibility with Dinov2
    args.square_resize_div_64 = False  # Don't force div 64, we'll handle Dinov2 div 14
    
    # Additional parameter mapping
    args.grad_accum_steps = args.gradient_accumulation_steps
    args.fp16_eval = args.use_fp16  # Use FP16 for evaluation
    
    main(args)
