#!/usr/bin/env python
"""Example of how to use refactored RF-DETR components for training.

This script demonstrates the new architecture without modifying the existing train.py.
Use this as a reference for gradual migration.
"""

import argparse
import torch
from pathlib import Path
from typing import Optional

from rfdetr.core.config import ConfigurationManager
from rfdetr.core.models import ModelFactory
from rfdetr.core.checkpoint import CheckpointManager
from rfdetr.core.training import Trainer, CallbackManager
from rfdetr.core.training.callbacks import (
    LoggingCallback,
    ModelCheckpointCallback,
    TensorBoardCallback,
    EarlyStoppingCallback
)
from rfdetr.core.data import DataLoaderFactory
from rfdetr.core.optimization import OptimizerFactory, SchedulerFactory


def create_argument_parser():
    """Create argument parser for training configuration."""
    parser = argparse.ArgumentParser(description='Train RF-DETR with refactored components')
    
    # Model configuration
    parser.add_argument('--model_name', default='base', choices=['base', 'large'],
                        help='Model variant to use')
    parser.add_argument('--num_classes', type=int, default=80,
                        help='Number of object classes')
    parser.add_argument('--masks', action='store_true',
                        help='Enable instance segmentation')
    parser.add_argument('--pretrain_weights', type=str, default=None,
                        help='Path to pretrained weights')
    
    # Training configuration
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory for outputs')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    
    # Hardware configuration
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')
    
    # Logging and checkpointing
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Steps between logging')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='Epochs between checkpoints')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable TensorBoard logging')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience (0 to disable)')
    
    # Testing configuration
    parser.add_argument('--test_mode', action='store_true',
                        help='Run in test mode with reduced data')
    parser.add_argument('--test_limit', type=int, default=20,
                        help='Number of samples to use in test mode')
    
    return parser


def setup_components(args):
    """Set up all training components using the refactored architecture."""
    # 1. Configuration Management
    config_manager = ConfigurationManager()
    
    model_config = config_manager.load_model_config(
        model_name=args.model_name,
        num_classes=args.num_classes,
        masks=args.masks,
        pretrain_weights=args.pretrain_weights
    )
    
    train_config = config_manager.load_training_config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        amp=args.amp,
        test_mode=args.test_mode,
        test_limit=args.test_limit
    )
    
    # 2. Model Creation
    model_factory = ModelFactory()
    model = model_factory.create_model(model_config)
    criterion, postprocessors = model_factory.create_criterion_and_postprocessors(model_config)
    
    # Load pretrained weights if specified
    if args.pretrain_weights:
        model_factory.load_pretrained_weights(model, args.pretrain_weights)
    
    # 3. Data Loading
    data_loader_factory = DataLoaderFactory()
    train_loader = data_loader_factory.create_train_loader(train_config)
    val_loader = data_loader_factory.create_val_loader(train_config)
    
    # 4. Optimization
    optimizer_factory = OptimizerFactory()
    optimizer = optimizer_factory.create_optimizer(model, train_config)
    
    scheduler_factory = SchedulerFactory()
    lr_scheduler = scheduler_factory.create_scheduler(optimizer, train_config)
    
    # 5. Checkpoint Management
    checkpoint_dir = Path(args.output_dir) / 'checkpoints'
    checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    # 6. Callback Setup
    callback_manager = CallbackManager()
    
    # Logging callback
    log_dir = Path(args.output_dir) / 'logs'
    callback_manager.register_callback_object(
        LoggingCallback(
            log_dir=log_dir,
            log_interval=args.log_interval
        )
    )
    
    # Checkpoint callback
    callback_manager.register_callback_object(
        ModelCheckpointCallback(
            checkpoint_manager=checkpoint_manager,
            save_interval=args.checkpoint_interval
        )
    )
    
    # TensorBoard callback (optional)
    if args.tensorboard:
        tensorboard_dir = Path(args.output_dir) / 'tensorboard'
        callback_manager.register_callback_object(
            TensorBoardCallback(log_dir=tensorboard_dir)
        )
    
    # Early stopping callback (optional)
    if args.early_stopping_patience > 0:
        callback_manager.register_callback_object(
            EarlyStoppingCallback(patience=args.early_stopping_patience)
        )
    
    # 7. Create Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        postprocessors=postprocessors,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        data_loader_factory=data_loader_factory,
        checkpoint_manager=checkpoint_manager,
        callback_manager=callback_manager,
        device=torch.device(args.device)
    )
    
    return trainer, train_config


def main():
    """Main training function using refactored components."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Setting up training components...")
    trainer, train_config = setup_components(args)
    
    print(f"Starting training for {args.epochs} epochs...")
    results = trainer.train(train_config)
    
    print("Training completed!")
    print(f"Best validation metric: {results.best_metric:.4f}")
    print(f"Final validation metric: {results.final_metric:.4f}")
    print(f"Total training time: {results.total_time:.2f} seconds")
    
    # Save final model
    final_model_path = output_path / 'final_model.pth'
    torch.save(trainer.model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()