#!/usr/bin/env python
"""Refactored training script using the new component architecture.

This script demonstrates how to use the refactored training components
while maintaining compatibility with the existing codebase.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfdetr.core.training.trainer import Trainer
from rfdetr.core.checkpoint import CheckpointManager
from rfdetr.core.training.callbacks import CallbackManager
from rfdetr.config import TrainConfig

# Import existing functionality for gradual migration
from scripts.train import (
    get_args_parser,
    setup_training_config,
    setup_seeds,
    setup_model_and_criterion,
    setup_data_loaders,
    get_param_dict,
)
import rfdetr.util.misc as utils


class SimpleDataLoaderFactory:
    """Simple data loader factory using existing functionality."""
    
    def __init__(self, args):
        self.args = args
        self._train_loader = None
        self._val_loader = None
        self._base_ds = None
        self._sampler_train = None
    
    def create_train_loader(self, config):
        if self._train_loader is None:
            self._setup_loaders()
        return self._train_loader
    
    def create_val_loader(self, config):
        if self._val_loader is None:
            self._setup_loaders()
        return self._val_loader
    
    def _setup_loaders(self):
        self._train_loader, self._val_loader, self._base_ds, self._sampler_train = setup_data_loaders(self.args)


class SimpleOptimizerFactory:
    """Simple optimizer factory using existing functionality."""
    
    def create_optimizer(self, model, config):
        # Use existing get_param_dict function
        param_dicts = get_param_dict(config, model)
        optimizer = torch.optim.AdamW(
            param_dicts, lr=config.lr, weight_decay=config.weight_decay
        )
        return optimizer
    
    def create_scheduler(self, optimizer, config):
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [config.lr_drop], gamma=0.1
        )


def main():
    """Main function demonstrating refactored training."""
    # Parse arguments
    parser = get_args_parser()
    parser.add_argument(
        "--use_refactored",
        action="store_true",
        help="Use the refactored training components"
    )
    args = parser.parse_args()
    
    # Setup training configuration
    args = setup_training_config(args)
    
    # Initialize distributed mode
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)
    
    device = torch.device(args.device)
    
    # Set random seeds
    setup_seeds(args)
    
    # Build model, criterion, and postprocessors
    model, criterion, postprocessors, model_without_ddp = setup_model_and_criterion(
        args, device
    )
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_parameters}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    if args.use_refactored:
        print("\n=== Using REFACTORED training components ===\n")
        
        # Create refactored components
        data_loader_factory = SimpleDataLoaderFactory(args)
        optimizer_factory = SimpleOptimizerFactory()
        
        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoints_dir,
            max_checkpoints=5,
            save_best=True,
            metric_name="val_mAP",
            metric_mode="max",
        )
        
        # Create callback manager
        callback_manager = CallbackManager()
        
        # Create trainer
        trainer = Trainer(
            model=model_without_ddp,
            criterion=criterion,
            data_loader_factory=data_loader_factory,
            optimizer_factory=optimizer_factory,
            checkpoint_manager=checkpoint_manager,
            callback_manager=callback_manager,
            device=device,
        )
        
        # Convert args to TrainConfig (temporary compatibility layer)
        # In a full implementation, we'd use a proper ConfigurationManager
        train_config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_drop=args.lr_drop,
            weight_decay=args.weight_decay,
            clip_max_norm=args.clip_max_norm,
            val_frequency=1,  # Validate every epoch
            checkpoint_frequency=10,  # Save checkpoint every 10 epochs
            early_stopping=False,  # Can be enabled if needed
        )
        
        # Add missing attributes to config for compatibility
        for key, value in vars(args).items():
            if not hasattr(train_config, key):
                setattr(train_config, key, value)
        
        # Train using refactored trainer
        results = trainer.train(
            config=train_config,
            resume_from=args.resume if args.resume else None,
        )
        
        print("\n=== Training completed ===")
        print(f"Final checkpoint: {results['final_checkpoint']}")
        print(f"Best checkpoint: {results['best_checkpoint']}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Epochs trained: {results['epochs_trained']}")
        
    else:
        print("\n=== Using LEGACY training (import from train.py) ===\n")
        # Import and run the legacy main function
        from scripts.train import main as legacy_main
        legacy_main(args)


if __name__ == "__main__":
    main()