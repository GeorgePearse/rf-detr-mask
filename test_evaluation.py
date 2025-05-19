#!/usr/bin/env python
"""
Script to test evaluation and confirm metrics are displayed properly.
"""

from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import CSVLogger

# Import module directly to avoid command-line argument parsing issues
from rfdetr.lightning_module import RFDETRDataModule, RFDETRLightningModule


def main():
    """Test evaluation and metrics display."""
    print("\n\n====== TESTING EVALUATION AND METRICS DISPLAY ======\n")

    # Create minimal args for testing
    class Args:
        def __init__(self):
            # Dataset parameters
            self.dataset = "coco"
            self.dataset_file = "coco"
            self.coco_path = "/home/georgepearse/data/cmr/annotations"
            self.coco_train = "2025-05-15_12:38:23.077836_train_ordered.json"
            self.coco_val = "2025-05-15_12:38:38.270134_val_ordered.json"
            self.coco_img_path = "/home/georgepearse/data/images"
            self.output_dir = "output_test_eval"

            # Model parameters
            self.masks = True
            self.num_classes = 69
            self.encoder = "dinov2_small"
            self.resolution = 336
            self.batch_size = 1
            self.train_batch_size = 1
            self.val_batch_size = 1
            self.num_workers = 4
            self.pretrain_weights = None
            self.resume = None

            # DETR parameters
            self.focal_loss = True
            self.focal_alpha = 0.25
            self.focal_gamma = 2.0
            self.num_queries = 100
            self.hidden_dim = 128
            self.position_embedding_scale = None
            self.backbone_feature_layers = ["res2", "res3", "res4", "res5"]
            self.vit_encoder_num_layers = 12
            self.num_decoder_layers = 3
            self.num_decoder_points = 4
            self.dec_layers = 3

            # Build model parameters
            self.pretrained_encoder = True
            self.window_block_indexes = []
            self.drop_path = 0.1
            self.out_feature_indexes = [2, 5, 8]
            self.projector_scale = ["P3", "P4", "P5"]
            self.use_cls_token = True
            self.position_embedding = "sine"
            self.freeze_encoder = False
            self.layer_norm = True
            self.rms_norm = False
            self.backbone_lora = False
            self.force_no_pretrain = False
            self.gradient_checkpointing = False
            self.encoder_only = False
            self.backbone_only = False

            # Transformer parameters
            self.sa_nheads = 4
            self.ca_nheads = 4
            self.dim_feedforward = 512
            self.num_feature_levels = 3
            self.dec_n_points = 4
            self.lite_refpoint_refine = True
            self.decoder_norm = "LN"

            # Additional model parameters
            self.aux_loss = True
            self.set_loss = "lw_detr"
            self.set_cost_class = 5
            self.set_cost_bbox = 2
            self.set_cost_giou = 1
            self.loss_class_coef = 4.5
            self.loss_bbox_coef = 2.0
            self.loss_giou_coef = 1.0
            self.cls_loss_coef = 4.5
            self.bbox_loss_coef = 2.0
            self.giou_loss_coef = 1.0
            self.loss_mask_coef = 1.0
            self.mask_loss_coef = 1.0
            self.loss_dice_coef = 1.0
            self.dice_loss_coef = 1.0
            self.use_varifocal_loss = False
            self.use_position_supervised_loss = False
            self.ia_bce_loss = False
            self.sum_group_losses = False
            self.num_select = 300
            self.bbox_reparam = True
            self.group_detr = 1
            self.two_stage = True
            self.no_intermittent_layers = False

            # Export parameters
            self.export_on_validation = False  # Disable exports for testing
            self.export_onnx = False
            self.export_torch = False
            self.simplify_onnx = False

            # Data augmentation parameters
            self.multi_scale = False
            self.expanded_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
            self.square_resize = True
            self.square_resize_div_64 = False

            # Misc parameters
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dropout = 0.0
            self.seed = 42
            self.test_limit = 10  # Use a small test size for faster testing

            # Lightning parameters
            self.amp = False
            self.use_fp16 = False
            self.fp16_eval = False
            self.ema_decay = 0.9997
            self.use_ema = True

    args = Args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model and datamodule
    print("Creating model and datamodule...")
    model = RFDETRLightningModule(args)
    data_module = RFDETRDataModule(args)

    # Setup data module
    print("Setting up data module...")
    data_module.setup()

    # Create simple logger
    logger = CSVLogger(save_dir=args.output_dir, name="test_eval_logs")

    # Create trainer with minimal settings
    print("Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        precision="32-true",  # Use full precision
        accelerator="auto",
        devices=1,
    )

    # Run validation
    print("\nRunning validation (evaluate)...")
    trainer.validate(model, datamodule=data_module)

    # Print metrics that were logged
    print("\nMetrics from trainer.callback_metrics:")
    if hasattr(trainer, "callback_metrics"):
        for k, v in trainer.callback_metrics.items():
            if not k.startswith("_"):  # Skip internal metrics
                print(f"  {k}: {v}")
    else:
        print("  No metrics found in trainer.callback_metrics")

    print("\n====== EVALUATION TEST COMPLETE ======\n")


if __name__ == "__main__":
    main()
