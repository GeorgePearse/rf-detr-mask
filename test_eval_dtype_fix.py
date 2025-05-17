#!/usr/bin/env python3
"""Test script to verify the dtype fix works during evaluation"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from rfdetr.util.misc import NestedTensor
from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.backbone.backbone import build_backbone
from rfdetr.models.matcher import build_matcher
from rfdetr.models.segmentation import (
    DETRsegm,
    PostProcessSegm,
    PostProcessPanoptic,
    dice_loss,
    sigmoid_focal_loss,
)
from rfdetr.config import BaseConfig


def create_test_model():
    """Create a simple test model with segmentation"""
    config = BaseConfig()
    config.num_classes = 2  # Simple 2-class test
    config.lr_vit_layer_decay = 1.0
    config.num_queries = 100
    config.with_box_refine = True
    config.two_stage = False
    config.use_enc_aux_loss = True
    config.enc_layers = 6
    config.dec_layers = 6
    config.dim_feedforward = 2048
    config.hidden_dim = 256
    config.dropout = 0.0
    config.nheads = 8
    config.position_embedding = 'sine'
    config.matcher = dict(
        cost_class=2,
        cost_bbox=5,
        cost_giou=2,
        cost_mask=2,
        cost_dice=2,
    )
    
    # Create backbone
    backbone = build_backbone(config.backbone, position_embedding=config.position_embedding, 
                            vit_layer_decay=config.lr_vit_layer_decay, 
                            from_pretrained=False)
    
    # Create matcher
    matcher = build_matcher(config)
    
    # Create DETR model
    model = LWDETR(
        backbone, matcher, num_classes=config.num_classes, num_queries=config.num_queries,
        aux_loss=config.aux_loss, with_box_refine=config.with_box_refine,
        two_stage=config.two_stage, use_enc_aux_loss=config.use_enc_aux_loss,
        enc_layers=config.enc_layers, dec_layers=config.dec_layers,
        dim_feedforward=config.dim_feedforward, hidden_dim=config.hidden_dim,
        dropout=config.dropout, activation="relu", nheads=config.nheads,
        pre_norm=config.pre_norm, batch_norm=config.batch_norm
    )
    
    # Wrap with segmentation
    model = DETRsegm(model, freeze_detr=False)
    
    return model


def test_dtype_fix():
    """Test that the dtype fix works properly during evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_test_model()
    model = model.to(device)
    
    # Test in float32 mode first
    print("Testing in float32 mode...")
    model.eval()
    model.float()
    
    # Create dummy input
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    
    images = torch.rand(batch_size, channels, height, width).to(device)
    mask = torch.zeros(batch_size, height, width, dtype=torch.bool).to(device)
    samples = NestedTensor(images, mask)
    
    # Forward pass in float32 - should work
    with torch.no_grad():
        outputs = model(samples)
    print("✓ Float32 evaluation works")
    
    # Test in half precision mode
    print("\nTesting in half precision mode...")
    model.half()
    samples.tensors = samples.tensors.half()
    
    # Forward pass in half precision - should work with our fix
    with torch.no_grad():
        outputs = model(samples)
    print("✓ Half precision evaluation works")
    
    print("\nDtype fix test passed successfully!")


if __name__ == "__main__":
    test_dtype_fix()