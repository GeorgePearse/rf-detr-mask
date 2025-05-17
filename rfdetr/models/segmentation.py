# ------------------------------------------------------------------------
# RF-DETR-MASK
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Segmentation module for RF-DETR-MASK. Adapted from:
https://github.com/facebookresearch/detr/blob/main/models/segmentation.py
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head that predicts instance masks from transformer output.

    Args:
        dim: Hidden dimension size of the transformer
        context_dim: Dimension of the image features from the backbone
        hidden_dim: Hidden dimension for the mask head
        num_groups: Number of groups for GroupNorm
    """

    def __init__(self, dim: int, context_dim: int, hidden_dim: int = 256, num_groups: int = 8):
        super().__init__()

        self.lay1 = nn.Conv2d(context_dim, hidden_dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, hidden_dim)

        self.lay2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, hidden_dim)

        self.lay3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.gn3 = nn.GroupNorm(num_groups, hidden_dim)

        self.adapter1 = nn.Conv2d(context_dim, dim, 1)
        self.adapter2 = nn.Conv2d(context_dim, dim, 1)

        self.out_lay = nn.Conv2d(dim, 1, 3, padding=1)
        self.dim = dim

        # Initialize layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, memory: Tensor, attention_map: Tensor) -> Tensor:
        """
        Args:
            x: Image features [batch_size, channels, H, W]
            memory: Transformer encoder output [batch_size, num_queries, dim]
            attention_map: Attention weights from the transformer [batch_size, num_queries, H*W]

        Returns:
            mask_logits: Predicted mask logits [batch_size, num_queries, H, W]
        """
        # Convert to float32 for stability
        x = x.float()
        memory = memory.float()
        attention_map = attention_map.float()

        # First branch
        out = self.lay1(x)
        out = self.gn1(out)
        out = F.relu(out)

        out = self.lay2(out)
        out = self.gn2(out)
        out = F.relu(out)

        out = self.lay3(out)
        out = self.gn3(out)
        out = F.relu(out)

        # Reduce channel dimensions
        x_reduced = self.adapter1(x)
        out = out + x_reduced

        # Apply attention map
        batch_size, num_queries = memory.shape[:2]
        h, w = x.shape[-2:]

        # Reshape attention map
        attention_map = attention_map.view(batch_size, num_queries, h, w)

        # Apply attention to features
        out = out.unsqueeze(1) * attention_map.unsqueeze(2)

        # Add query information
        memory_proj = memory.view(batch_size, num_queries, self.dim, 1, 1)
        memory_proj = memory_proj.expand(-1, -1, -1, h, w)
        out = out + memory_proj

        # Final convolution
        out = self.out_lay(out)

        return out


class RFDETRSegmentation(nn.Module):
    """
    Segmentation wrapper for RF-DETR that adds instance segmentation capabilities.

    This module wraps the base RF-DETR detector and adds a mask head for
    instance segmentation.
    """

    def __init__(self, detr: nn.Module, freeze_detr: bool = False, hidden_dim: int = 256):
        """
        Args:
            detr: The base RF-DETR detector model
            freeze_detr: Whether to freeze the detector weights
            hidden_dim: Hidden dimension for the mask head
        """
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.detr.parameters():
                p.requires_grad = False

        # Get dimensions from the DETR model
        context_dim = self.detr.backbone.num_channels
        query_dim = self.detr.transformer.d_model

        # Multi-head attention for bbox features
        self.bbox_attention = nn.MultiheadAttention(
            query_dim, num_heads=8, dropout=0.0, batch_first=True
        )

        # Mask prediction head
        self.mask_head = MaskHeadSmallConv(
            dim=query_dim, context_dim=context_dim, hidden_dim=hidden_dim
        )

        # FPN-style layers for multi-scale features
        self.fpn_layers = nn.ModuleList(
            [
                nn.Conv2d(context_dim, context_dim, 1),
                nn.Conv2d(context_dim, context_dim, 1),
                nn.Conv2d(context_dim, context_dim, 1),
            ]
        )

    def forward(self, samples) -> dict[str, Tensor]:
        """
        Forward pass through the segmentation model.

        Args:
            samples: NestedTensor containing images and masks

        Returns:
            Dictionary containing:
                - pred_logits: Classification scores [batch_size, num_queries, num_classes]
                - pred_boxes: Bounding box predictions [batch_size, num_queries, 4]
                - pred_masks: Mask predictions [batch_size, num_queries, H/4, W/4]
        """
        # Get detections from the base DETR model
        detr_outputs = self.detr(samples)

        # Extract features from backbone
        features, _ = self.detr.backbone(samples)

        # Get the highest resolution feature map
        # RF-DETR typically uses a feature pyramid, so we take the last (highest res) level
        src = features[-1] if isinstance(features, list) else features

        # Get transformer memory output
        mask = samples.mask if hasattr(samples, "mask") else None
        self.detr.transformer.encoder(src, src_key_padding_mask=mask)

        # Get decoder output
        hs = (
            detr_outputs["decoder_outputs"]
            if "decoder_outputs" in detr_outputs
            else detr_outputs["hs"]
        )

        # Take the output from the last decoder layer
        query_embed = hs[-1]

        # Generate attention maps
        batch_size, num_queries = query_embed.shape[:2]
        h, w = src.shape[-2:]

        # Flatten spatial dimensions for attention
        src_flat = src.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # Compute attention between queries and image features
        attention_out, attention_weights = self.bbox_attention(query_embed, src_flat, src_flat)

        # Reshape attention weights
        attention_weights = attention_weights.view(batch_size, num_queries, h, w)

        # Predict masks
        mask_logits = self.mask_head(src, query_embed, attention_weights)

        # Apply sigmoid and handle numerical stability
        mask_probs = mask_logits.sigmoid()

        # Check for inf/nan values
        if torch.isnan(mask_probs).any() or torch.isinf(mask_probs).any():
            print("Warning: inf/nan in mask predictions, clamping values")
            mask_probs = torch.clamp(mask_probs, min=0.0, max=1.0)

        # Add mask predictions to the output
        outputs = {
            "pred_logits": detr_outputs["pred_logits"],
            "pred_boxes": detr_outputs["pred_boxes"],
            "pred_masks": mask_probs,  # [B, N, H/4, W/4]
        }

        # Include auxiliary outputs if present
        if "aux_outputs" in detr_outputs:
            outputs["aux_outputs"] = detr_outputs["aux_outputs"]

        return outputs


def build_segmentation_model(model_config: dict, **kwargs) -> RFDETRSegmentation:
    """
    Build the RF-DETR segmentation model.

    Args:
        model_config: Configuration dictionary for the model
        **kwargs: Additional arguments

    Returns:
        RFDETRSegmentation model instance
    """
    # Import here to avoid circular imports
    from rfdetr.models.lwdetr import build_model

    # Build the base detector
    detr = build_model(model_config)

    # Wrap it with segmentation capabilities
    segmentation_model = RFDETRSegmentation(detr, freeze_detr=False)

    return segmentation_model
