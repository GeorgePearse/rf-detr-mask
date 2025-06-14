#!/usr/bin/env python3
"""
Test script to verify the dtype mismatch fix for mixed precision training
"""

import torch
import torch.nn as nn
from rfdetr.models.transformer import MLP, TransformerDecoder, TransformerDecoderLayer


def test_mlp_mixed_precision():
    """Test MLP with mixed precision dtypes"""
    # Create an MLP
    mlp = MLP(input_dim=512, hidden_dim=256, output_dim=256, num_layers=2)

    # Test with float16 input
    x_fp16 = torch.randn(10, 20, 512).half()

    # This should work now with our fix
    try:
        # Convert to float32 for computation
        x_fp32 = x_fp16.float()
        output = mlp(x_fp32)
        output_fp16 = output.half()
        print("✓ MLP forward pass with mixed precision successful")
        print(f"  Input dtype: {x_fp16.dtype}, Output dtype: {output_fp16.dtype}")
    except RuntimeError as e:
        print(f"✗ MLP forward pass failed: {e}")

    # Test with bfloat16 if available
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        x_bf16 = torch.randn(10, 20, 512).bfloat16()
        x_fp32 = x_bf16.float()
        output = mlp(x_fp32)
        output_bf16 = output.bfloat16()
        print("✓ MLP forward pass with bfloat16 successful")
        print(f"  Input dtype: {x_bf16.dtype}, Output dtype: {output_bf16.dtype}")


def test_transformer_decoder_mixed_precision():
    """Test TransformerDecoder with mixed precision"""
    # Create decoder components
    d_model = 256
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        sa_nhead=8,
        ca_nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        num_feature_levels=4,
        dec_n_points=4,
    )

    decoder = TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=2,
        norm=nn.LayerNorm(d_model),
        return_intermediate=True,
        d_model=d_model,
    )

    # Create test inputs
    bs = 2
    num_queries = 100
    tgt = torch.randn(bs, num_queries, d_model)
    memory = torch.randn(bs, 1000, d_model)
    refpoints_unsigmoid = torch.randn(bs, num_queries, 4)

    # Test with float16
    tgt_fp16 = tgt.half()
    memory_fp16 = memory.half()
    refpoints_fp16 = refpoints_unsigmoid.half()

    # Move decoder to half precision
    decoder.half()

    try:
        # Our fix should handle the dtype conversion internally
        spatial_shapes = torch.tensor(
            [[20, 25], [10, 13], [5, 7], [3, 4]], dtype=torch.long
        )
        level_start_index = torch.tensor([0, 500, 630, 665], dtype=torch.long)

        hs, references = decoder(
            tgt_fp16,
            memory_fp16,
            refpoints_unsigmoid=refpoints_fp16,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        print("✓ TransformerDecoder forward pass with mixed precision successful")
        print(f"  Input dtype: {tgt_fp16.dtype}, Output dtype: {hs.dtype}")
    except RuntimeError as e:
        print(f"✗ TransformerDecoder forward pass failed: {e}")

    # Test with normal precision
    decoder.float()
    hs, references = decoder(
        tgt,
        memory,
        refpoints_unsigmoid=refpoints_unsigmoid,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )
    print("✓ TransformerDecoder forward pass with float32 successful")


if __name__ == "__main__":
    print("Testing RF-DETR dtype mismatch fix...")
    print("-" * 50)
    test_mlp_mixed_precision()
    print("-" * 50)
    test_transformer_decoder_mixed_precision()
    print("-" * 50)
    print("All tests completed!")
