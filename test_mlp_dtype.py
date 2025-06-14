#!/usr/bin/env python3
"""Test MLP dtype handling."""

import torch
from rfdetr.models.transformer import MLP


def test_mlp_mixed_precision():
    """Test that MLP handles mixed precision correctly."""
    print("Testing MLP with mixed precision...")

    # Create MLP
    mlp = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=3)
    mlp.eval()

    # Test with float16
    print("\n1. Testing with float16 input:")
    x_fp16 = torch.randn(2, 10, 256).half()
    with torch.no_grad():
        y_fp16 = mlp(x_fp16)
    print(f"   Input dtype: {x_fp16.dtype}, Output dtype: {y_fp16.dtype}")
    print(f"   ✓ Success! Output shape: {y_fp16.shape}")

    # Test with bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("\n2. Testing with bfloat16 input:")
        x_bf16 = torch.randn(2, 10, 256).bfloat16()
        with torch.no_grad():
            y_bf16 = mlp(x_bf16)
        print(f"   Input dtype: {x_bf16.dtype}, Output dtype: {y_bf16.dtype}")
        print(f"   ✓ Success! Output shape: {y_bf16.shape}")
    else:
        print("\n2. Skipping bfloat16 test (not supported on this device)")

    # Test with float32
    print("\n3. Testing with float32 input:")
    x_fp32 = torch.randn(2, 10, 256)
    with torch.no_grad():
        y_fp32 = mlp(x_fp32)
    print(f"   Input dtype: {x_fp32.dtype}, Output dtype: {y_fp32.dtype}")
    print(f"   ✓ Success! Output shape: {y_fp32.shape}")

    print("\n✓ All MLP dtype tests passed!")


if __name__ == "__main__":
    test_mlp_mixed_precision()
