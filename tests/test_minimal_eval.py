#!/usr/bin/env python3
"""Minimal test to check evaluation with fp16_eval works after fix"""

import torch

from rfdetr.models.transformer import MLP, gen_sineembed_for_position


def test_mlp_dtype():
    """Test that MLP and sine embedding work with mixed precision"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test MLP with mixed precision
    mlp = MLP(512, 256, 256, 2)
    mlp.to(device)

    # Test in float32
    input_float32 = torch.randn(2, 100, 512).to(device)
    output_float32 = mlp(input_float32)
    print(f"Float32 - Input dtype: {input_float32.dtype}, Output dtype: {output_float32.dtype}")

    # Test in float16
    mlp.half()
    input_float16 = torch.randn(2, 100, 512).half().to(device)
    output_float16 = mlp(input_float16)
    print(f"Float16 - Input dtype: {input_float16.dtype}, Output dtype: {output_float16.dtype}")

    # Test sine embedding with mixed precision
    print("\nTesting sine embedding generation...")

    # Float32 test
    pos_tensor_float32 = torch.randn(2, 100, 4).to(device)
    sine_embed_float32 = gen_sineembed_for_position(
        pos_tensor_float32, dim=128
    )  # Returns 512 dims for 4D input
    print(
        f"Float32 - Pos tensor dtype: {pos_tensor_float32.dtype}, Sine embed dtype: {sine_embed_float32.dtype}"
    )
    print(f"Float32 - Sine embed shape: {sine_embed_float32.shape}")

    # Float16 test (this is where the bug was)
    pos_tensor_float16 = torch.randn(2, 100, 4).half().to(device)
    sine_embed_float16 = gen_sineembed_for_position(pos_tensor_float16, dim=128)
    print(
        f"Float16 - Pos tensor dtype: {pos_tensor_float16.dtype}, Sine embed dtype: {sine_embed_float16.dtype}"
    )
    print(f"Float16 - Sine embed shape: {sine_embed_float16.shape}")

    # Test the full chain (as used in the transformer)
    print("\nTesting full chain (sine embed -> MLP)...")
    # The ref_point_head MLP expects 2 * d_model input
    mlp_test = MLP(2 * 256, 256, 256, 2)  # 512 -> 256 -> 256, matching the ref_point_head
    mlp_test.to(device).half()

    # This should work now
    pos_tensor_half = torch.randn(2, 100, 4).half().to(device)
    sine_embed_half = gen_sineembed_for_position(pos_tensor_half, dim=128)  # Will output 512 dims
    output = mlp_test(sine_embed_half)
    print(f"Success! Output dtype: {output.dtype}, shape: {output.shape}")

    print("\n✅ All dtype tests passed! The fix works correctly.")


if __name__ == "__main__":
    test_mlp_dtype()
