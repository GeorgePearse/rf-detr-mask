#!/usr/bin/env python3
"""Test to verify mixed precision (fp16) evaluation works correctly"""

import unittest

import torch

from rfdetr.models.transformer import MLP, gen_sineembed_for_position


class TestMixedPrecisionEval(unittest.TestCase):
    """Test mixed precision evaluation functionality"""

    def setUp(self):
        """Set up the test environment"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_mlp_dtype(self):
        """Test that MLP handles mixed precision correctly"""
        # Test MLP with mixed precision
        mlp = MLP(512, 256, 256, 2)
        mlp.to(self.device)

        # Test in float32
        input_float32 = torch.randn(2, 100, 512).to(self.device)
        output_float32 = mlp(input_float32)
        print(f"Float32 - Input dtype: {input_float32.dtype}, Output dtype: {output_float32.dtype}")
        self.assertEqual(output_float32.dtype, torch.float32)

        # Test in float16
        mlp.half()
        input_float16 = torch.randn(2, 100, 512).half().to(self.device)
        output_float16 = mlp(input_float16)
        print(f"Float16 - Input dtype: {input_float16.dtype}, Output dtype: {output_float16.dtype}")
        self.assertEqual(output_float16.dtype, torch.float16)

    def test_sine_embedding(self):
        """Test sine embedding with mixed precision"""
        print("\nTesting sine embedding generation...")

        # Float32 test
        pos_tensor_float32 = torch.randn(2, 100, 4).to(self.device)
        sine_embed_float32 = gen_sineembed_for_position(
            pos_tensor_float32, dim=128
        )  # Returns 512 dims for 4D input
        print(
            f"Float32 - Pos tensor dtype: {pos_tensor_float32.dtype}, Sine embed dtype: {sine_embed_float32.dtype}"
        )
        print(f"Float32 - Sine embed shape: {sine_embed_float32.shape}")
        self.assertEqual(sine_embed_float32.dtype, torch.float32)
        self.assertEqual(sine_embed_float32.shape, (2, 100, 512))

        # Float16 test (this is where the bug was)
        pos_tensor_float16 = torch.randn(2, 100, 4).half().to(self.device)
        sine_embed_float16 = gen_sineembed_for_position(pos_tensor_float16, dim=128)
        print(
            f"Float16 - Pos tensor dtype: {pos_tensor_float16.dtype}, Sine embed dtype: {sine_embed_float16.dtype}"
        )
        print(f"Float16 - Sine embed shape: {sine_embed_float16.shape}")
        self.assertEqual(sine_embed_float16.dtype, torch.float16)
        self.assertEqual(sine_embed_float16.shape, (2, 100, 512))

    def test_full_chain(self):
        """Test the full chain (sine embed -> MLP) as used in the transformer"""
        print("\nTesting full chain (sine embed -> MLP)...")
        # The ref_point_head MLP expects 2 * d_model input
        mlp_test = MLP(2 * 256, 256, 256, 256)  # 512 -> 256 -> 256, matching the ref_point_head
        mlp_test.to(self.device).half()

        # This should work now
        pos_tensor_half = torch.randn(2, 100, 4).half().to(self.device)
        sine_embed_half = gen_sineembed_for_position(
            pos_tensor_half, dim=128
        )  # Will output 512 dims
        output = mlp_test(sine_embed_half)
        print(f"Success! Output dtype: {output.dtype}, shape: {output.shape}")

        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(output.shape, (2, 100, 256))

        print("\n✅ All dtype tests passed! The fix works correctly.")


if __name__ == "__main__":
    unittest.main()
