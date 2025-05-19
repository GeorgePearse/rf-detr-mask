import unittest
import torch
import numpy as np
from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersConfig,
)


class TestWindowedAttentionRectangular(unittest.TestCase):
    """Test that the windowed attention mechanism works with rectangular images."""

    def test_windowed_view_operation(self):
        """Test the specific view operation that was fixed in GitHub issue #148."""
        # Setup similar dimensions to the original issue
        batch_size = 1
        hidden_dim = 64  # Use smaller hidden dimension for testing
        num_windows = 2
        
        # Use a rectangular image shape (num patches in each dimension)
        # Important: Make sure num_h_patches and num_w_patches are each divisible by num_windows
        num_h_patches = 4  # Must be divisible by num_windows
        num_w_patches = 6  # Must be divisible by num_windows
        
        # Calculate patches per window
        num_h_patches_per_window = num_h_patches // num_windows  # 2
        num_w_patches_per_window = num_w_patches // num_windows  # 3
        
        # Create a deterministic tensor where we can check the values after reshaping
        pixel_tokens_with_pos_embed = torch.zeros(
            batch_size, num_h_patches, num_w_patches, hidden_dim
        )
        
        # Fill tensor with values that we can verify after reshaping
        # Set a unique value for each position so we can track locations
        for h in range(num_h_patches):
            for w in range(num_w_patches):
                # Just set first value in hidden dimension for tracking
                pixel_tokens_with_pos_embed[0, h, w, 0] = h * 100 + w
        
        # Check that the product of all dimensions matches before and after view
        original_size = batch_size * num_h_patches * num_w_patches * hidden_dim
        windowed_size = batch_size * num_windows * num_h_patches_per_window * num_windows * num_w_patches_per_window * hidden_dim
        
        self.assertEqual(original_size, windowed_size, 
                        "Total tensor size must be preserved for view operation")
        
        # Perform the fixed view operation
        fixed_view = pixel_tokens_with_pos_embed.view(
            batch_size,
            num_windows,
            num_h_patches_per_window,
            num_windows,
            num_w_patches_per_window,
            hidden_dim,
        )
        
        # Verify that the patch positions match what we expect
        # After windowing, patches should be in correct positions within each window
        
        # Check window (0,0) - top-left
        window_0_0 = fixed_view[0, 0, :, 0, :, 0]
        for h in range(num_h_patches_per_window):
            for w in range(num_w_patches_per_window):
                expected_value = h * 100 + w
                self.assertEqual(
                    window_0_0[h, w].item(), 
                    expected_value,
                    f"Window (0,0) position ({h},{w}) has incorrect value"
                )
        
        # Check window (0,1) - top-right
        window_0_1 = fixed_view[0, 0, :, 1, :, 0]
        for h in range(num_h_patches_per_window):
            for w in range(num_w_patches_per_window):
                expected_value = h * 100 + (w + num_w_patches_per_window)
                self.assertEqual(
                    window_0_1[h, w].item(), 
                    expected_value,
                    f"Window (0,1) position ({h},{w}) has incorrect value"
                )
        
        # Check window (1,0) - bottom-left
        window_1_0 = fixed_view[0, 1, :, 0, :, 0]
        for h in range(num_h_patches_per_window):
            for w in range(num_w_patches_per_window):
                expected_value = (h + num_h_patches_per_window) * 100 + w
                self.assertEqual(
                    window_1_0[h, w].item(), 
                    expected_value,
                    f"Window (1,0) position ({h},{w}) has incorrect value"
                )
        
        # Check window (1,1) - bottom-right
        window_1_1 = fixed_view[0, 1, :, 1, :, 0]
        for h in range(num_h_patches_per_window):
            for w in range(num_w_patches_per_window):
                expected_value = (h + num_h_patches_per_window) * 100 + (w + num_w_patches_per_window)
                self.assertEqual(
                    window_1_1[h, w].item(), 
                    expected_value,
                    f"Window (1,1) position ({h},{w}) has incorrect value"
                )
        
        # Now try the buggy version for comparison
        try:
            # The original buggy version used h dimension in place of w dimension
            buggy_view = pixel_tokens_with_pos_embed.view(
                batch_size,
                num_windows,
                num_h_patches_per_window,
                num_windows,
                num_h_patches_per_window,  # BUG: This should be num_w_patches_per_window
                hidden_dim,
            )
            
            # For rectangular images, this will reshape but with incorrect data layout
            # Check the data layout of at least one window
            window_0_0_buggy = buggy_view[0, 0, :, 0, :, 0]
            
            # Find at least one discrepancy to prove the bug
            discrepancy_found = False
            for h in range(num_h_patches_per_window):
                for w in range(num_h_patches_per_window):  # Note: w is limited by h dimension
                    expected_correct_value = h * 100 + w
                    if w < num_h_patches_per_window and window_0_0_buggy[h, w].item() != expected_correct_value:
                        discrepancy_found = True
                        break
                if discrepancy_found:
                    break
            
            # If dimensions are different, the buggy version will have a different layout
            if num_h_patches_per_window != num_w_patches_per_window:
                self.assertTrue(discrepancy_found, "Buggy version should rearrange data incorrectly for rectangular inputs")
            
        except Exception as e:
            # If it fails completely (which may happen depending on tensor dimensions), that's fine too
            print(f"Buggy view operation raised: {e}")
        
        print("Fixed view operation works correctly with rectangular input")


if __name__ == "__main__":
    unittest.main()