# Fix for Windowed Attention with Rectangular Images

## Issue

The windowed attention mechanism in `WindowedDinov2WithRegistersEmbeddings` was causing shape errors when processing rectangular (non-square) images, particularly with the 504x560 training dimensions mentioned in [Issue #148](https://github.com/roboflow/rf-detr/issues/148).

## Root Cause

The bug was in the `view` operation for `windowed_pixel_tokens` where it incorrectly used `num_h_patches_per_window` for both the height and width dimensions:

```python
# Original buggy code
windowed_pixel_tokens = pixel_tokens_with_pos_embed.view(
    batch_size,
    num_windows,
    num_h_patches_per_window,
    num_windows,
    num_h_patches_per_window,  # Bug: Should be num_w_patches_per_window
    -1,
)
```

When processing non-square images, this led to tensor shape errors because the total number of elements didn't match between the input and the reshaped tensor.

## Fix

The fix was to correctly use `num_w_patches_per_window` for the width dimension:

```python
# Fixed code
windowed_pixel_tokens = pixel_tokens_with_pos_embed.view(
    batch_size,
    num_windows,
    num_h_patches_per_window,
    num_windows,
    num_w_patches_per_window,  # Fixed: Now correctly using width patches
    -1,
)
```

## Validation

The fix was validated with a test case in `tests/test_windowed_attn_rectangular.py` that verifies:

1. The fixed view operation correctly reshapes tensors for rectangular inputs
2. The patch positions in the reshaped tensor maintain spatial relationships

This fix enables RF-DETR to properly process rectangular images, including the 504x560 dimensions that were previously problematic.