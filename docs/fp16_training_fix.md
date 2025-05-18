# Half-Precision (FP16) Training Fix

## Issue Description

When training the RF-DETR-Mask model with half-precision (FP16), we have encountered two major issues:

### Issue 1: Optimizer Error

During the second epoch, we encountered the following error in the optimizer step:

```
RuntimeError: expected dtype float for `end` but got dtype c10::Half
```

The error occurred in:
```python
scaler.step(optimizer)
  -> optimizer.step() 
    -> torch/optim/adam.py 
      -> exp_avg.lerp_(grad, 1 - device_beta1)
```

### Issue 2: cdist_cuda Error

When running with AMP and half precision, we encountered an error in the matcher:

```
RuntimeError: "cdist_cuda" not implemented for 'Half'
```

The error occurred in `rfdetr/models/matcher.py` where the L1 distance between predicted and target bounding boxes is computed:

```python
cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
```

## Root Cause Analysis

### For Optimizer Error:
The error happens because certain operations in PyTorch's Adam optimizer implementation don't fully support half-precision (float16) tensors. Specifically, the `lerp_` function expects float32 tensors but receives float16 tensors during the optimization step.

This occurs during the exponential moving average update in the Adam optimizer where the `lerp_` function is used to update momentum buffers. The error only appears after the first epoch when the optimizer has built its internal state and is attempting to update the momentum buffers.

### For cdist_cuda Error:
The PyTorch implementation of `cdist` for CUDA doesn't support half-precision inputs. This operation is used in the Hungarian matcher to compute the L1 distance between predicted and target bounding box coordinates.

## Solution

To address these issues, we've implemented two approaches:

### Fix 1: For Optimizer Error
1. **Increased epsilon value in AdamW optimizer**:
   Added `eps=1e-4` to the AdamW optimizer configuration to improve numerical stability with half-precision training.

2. **Disabled fused operations**:
   Set `fused=False` in the AdamW optimizer to prevent operations that might not be compatible with half-precision.

### Fix 2: For cdist_cuda Error (Current Implementation)
**Disable half precision training completely** by:
1. Modified `_setup_autocast_args` in `rfdetr/lightning_module.py` to:
   - Always use `torch.float32` as the default dtype
   - Disable autocast by default

2. Updated `scripts/train.py` to:
   - Set `args.use_fp16 = False`
   - Set `args.amp = False`
   - Set PyTorch Lightning's precision parameter to `"32-true"` to ensure full precision

## Implementation

### For Optimizer Error:
Changes were made to both training scripts:

1. In `scripts/train.py`:
```python
optimizer = torch.optim.AdamW(
    param_dicts, 
    lr=args.lr, 
    weight_decay=args.weight_decay, 
    fused=False,
    eps=1e-4
)
```

2. In `scripts/train_coco_segmentation.py`:
```python
optimizer = torch.optim.AdamW(
    param_dicts, 
    lr=args.lr, 
    weight_decay=args.weight_decay, 
    fused=False,
    eps=1e-4
)
```

### For cdist_cuda Error:
Instead of fixing the code to support half precision, we've decided to use full precision training as the default approach:

1. In `rfdetr/lightning_module.py`:
```python
def _setup_autocast_args(self):
    # Use full precision (float32) by default
    import torch

    # Always use float32 for main training to avoid cdist_cuda issues
    self._dtype = torch.float32

    # Disable autocast by default
    self.autocast_args = {
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        "enabled": getattr(self.args, "amp", False),
        "dtype": self._dtype,
    }
```

2. In `scripts/train.py`:
```python
# Disable half precision training by default to avoid cdist_cuda issues
args.use_fp16 = False
args.amp = False
args.fp16_eval = False

# Always use full precision
precision="32-true"
```

## Impact

These changes ensure that training can proceed without errors, though at the cost of slightly increased memory usage and potentially slower training. The matcher's `torch.cdist` operation now works correctly with float32 inputs.

## Future Work

If half-precision training is required for memory efficiency, possible solutions for the cdist_cuda issue include:

1. Implementing a custom version of the `cdist` operation that supports half precision
2. Converting only the critical operations to full precision while keeping the rest in half precision, similar to:
```python
if out_bbox.dtype == torch.float16 or tgt_bbox.dtype == torch.float16:
    cost_bbox = torch.cdist(out_bbox.float(), tgt_bbox.float(), p=1).to(out_bbox.dtype)
else:
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
```

3. Using a different matching algorithm that avoids the `cdist` operation

## Best Practices for Half-Precision Training

For future development with half-precision training:

1. **Use GradScaler**: Always use PyTorch's GradScaler when enabling AMP to help prevent underflows.
2. **Disable fused operations**: Set `fused=False` in optimizers when working with half-precision.
3. **Increase epsilon values**: Use higher epsilon values (e.g., 1e-4 instead of 1e-8) for optimizers.
4. **Convert to float32 for sensitive operations**: Cast tensors to float32 for operations that might cause numerical issues, then convert back to half-precision afterward.
5. **Consider bfloat16**: When available, bfloat16 provides more stable training than float16 while still offering memory savings.