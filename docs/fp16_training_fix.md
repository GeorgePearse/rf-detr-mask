# Half-Precision (FP16) Training Fix

## Issue Description

When training the RF-DETR-Mask model with half-precision (FP16), we encountered the following error during the second epoch:

```
RuntimeError: expected dtype float for `end` but got dtype c10::Half
```

The error occurred in the optimizer step:
```python
scaler.step(optimizer)
  -> optimizer.step() 
    -> torch/optim/adam.py 
      -> exp_avg.lerp_(grad, 1 - device_beta1)
```

## Root Cause Analysis

The error happens because certain operations in PyTorch's Adam optimizer implementation don't fully support half-precision (float16) tensors. Specifically, the `lerp_` function expects float32 tensors but receives float16 tensors during the optimization step.

This occurs during the exponential moving average update in the Adam optimizer where the `lerp_` function is used to update momentum buffers. The error only appears after the first epoch when the optimizer has built its internal state and is attempting to update the momentum buffers.

## Solution

We implemented the following fixes:

1. **Increased epsilon value in AdamW optimizer**:
   Added `eps=1e-4` to the AdamW optimizer configuration to improve numerical stability with half-precision training.

2. **Disabled fused operations**:
   Set `fused=False` in the AdamW optimizer to prevent operations that might not be compatible with half-precision.

## Implementation

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

## Additional Notes

Similar fixes were already applied to other parts of the codebase:

1. In the `HungarianMatcher` class (`rfdetr/models/matcher.py`), tensor dtype conversion is applied for the `cdist` operation:
```python
if out_bbox.dtype == torch.float16 or tgt_bbox.dtype == torch.float16:
    cost_bbox = torch.cdist(out_bbox.float(), tgt_bbox.float(), p=1).to(out_bbox.dtype)
else:
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
```

## Best Practices for Half-Precision Training

For future development with half-precision training:

1. **Use GradScaler**: Always use PyTorch's GradScaler when enabling AMP to help prevent underflows.
2. **Disable fused operations**: Set `fused=False` in optimizers when working with half-precision.
3. **Increase epsilon values**: Use higher epsilon values (e.g., 1e-4 instead of 1e-8) for optimizers.
4. **Convert to float32 for sensitive operations**: Cast tensors to float32 for operations that might cause numerical issues, then convert back to half-precision afterward.
5. **Consider bfloat16**: When available, bfloat16 provides more stable training than float16 while still offering memory savings.