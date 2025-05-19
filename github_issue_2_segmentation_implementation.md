# Issue: Better align segmentation implementation with Facebook DETR

## Description
After comparing the current segmentation implementation in RF-DETR-Mask with the original Facebook DETR implementation, I've identified several areas where we could improve alignment to ensure complete compatibility and correctness.

## Findings

### 1. Loss Functions
The Facebook DETR segmentation model implements `dice_loss` and uses it for mask training, but our implementation relies on a generic loss defined in the SetCriterion class (`loss_mask`). The original implementation's `dice_loss` has numerical stability improvements that could benefit our implementation.

### 2. Panoptic Segmentation
The original Facebook implementation includes a `PostProcessPanoptic` class that supports panoptic segmentation (unified instance and semantic segmentation), which is missing from our implementation. If we want to support panoptic segmentation in the future, we should add this functionality.

### 3. FPN Feature Handling
While we have FPN-style layers defined in `RFDETRSegmentation`, they don't appear to be fully utilized in the forward pass. The original implementation processes multi-scale features more explicitly.

### 4. Segmentation Testing and Examples
We could benefit from additional tests specifically for the segmentation functionality, similar to how the original implementation provides examples and testing infrastructure.

## Proposed solution

1. Consider adopting the specialized `dice_loss` and focal loss implementations from the Facebook code for improved training stability
2. Add support for panoptic segmentation by implementing the `PostProcessPanoptic` class
3. Enhance the multi-scale feature processing in the `forward` method of `RFDETRSegmentation`
4. Add comprehensive tests and examples specifically for segmentation

## Additional context
These improvements would help ensure that our segmentation implementation fully captures the capabilities of the original DETR model while maintaining compatibility with our RF-DETR architecture.