"""
Hooks for RF-DETR-Mask training.
"""

from rfdetr.hooks.onnx_checkpoint_hook import ONNXCheckpointHook

__all__ = ["ONNXCheckpointHook"]
