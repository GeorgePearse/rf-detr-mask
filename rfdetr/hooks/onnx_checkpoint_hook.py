# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
PyTorch Lightning callback for exporting models to ONNX format.
"""

import datetime
import os
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

import rfdetr.util.misc as utils

# Try to import ONNX-related modules
try:
    import onnx
    import onnxsim
    from rfdetr.deploy._onnx import OnnxOptimizer
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX libraries not fully available - will only export PyTorch weights")


class ONNXCheckpointHook(Callback):
    """
    PyTorch Lightning callback that exports models to ONNX and torch formats.
    
    This callback is triggered before validation epochs, allowing for model checkpointing
    in both ONNX and PyTorch formats.
    """
    
    def __init__(
        self,
        export_dir=None,
        export_onnx=True,
        export_torch=True,
        simplify_onnx=True,
        export_frequency=1,
        input_shape=(640, 640),
        opset_version=17
    ):
        """
        Initialize the ONNXCheckpointHook.
        
        Args:
            export_dir: Directory to save exports (defaults to 'exports' under output_dir)
            export_onnx: Whether to export ONNX models
            export_torch: Whether to export PyTorch weights
            simplify_onnx: Whether to run ONNX simplification
            export_frequency: How often to export (in epochs)
            input_shape: Shape of input images (height, width)
            opset_version: ONNX opset version
        """
        super().__init__()
        self.export_dir = export_dir
        self.export_onnx = export_onnx
        self.export_torch = export_torch
        self.simplify_onnx = simplify_onnx
        self.export_frequency = export_frequency
        self.input_shape = input_shape
        self.opset_version = opset_version
    
    def _make_dummy_input(self, pl_module, batch_size=1):
        """
        Generate a dummy input for ONNX export.
        
        Args:
            pl_module: Lightning module containing the model
            batch_size: Number of samples in the batch
            
        Returns:
            A dummy input tensor with the correct shape for ONNX export
        """
        # Get resolution from args or use default
        height, width = self.input_shape
        
        # Create dummy input
        dummy = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        image = torch.from_numpy(dummy).permute(2, 0, 1).float() / 255.0
        
        # Apply normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        image = (image - mean) / std
        
        # Repeat for batch size
        images = torch.stack([image for _ in range(batch_size)])
        
        # Create nested tensor
        mask = torch.zeros((batch_size, height, width), dtype=torch.bool)
        nested_tensor = utils.NestedTensor(images, mask)
        
        return nested_tensor
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """
        Called before validation epoch starts.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module containing the model
        """
        # Skip if not at the right frequency
        current_epoch = trainer.current_epoch
        if current_epoch % self.export_frequency != 0:
            return
        
        # Skip if no exports enabled
        if not (self.export_onnx or self.export_torch):
            return
        
        # Setup export directory
        if self.export_dir is None:
            # Default to exports dir under output_dir
            output_dir = getattr(pl_module.args, "output_dir", "output")
            self.export_dir = Path(output_dir) / "exports"
        else:
            self.export_dir = Path(self.export_dir)
        
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped directory for this export
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = self.export_dir / f"epoch_{current_epoch:04d}_{timestamp}"
        export_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Exporting model to {export_path}")
        
        # Use CPU for exports to avoid CUDA errors
        device = torch.device("cpu")
        
        # Get model to export (use EMA if available)
        if hasattr(pl_module, "ema") and pl_module.ema is not None:
            model_to_export = pl_module.ema.ema
        else:
            model_to_export = pl_module.model
        
        model_to_export = model_to_export.to(device)
        model_to_export.eval()
        
        try:
            # Export PyTorch weights
            if self.export_torch:
                torch_path = export_path / "model.pth"
                torch.save({
                    "model": model_to_export.state_dict(),
                    "args": pl_module.args,
                    "epoch": current_epoch,
                    "map": getattr(pl_module, "best_map", 0.0)
                }, torch_path)
                print(f"Saved PyTorch weights to {torch_path}")
            
            # Perform ONNX export
            if self.export_onnx:
                if not ONNX_AVAILABLE:
                    print("Skipping ONNX export - ONNX libraries not available")
                else:
                    # Prepare model for export
                    if hasattr(model_to_export, "export"):
                        model_to_export.export()
                    
                    # Create dummy input tensor for export
                    dummy_input = self._make_dummy_input(pl_module).tensors.to(device)
                    
                    # Setup export parameters
                    onnx_path = export_path / "inference_model.onnx"
                    input_names = ["input"]
                    output_names = ["dets", "labels"]
                    
                    # Dynamic axes for batch size
                    dynamic_axes = {
                        "input": {0: "batch_size"},
                        "dets": {0: "batch_size"},
                        "labels": {0: "batch_size"}
                    }
                    
                    try:
                        # Export to ONNX format
                        torch.onnx.export(
                            model_to_export,
                            dummy_input,
                            onnx_path,
                            input_names=input_names,
                            output_names=output_names,
                            export_params=True,
                            keep_initializers_as_inputs=False,
                            do_constant_folding=True,
                            verbose=False,
                            opset_version=self.opset_version,
                            dynamic_axes=dynamic_axes
                        )
                        
                        print(f"Exported ONNX model to: {onnx_path}")
                        
                        # Simplify ONNX model if requested
                        if self.simplify_onnx:
                            sim_onnx_path = export_path / "inference_model.sim.onnx"
                            
                            # First use OnnxOptimizer for common optimizations
                            opt = OnnxOptimizer(str(onnx_path))
                            opt.common_opt()
                            opt.save_onnx(str(sim_onnx_path))
                            
                            # Then use onnxsim for more aggressive simplification
                            input_dict = {"input": dummy_input.detach().cpu().numpy()}
                            model_opt, check_ok = onnxsim.simplify(
                                str(onnx_path),
                                check_n=3,
                                input_data=input_dict,
                                dynamic_input_shape=False
                            )
                            
                            if check_ok:
                                onnx.save(model_opt, str(sim_onnx_path))
                                print(f"Simplified ONNX model saved to: {sim_onnx_path}")
                            else:
                                print("ONNX simplification failed - using the non-simplified version")
                    except Exception as e:
                        print(f"Error during ONNX export: {e}")
        except Exception as e:
            print(f"Error during model export: {e}")
        finally:
            # Move model back to original device
            model_to_export.to(pl_module.device)