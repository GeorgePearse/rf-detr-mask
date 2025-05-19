import os
import argparse
import torch
from pathlib import Path
from collections import OrderedDict

from rfdetr.models import build_model
from rfdetr.model_config import ModelConfig
from rfdetr.util.utils import clean_state_dict

def test_checkpoint_loading(checkpoint_path, verbose=True):
    """Test loading a checkpoint and identify mismatching keys."""
    print(f"Testing checkpoint loading from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} not found")
        return
    
    # Load checkpoint
    try:
        print("Loading checkpoint...")
        try:
            # First try with weights_only=False (for older checkpoints)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e1:
            print(f"First attempt failed: {e1}")
            print("Trying with weights_only=True...")
            # Try with weights_only=True as fallback
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Extract model state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("Found model state dict in checkpoint")
    else:
        print("Error: No model state dict found in checkpoint")
        return
    
    # Clean state dict if needed
    state_dict = clean_state_dict(state_dict)
    
    # Extract configuration if available
    if "args" in checkpoint:
        print("Using configuration from checkpoint args")
        # Convert args to dict for consistency
        if hasattr(checkpoint["args"], "__dict__"):
            config_dict = vars(checkpoint["args"])
        else:
            config_dict = {k: getattr(checkpoint["args"], k) for k in dir(checkpoint["args"]) 
                          if not k.startswith("_") and not callable(getattr(checkpoint["args"], k))}
    elif "config" in checkpoint:
        print("Using configuration from checkpoint config")
        config_dict = checkpoint["config"]
    else:
        # Default configuration dictionary
        print("No configuration found in checkpoint, using default config")
        config_dict = {
            "encoder": "dinov2_windowed_small",
            "out_feature_indexes": [-1],
            "projector_scale": ["P4"],
            "num_classes": 90,
            "two_stage": True,
            "hidden_dim": 256,
            "pretrained_encoder": True,
            "position_embedding": "sine",
            "dec_layers": 3,
            "vit_encoder_num_layers": 12,
            "sa_nheads": 8,
            "ca_nheads": 8,
            "dim_feedforward": 1024,
            "bbox_reparam": True,
            "aux_loss": True,
            "lite_refpoint_refine": True
        }
    
    # Set defaults for any missing critical keys
    required_keys = [
        "encoder", "out_feature_indexes", "projector_scale", "num_classes", 
        "two_stage", "hidden_dim", "pretrained_encoder", 
        "position_embedding", "dec_layers", "vit_encoder_num_layers",
        "sa_nheads", "ca_nheads", "dim_feedforward"
    ]
    
    default_values = {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [-1],
        "projector_scale": ["P4"],
        "num_classes": 90,
        "two_stage": True,
        "hidden_dim": 256,
        "pretrained_encoder": True,
        "position_embedding": "sine",
        "dec_layers": 3,
        "vit_encoder_num_layers": 12,
        "sa_nheads": 8,
        "ca_nheads": 8,
        "dim_feedforward": 1024,
        "bbox_reparam": True,
        "aux_loss": True,
        "lite_refpoint_refine": True,
        "layer_norm": True,
        "rms_norm": False
    }
    
    # Ensure required keys have values
    for key in required_keys:
        if key not in config_dict or config_dict[key] is None:
            print(f"Adding missing key '{key}' with default value: {default_values[key]}")
            config_dict[key] = default_values[key]
    
    # Print config for debugging
    print("\nConfiguration being used:")
    for key in sorted(config_dict.keys()):
        if key in required_keys:  # Only print essential keys to avoid clutter
            print(f"  {key}: {config_dict[key]}")
    
    # Build model
    try:
        print("\nBuilding model...")
        model = build_model(config_dict)
    except Exception as e:
        print(f"Error building model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check number of classes
    checkpoint_num_classes = state_dict.get("class_embed.bias", None)
    if checkpoint_num_classes is not None:
        num_classes = checkpoint_num_classes.shape[0]
        print(f"Checkpoint has {num_classes} classes")
        
        # Check if model classes match
        model_num_classes = model.class_embed.bias.shape[0]
        print(f"Model expects {model_num_classes} classes")
        
        if num_classes != model_num_classes:
            print(f"Warning: Number of classes mismatch. Checkpoint: {num_classes}, Model: {model_num_classes}")
    
    # Try loading state dict and identify mismatching keys
    print("\nTesting state_dict loading...")
    missing_keys = []
    unexpected_keys = []
    
    # Collect model and checkpoint keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    
    # Find missing and unexpected keys
    missing_keys = list(model_keys - checkpoint_keys)
    unexpected_keys = list(checkpoint_keys - model_keys)
    
    # Try loading
    try:
        if missing_keys or unexpected_keys:
            print("Warning: Strict loading would fail due to mismatched keys")
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            
            if verbose:
                print("\nMissing keys:")
                for key in sorted(missing_keys):
                    print(f"  {key}")
                print("\nUnexpected keys:")
                for key in sorted(unexpected_keys):
                    print(f"  {key}")
            
            print("\nTrying non-strict loading...")
            model.load_state_dict(state_dict, strict=False)
            print("Non-strict loading successful")
        else:
            print("All keys match, performing strict loading...")
            model.load_state_dict(state_dict, strict=True)
            print("Strict loading successful")
    except Exception as e:
        print(f"Error loading state dict: {e}")
    
    # Check for common issues like shape mismatches
    print("\nChecking for shape mismatches...")
    shape_mismatches = []
    
    for key in model_keys.intersection(checkpoint_keys):
        model_shape = model.state_dict()[key].shape
        checkpoint_shape = state_dict[key].shape
        
        if model_shape != checkpoint_shape:
            shape_mismatches.append((key, model_shape, checkpoint_shape))
    
    if shape_mismatches:
        print(f"Found {len(shape_mismatches)} shape mismatches:")
        for key, model_shape, checkpoint_shape in shape_mismatches:
            print(f"  {key}: Model {model_shape} vs Checkpoint {checkpoint_shape}")
    else:
        print("No shape mismatches found for common keys")
    
    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test checkpoint loading and identify issues")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed key information")
    args = parser.parse_args()
    
    test_checkpoint_loading(args.checkpoint, args.verbose)