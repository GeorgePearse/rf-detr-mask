# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
import math
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from transformers import AutoBackbone

from .dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersBackbone,
    WindowedDinov2WithRegistersConfig,
)

size_to_width = {
    "tiny": 192,
    "small": 384,
    "base": 768,
    "large": 1024,
}

size_to_config = {
    "small": "dinov2_small.json",
    "base": "dinov2_base.json",
    "large": "dinov2_large.json",
}

size_to_config_with_registers = {
    "small": "dinov2_with_registers_small.json",
    "base": "dinov2_with_registers_base.json",
    "large": "dinov2_with_registers_large.json",
}


def get_config(size, use_registers):
    config_dict = size_to_config_with_registers if use_registers else size_to_config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(current_dir, "dinov2_configs")
    config_path = os.path.join(configs_dir, config_dict[size])
    with open(config_path) as f:
        dino_config = json.load(f)
    return dino_config


class DinoV2(nn.Module):
    def __init__(
        self,
        shape=(640, 640),
        out_feature_indexes=None,
        size="base",
        use_registers=True,
        use_windowed_attn=True,
        gradient_checkpointing=False,
        load_dinov2_weights=True,
    ):
        if out_feature_indexes is None:
            out_feature_indexes = [2, 4, 5, 9]
        super().__init__()

        name = (
            f"facebook/dinov2-with-registers-{size}" if use_registers else f"facebook/dinov2-{size}"
        )

        self.shape = shape

        # Create the encoder

        if not use_windowed_attn:
            assert (
                not gradient_checkpointing
            ), "Gradient checkpointing is not supported for non-windowed attention"
            assert (
                load_dinov2_weights
            ), "Using non-windowed attention requires loading dinov2 weights from hub"
            self.encoder = AutoBackbone.from_pretrained(
                name,
                out_features=[f"stage{i}" for i in out_feature_indexes],
                return_dict=False,
            )
        else:
            window_block_indexes = set(range(out_feature_indexes[-1] + 1))
            window_block_indexes.difference_update(out_feature_indexes)
            window_block_indexes = list(window_block_indexes)

            dino_config = get_config(size, use_registers)

            dino_config["return_dict"] = False
            dino_config["out_features"] = [f"stage{i}" for i in out_feature_indexes]

            # Calculate the number of patches based on shape and patch size
            patch_size = dino_config.get("patch_size", 16)
            # Make sure resolution is divisible by patch_size
            if shape[0] % patch_size != 0 or shape[1] % patch_size != 0:
                print(
                    f"Warning: Image dimensions {shape} not divisible by patch size {patch_size}."
                )
                # Adjust to closest divisible size
                h = (shape[0] // patch_size) * patch_size
                w = (shape[1] // patch_size) * patch_size
                print(f"Adjusting to {h}x{w}")
                shape = (h, w)

            h_patches, w_patches = shape[0] // patch_size, shape[1] // patch_size

            # Choose a number of windows that divides the patches evenly
            # Try to get as close to 4 as possible, but ensure it's a divisor
            for num_windows in [4, 5, 7, 1]:
                if h_patches % num_windows == 0 and w_patches % num_windows == 0:
                    break

            if use_registers:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config,
                    num_windows=num_windows,
                    window_block_indexes=window_block_indexes,
                    gradient_checkpointing=gradient_checkpointing,
                )
            else:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config,
                    num_windows=num_windows,
                    window_block_indexes=window_block_indexes,
                    num_register_tokens=0,
                    gradient_checkpointing=gradient_checkpointing,
                )
            self.encoder = (
                WindowedDinov2WithRegistersBackbone.from_pretrained(
                    name,
                    config=windowed_dino_config,
                )
                if load_dinov2_weights
                else WindowedDinov2WithRegistersBackbone(windowed_dino_config)
            )

        self._out_feature_channels = [size_to_width[size]] * len(out_feature_indexes)
        self._export = False

    def export(self):
        if self._export:
            return
        self._export = True
        shape = self.shape

        def make_new_interpolated_pos_encoding(position_embeddings, patch_size, height, width):
            num_positions = position_embeddings.shape[1] - 1
            dim = position_embeddings.shape[-1]
            height = height // patch_size
            width = width // patch_size

            class_pos_embed = position_embeddings[:, 0]
            patch_pos_embed = position_embeddings[:, 1:]

            # Reshape and permute
            patch_pos_embed = patch_pos_embed.reshape(
                1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
            )
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

            # Use bilinear interpolation without antialias
            patch_pos_embed = torch_functional.interpolate(
                patch_pos_embed,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

            # Reshape back
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        # If the shape of self.encoder.embeddings.position_embeddings
        # matches the shape of your new tensor, use copy_:
        with torch.no_grad():
            new_positions = make_new_interpolated_pos_encoding(
                self.encoder.embeddings.position_embeddings,
                self.encoder.config.patch_size,
                shape[0],
                shape[1],
            )
        # Create a new Parameter with the new size
        old_interpolate_pos_encoding = self.encoder.embeddings.interpolate_pos_encoding

        def new_interpolate_pos_encoding(self_mod, embeddings, height, width):
            num_patches = embeddings.shape[1] - 1
            num_positions = self_mod.position_embeddings.shape[1] - 1
            if num_patches == num_positions and height == width:
                return self_mod.position_embeddings
            return old_interpolate_pos_encoding(embeddings, height, width)

        self.encoder.embeddings.position_embeddings = nn.Parameter(new_positions)
        self.encoder.embeddings.interpolate_pos_encoding = types.MethodType(
            new_interpolate_pos_encoding, self.encoder.embeddings
        )

    def forward(self, x):
        # Pad the input to make it divisible by 14
        h, w = x.shape[2], x.shape[3]
        pad_h = (14 - h % 14) % 14
        pad_w = (14 - w % 14) % 14
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

        # Additional checks for windowed attention
        if hasattr(self.encoder.config, "num_windows") and self.encoder.config.num_windows > 1:
            # Also ensure divisibility by patch_size * num_windows
            new_h, new_w = x.shape[2], x.shape[3]
            patch_size = self.encoder.config.patch_size
            num_windows = self.encoder.config.num_windows

            # Check if the number of patches is divisible by num_windows
            h_patches, w_patches = new_h // patch_size, new_w // patch_size
            if h_patches % num_windows != 0 or w_patches % num_windows != 0:
                # Adjust padding to make it divisible
                h_patches_target = ((h_patches + num_windows - 1) // num_windows) * num_windows
                w_patches_target = ((w_patches + num_windows - 1) // num_windows) * num_windows
                new_h_target = h_patches_target * patch_size
                new_w_target = w_patches_target * patch_size

                extra_pad_h = new_h_target - new_h
                extra_pad_w = new_w_target - new_w

                if extra_pad_h > 0 or extra_pad_w > 0:
                    x = torch.nn.functional.pad(x, (0, extra_pad_w, 0, extra_pad_h))

        assert (
            x.shape[2] % 14 == 0 and x.shape[3] % 14 == 0
        ), f"Dinov2 requires input shape to be divisible by 14, but got {x.shape}"
        x = self.encoder(x)
        return list(x[0])


if __name__ == "__main__":
    model = DinoV2()
    model.export()
    x = torch.randn(1, 3, 640, 640)
    print(model(x))
    for j in model(x):
        print(j.shape)
