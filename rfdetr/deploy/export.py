# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Utility functions for model deployment
"""

import os
import random
import re
import subprocess

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import rfdetr.util.misc as utils
from rfdetr.datasets.transforms import Compose, Normalize, SquareResize, ToTensor
from rfdetr.models import build_model

# Create transforms namespace for compatibility with existing code
class T:
    Compose = Compose
    SquareResize = SquareResize
    ToTensor = ToTensor
    Normalize = Normalize


def run_command_shell(command, dry_run: bool = False) -> int:
    if dry_run:
        print("")
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} {command}")
        print("")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error output:\n{e.stderr.decode('utf-8')}")
        raise


def make_infer_image(infer_dir, shape, batch_size, device="cuda"):
    if infer_dir is None:
        dummy = np.random.randint(0, 256, (shape[0], shape[1], 3), dtype=np.uint8)
        image = Image.fromarray(dummy, mode="RGB")
    else:
        image = Image.open(infer_dir).convert("RGB")

    transforms = T.Compose(
        [
            T.SquareResize([shape[0]]),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    inps, _ = transforms(image, None)
    inps = inps.to(device)
    inps = torch.stack([inps for _ in range(batch_size)])
    return inps


def no_batch_norm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            raise ValueError("BatchNorm2d found in the model. Please remove it.")