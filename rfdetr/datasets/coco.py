# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.utils.data
import torchvision

from rfdetr.datasets import transforms as transforms_module


def compute_multi_scale_scales(resolution, expanded_scales=False):
    if resolution == 640:
        # assume we're doing the original 640x640 and therefore patch_size is 16
        patch_size = 16
    elif resolution % (14 * 4) == 0:
        # assume we're doing some dinov2 resolution variant and therefore patch_size is 14
        patch_size = 14
    elif resolution % (16 * 4) == 0:
        # assume we're doing some other resolution and therefore patch_size is 16
        patch_size = 16
    else:
        raise ValueError(f"Resolution {resolution} is not divisible by 16*4 or 14*4")
    # round to the nearest multiple of 4*patch_size to enable both patching and windowing
    base_num_patches_per_window = resolution // (patch_size * 4)
    offsets = (
        [-3, -2, -1, 0, 1, 2, 3, 4]
        if not expanded_scales
        else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    )
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * 4 for scale in scales]
    proposed_scales = [
        scale for scale in proposed_scales if scale >= patch_size * 4
    ]  # ensure minimum image size
    return proposed_scales





def make_coco_transforms(image_set, args, multi_scale=False, expanded_scales=False):
    normalize = transforms_module.Compose(
        [
            transforms_module.ToTensor(),
            transforms_module.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Get training dimensions from args
    if hasattr(args, "training_width") and hasattr(args, "training_height"):
        training_width = args.training_width
        training_height = args.training_height
    else:
        training_width = getattr(args, "training_width", 560)
        training_height = getattr(args, "training_height", 560)
    
    max_dimension = max(training_width, training_height)
    scales = [max_dimension]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(max_dimension, expanded_scales)
        print(scales)

    if image_set == "train":
        return transforms_module.Compose(
            [
                transforms_module.RandomHorizontalFlip(),
                transforms_module.RandomSelect(
                    transforms_module.RandomResize(scales, max_size=1333),
                    transforms_module.Compose(
                        [
                            transforms_module.RandomResize([400, 500, 600]),
                            transforms_module.RandomSizeCrop(384, 600),
                            transforms_module.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return transforms_module.Compose(
            [
                transforms_module.RandomResize([max_dimension], max_size=1333),
                normalize,
            ]
        )
    if image_set == "val_speed":
        return transforms_module.Compose(
            [
                transforms_module.SquareResize([max_dimension]),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def make_coco_transforms_square_div_64(
    image_set, args, multi_scale=False, expanded_scales=False
):
    """ """

    normalize = transforms_module.Compose(
        [
            transforms_module.ToTensor(),
            transforms_module.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Get training dimensions from args
    if hasattr(args, "training_width") and hasattr(args, "training_height"):
        training_width = args.training_width
        training_height = args.training_height
    else:
        training_width = getattr(args, "training_width", 560)
        training_height = getattr(args, "training_height", 560)
    
    max_dimension = max(training_width, training_height)
    scales = [max_dimension]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(max_dimension, expanded_scales)
        print(scales)

    if image_set == "train":
        return transforms_module.Compose(
            [
                transforms_module.RandomHorizontalFlip(),
                transforms_module.RandomSelect(
                    transforms_module.SquareResize(scales),
                    transforms_module.Compose(
                        [
                            transforms_module.RandomResize([400, 500, 600]),
                            transforms_module.RandomSizeCrop(384, 600),
                            transforms_module.SquareResize(scales),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return transforms_module.Compose(
            [
                transforms_module.SquareResize([max_dimension]),
                normalize,
            ]
        )
    if image_set == "val_speed":
        return transforms_module.Compose(
            [
                transforms_module.SquareResize([max_dimension]),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    paths = {
        "train": (root / "train2017", root / "annotations" / f"{mode}_train2017.json"),
        "val": (root / "val2017", root / "annotations" / f"{mode}_val2017.json"),
        "test": (root / "test2017", root / "annotations" / "image_info_test-dev2017.json"),
    }

    # Override with custom annotation files if provided
    if hasattr(args, "coco_train") and args.coco_train and image_set == "train":
        img_folder = (
            Path(args.coco_img_path) if hasattr(args, "coco_img_path") else root / "train2017"
        )
        ann_file = (
            root / args.coco_train
            if not Path(args.coco_train).is_absolute()
            else Path(args.coco_train)
        )
    elif hasattr(args, "coco_val") and args.coco_val and image_set == "val":
        img_folder = (
            Path(args.coco_img_path) if hasattr(args, "coco_img_path") else root / "val2017"
        )
        ann_file = (
            root / args.coco_val if not Path(args.coco_val).is_absolute() else Path(args.coco_val)
        )
    else:
        # Use default paths
        img_folder, ann_file = paths[image_set.split("_")[0]]

    # Ensure the paths exist
    assert img_folder.exists(), f"Image folder {img_folder} does not exist"
    assert ann_file.exists(), f"Annotation file {ann_file} does not exist"

    square_resize_div_64 = getattr(args, "square_resize_div_64", False)

    # Get test_limit from args if available
    test_limit = getattr(args, "test_limit", None)

    if square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(
                image_set,
                args,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
            ),
            test_limit=test_limit,
        )
    else:
     
    return dataset


def build_roboflow(image_set, args):
   