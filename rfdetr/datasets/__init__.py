# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

from pathlib import Path

import torch.utils.data
import torchvision

from .coco import build as build_coco
from .coco import build_roboflow
from .o365 import build_o365


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args, resolution):
    # Check if we should use fixed size training dimensions
    if hasattr(args, "training_width") and hasattr(args, "training_height"):
        # Import the function only when needed to avoid circular imports
        from .fixed_size_transforms import make_training_dimensions_transforms

        if args.dataset_file == "coco":
            from .coco import CocoDetection

            # Build the dataset with custom transforms using training dimensions
            if image_set == "train":
                img_folder = (
                    Path(args.coco_img_path)
                    if hasattr(args, "coco_img_path")
                    else Path(args.coco_path) / "train2017"
                )
                ann_file = (
                    Path(args.coco_path) / args.coco_train
                    if hasattr(args, "coco_train")
                    and args.coco_train
                    and not Path(args.coco_train).is_absolute()
                    else Path(args.coco_train)
                    if hasattr(args, "coco_train") and args.coco_train
                    else Path(args.coco_path) / "annotations" / "instances_train2017.json"
                )
            elif image_set == "val":
                img_folder = (
                    Path(args.coco_img_path)
                    if hasattr(args, "coco_img_path")
                    else Path(args.coco_path) / "val2017"
                )
                ann_file = (
                    Path(args.coco_path) / args.coco_val
                    if hasattr(args, "coco_val")
                    and args.coco_val
                    and not Path(args.coco_val).is_absolute()
                    else Path(args.coco_val)
                    if hasattr(args, "coco_val") and args.coco_val
                    else Path(args.coco_path) / "annotations" / "instances_val2017.json"
                )
            else:
                raise ValueError(f"Unknown image_set: {image_set}")

            # Ensure the paths exist
            assert img_folder.exists(), f"Image folder {img_folder} does not exist"
            assert ann_file.exists(), f"Annotation file {ann_file} does not exist"

            # Get test_limit/val_limit from args if available
            test_limit = getattr(args, "test_limit", None)
            val_limit = getattr(args, "val_limit", None)
            limit = val_limit if image_set == "val" and val_limit is not None else test_limit

            # Build dataset with the fixed training dimensions
            transforms = make_training_dimensions_transforms(image_set, args)
            return CocoDetection(img_folder, ann_file, transforms=transforms, test_limit=limit)

    # Fall back to original implementation if no training dimensions or for other dataset types
    if args.dataset_file == "coco":
        return build_coco(image_set, args, resolution)
    if args.dataset_file == "o365":
        return build_o365(image_set, args, resolution)
    if args.dataset_file == "roboflow":
        return build_roboflow(image_set, args, resolution)
    raise ValueError(f"dataset {args.dataset_file} not supported")
