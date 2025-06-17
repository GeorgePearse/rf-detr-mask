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
import torch.utils.data
import torchvision

import rfdetr.datasets.transforms as T


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


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, use_albumentations=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCoco()
        self.use_albumentations = use_albumentations

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCoco(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # Handle segmentation masks if available
        masks = None
        if anno and "segmentation" in anno[0]:
            masks = []
            for obj in anno:
                if "segmentation" in obj:
                    if isinstance(obj["segmentation"], list):
                        # Polygon format
                        rles = mask_util.frPyObjects(obj["segmentation"], h, w)
                        rle = mask_util.merge(rles)
                    else:
                        # RLE format
                        rle = obj["segmentation"]
                    mask = mask_util.decode(rle)
                    masks.append(mask)
                else:
                    # Create an empty mask if segmentation is missing
                    masks.append(np.zeros((h, w), dtype=np.uint8))

            if masks:
                masks = torch.as_tensor(np.stack(masks), dtype=torch.bool)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if masks is not None:
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # Add masks to target if available
        if masks is not None:
            target["masks"] = masks

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(
    image_set, resolution, multi_scale=False, expanded_scales=False
):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    scales = [resolution]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(resolution, expanded_scales)
        print(scales)

    if image_set == "train":
        # For windowed attention, we need square images
        print(f"[TRANSFORM DEBUG] Creating train transforms with scales: {scales}")
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.SquareResize(
                    scales
                ),  # Changed to SquareResize for windowed attention
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.SquareResize(
                    [resolution]
                ),  # Changed to SquareResize for windowed attention
                normalize,
            ]
        )
    if image_set == "val_speed":
        return T.Compose(
            [
                T.SquareResize([resolution]),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def make_coco_transforms_square_div_64(
    image_set, resolution, multi_scale=False, expanded_scales=False
):
    """ """

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    scales = [resolution]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(resolution, expanded_scales)
        print(scales)

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.SquareResize(scales),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.SquareResize([resolution]),
                normalize,
            ]
        )
    if image_set == "val_speed":
        return T.Compose(
            [
                T.SquareResize([resolution]),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def make_coco_transforms_rectangular(
    image_set, width, height, multi_scale=False, expanded_scales=False
):
    """Create transforms for rectangular (non-square) training."""
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RectangularResize(width, height),
                normalize,
            ]
        )

    if image_set == "val" or image_set == "val_speed":
        return T.Compose(
            [
                T.RectangularResize(width, height),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def make_albumentations_transforms(image_set, config_path=None, with_masks=True):
    """Create albumentations transforms from YAML config."""
    # Lazy import to avoid circular dependency
    from rfdetr.datasets.albumentations_wrapper import create_albumentations_transform

    if config_path is None:
        # Use default config based on image set
        if image_set == "train":
            config_path = (
                Path(__file__).parent.parent.parent
                / "configs"
                / "transforms"
                / "default_detection.yaml"
            )
        else:
            config_path = (
                Path(__file__).parent.parent.parent
                / "configs"
                / "transforms"
                / "default_detection.yaml"
            )

    return create_albumentations_transform(
        config_path=config_path, is_train=(image_set == "train"), with_masks=with_masks
    )


def build(image_set, args, resolution):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f"{mode}_train2017.json"),
        "val": (root / "val2017", root / "annotations" / f"{mode}_val2017.json"),
        "test": (
            root / "test2017",
            root / "annotations" / "image_info_test-dev2017.json",
        ),
    }

    # Override with custom annotation files if provided
    if hasattr(args, "coco_train") and args.coco_train and image_set == "train":
        img_folder = (
            Path(args.coco_img_path)
            if hasattr(args, "coco_img_path")
            else root / "train2017"
        )
        ann_file = (
            root / args.coco_train
            if not Path(args.coco_train).is_absolute()
            else Path(args.coco_train)
        )
    elif hasattr(args, "coco_val") and args.coco_val and image_set == "val":
        img_folder = (
            Path(args.coco_img_path)
            if hasattr(args, "coco_img_path")
            else root / "val2017"
        )
        ann_file = (
            root / args.coco_val
            if not Path(args.coco_val).is_absolute()
            else Path(args.coco_val)
        )
    else:
        # Use default paths
        img_folder, ann_file = PATHS[image_set.split("_")[0]]

    # Ensure the paths exist
    assert img_folder.exists(), f"Image folder {img_folder} does not exist"
    assert ann_file.exists(), f"Annotation file {ann_file} does not exist"

    try:
        square_resize_div_64 = args.square_resize_div_64
    except AttributeError:
        square_resize_div_64 = False

    # Check if we should use albumentations
    use_albumentations = getattr(args, "use_albumentations", False)
    albumentations_config = getattr(args, "albumentations_config", None)
    use_rectangular = getattr(args, "rectangular", False)
    rect_width = getattr(args, "rect_width", 832)
    rect_height = getattr(args, "rect_height", 640)

    if use_albumentations:
        # Use albumentations transforms
        with_masks = getattr(args, "masks", False)
        transforms = make_albumentations_transforms(
            image_set, config_path=albumentations_config, with_masks=with_masks
        )
        dataset = CocoDetection(
            img_folder, ann_file, transforms=transforms, use_albumentations=True
        )
    elif use_rectangular:
        # Use rectangular transforms
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_rectangular(
                image_set,
                rect_width,
                rect_height,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
            ),
        )
    elif square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
            ),
        )
    else:
        print(
            f"[DATASET DEBUG] Creating dataset with make_coco_transforms for {image_set}"
        )
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
            ),
        )
    return dataset


def build_roboflow(image_set, args, resolution):
    root = Path(args.dataset_dir)
    assert root.exists(), f"provided Roboflow path {root} does not exist"
    PATHS = {
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val": (root / "valid", root / "valid" / "_annotations.coco.json"),
        "test": (root / "test", root / "test" / "_annotations.coco.json"),
    }

    img_folder, ann_file = PATHS[image_set.split("_")[0]]

    try:
        square_resize_div_64 = args.square_resize_div_64
    except AttributeError:
        square_resize_div_64 = False

    # Check if we should use albumentations
    use_albumentations = getattr(args, "use_albumentations", False)
    albumentations_config = getattr(args, "albumentations_config", None)

    if use_albumentations:
        # Use albumentations transforms
        with_masks = getattr(args, "masks", False)
        transforms = make_albumentations_transforms(
            image_set, config_path=albumentations_config, with_masks=with_masks
        )
        dataset = CocoDetection(
            img_folder, ann_file, transforms=transforms, use_albumentations=True
        )
    elif square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(
                image_set, resolution, multi_scale=args.multi_scale
            ),
        )
    else:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(
                image_set, resolution, multi_scale=args.multi_scale
            ),
        )
    return dataset
