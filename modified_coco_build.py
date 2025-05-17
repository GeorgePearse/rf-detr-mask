import contextlib
from pathlib import Path

from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.transforms import make_coco_transforms, make_coco_transforms_square_div_64


def build_custom_coco(image_set, args, resolution):
    """
    Modified build function that supports custom COCO annotation files
    through coco_train and coco_val parameters.
    """
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"

    # Set default annotation paths like the original
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

    # Make sure paths exist
    if not img_folder.exists():
        raise ValueError(f"Image folder {img_folder} does not exist")
    if not ann_file.exists():
        raise ValueError(f"Annotation file {ann_file} does not exist")

    # Create dataset with appropriate transforms
    with contextlib.suppress(Exception):
        pass

    square_resize_div_64 = getattr(args, "square_resize_div_64", False)

    # Add default multi_scale and expanded_scales if not present
    multi_scale = getattr(args, "multi_scale", False)
    expanded_scales = getattr(args, "expanded_scales", False)

    if square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(
                image_set, resolution, multi_scale=multi_scale, expanded_scales=expanded_scales
            ),
        )
    else:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(
                image_set, resolution, multi_scale=multi_scale, expanded_scales=expanded_scales
            ),
        )
    return dataset
