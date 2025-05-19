import os
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from rfdetr.datasets.coco import build


# Define a minimal args class
@dataclass
class Args:
    coco_path: str
    coco_train: str
    coco_val: str
    coco_img_path: str
    square_resize_div_56: bool
    multi_scale: bool
    expanded_scales: bool
    dataset: object = None


@dataclass
class Dataset:
    val_limit: int


def main():
    # Configure basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dataset with square_resize_div_56=True
    args = Args(
        coco_path="/home/georgepearse/data/cmr/annotations",
        coco_train="2025-05-15_12:38:23.077836_train_ordered.json",
        coco_val="2025-05-15_12:38:38.270134_val_ordered.json",
        coco_img_path="/home/georgepearse/data/images",
        square_resize_div_56=True,
        multi_scale=False,
        expanded_scales=False,
        dataset=Dataset(val_limit=1)
    )
    
    # Build dataset
    print("Building dataset with square_resize_div_56=True")
    dataset = build('val', args, 448)
    
    # Load a single image to test
    print(f"Dataset contains {len(dataset)} images")
    if len(dataset) > 0:
        img, target = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Target size: {target['size']}")
        
        # Verify that dimensions are divisible by 56
        h, w = img.shape[1:]
        print(f"Height {h} is divisible by 56: {h % 56 == 0}")
        print(f"Width {w} is divisible by 56: {w % 56 == 0}")
    
    print("Test completed successfully")


if __name__ == "__main__":
    main()