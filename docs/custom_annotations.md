# Using Custom COCO Annotation Files with RF-DETR

RF-DETR now supports using custom COCO annotation files that don't follow the standard COCO directory structure. This is particularly useful when working with custom datasets that use COCO format but have different file names or directory structures.

## Parameters

The following parameters control custom annotation file loading:

- `--coco_path`: Path to the directory containing your annotation files
- `--coco_train`: Name of the training annotation file (relative to coco_path)
- `--coco_val`: Name of the validation annotation file (relative to coco_path)
- `--coco_img_path`: Path to the directory containing your images

## Example Usage

### Training with CMR Dataset

```bash
python scripts/train.py \
    --dataset_file coco \
    --coco_path /home/georgepearse/data/cmr/annotations \
    --coco_train 2025-05-15_12:38:23.077836_train_ordered.json \
    --coco_val 2025-05-15_12:38:38.270134_val_ordered.json \
    --coco_img_path /home/georgepearse/data/images \
    --output_dir output_cmr_segmentation
```

### In Python Code

```python
args = argparse.Namespace()
args.dataset_file = 'coco'
args.coco_path = '/home/georgepearse/data/cmr/annotations'
args.coco_train = '2025-05-15_12:38:23.077836_train_ordered.json'
args.coco_val = '2025-05-15_12:38:38.270134_val_ordered.json'
args.coco_img_path = '/home/georgepearse/data/images'

# Build datasets
train_dataset = build_dataset(image_set='train', args=args, resolution=640)
val_dataset = build_dataset(image_set='val', args=args, resolution=640)
```

## How It Works

The modified `build` function in `rfdetr/datasets/coco.py` now checks for custom annotation parameters:

1. If `coco_train` is provided and we're building the train set, it uses that file
2. If `coco_val` is provided and we're building the val set, it uses that file
3. If custom files aren't provided, it falls back to the standard COCO structure

The paths can be:
- Relative to `coco_path` (e.g., `my_train.json`)
- Absolute paths (e.g., `/path/to/my_train.json`)

## Standard vs Custom Structure

### Standard COCO Structure (default)
```
coco_path/
├── train2017/          # Images
├── val2017/            # Images
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Custom Structure (using parameters)
```
coco_path/                    # Set with --coco_path
├── 2025-05-15_train.json    # Set with --coco_train
└── 2025-05-15_val.json      # Set with --coco_val

images_path/                  # Set with --coco_img_path
├── image1.jpg
├── image2.jpg
└── ...
```

## Backward Compatibility

This change is fully backward compatible. If you don't provide custom annotation parameters, the system will use the standard COCO directory structure as before.