# Summary: Custom COCO Annotation Files in RF-DETR

## What Was Found

The RF-DETR codebase originally used hardcoded paths for COCO annotations following the standard COCO directory structure. However, the training scripts already included parameters for custom annotation files (`coco_train` and `coco_val`), but these weren't being used by the dataset building function.

## What Was Modified

The `build` function in `rfdetr/datasets/coco.py` was modified to:
1. Check for custom annotation file parameters (`coco_train`, `coco_val`)
2. Use custom image path parameter (`coco_img_path`)
3. Fall back to standard COCO structure if custom parameters aren't provided

## Key Files Modified

- `/home/georgepearse/rf-detr-mask/rfdetr/datasets/coco.py`

## Example Usage

```bash
python scripts/train.py \
    --coco_path /home/georgepearse/data/cmr/annotations \
    --coco_train 2025-05-15_12:38:23.077836_train_ordered.json \
    --coco_val 2025-05-15_12:38:38.270134_val_ordered.json \
    --coco_img_path /home/georgepearse/data/images
```

## Testing

A test script was created (`test_custom_annotations.py`) that verifies the custom annotation loading works correctly. The test confirmed that:
- Custom annotation files are loaded correctly when specified
- The system falls back to standard paths when custom files aren't specified
- Both relative and absolute paths work correctly