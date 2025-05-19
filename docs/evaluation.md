# RF-DETR Evaluation

This document explains how to use the evaluation script for the RF-DETR model, including how to limit the number of validation samples.

## Basic Usage

The evaluation script is located at `scripts/evaluate.py`. It can be used to evaluate a trained RF-DETR model against a validation dataset and generate detailed metrics.

Basic usage:

```bash
python scripts/evaluate.py --checkpoint /path/to/checkpoint.pth --coco_path /path/to/annotations
```

## Command-Line Arguments

The evaluate script supports the following arguments:

### Required Arguments

- `--checkpoint`: Path to the model checkpoint file (.pth)

### Configuration Options

- `--config`: Path to a YAML configuration file
- `--output_dir`: Directory to save evaluation results (default: `eval_results`)

### Dataset Parameters

- `--dataset_file`: Dataset format (default: `coco`)
- `--coco_path`: Path to the annotations directory
- `--coco_val`: Validation annotation file name
- `--coco_img_path`: Path to the images directory

### Model Parameters

- `--num_classes`: Number of classes in the model
- `--resolution`: Input resolution for the model (default: 560)

### Evaluation Parameters

- `--batch_size`: Evaluation batch size (default: 1)
- `--num_workers`: Number of data loading workers (default: 4)
- `--device`: Device to run evaluation on (cuda/cpu) (default: cuda)
- `--fp16_eval`: Use FP16 precision for evaluation (flag)
- `--detailed`: Show detailed per-class metrics and confidence thresholds (flag)
- `--test_limit`: Limit the number of samples in the validation dataset

## Limiting Validation Samples

You can use the `--test_limit` parameter to limit the number of validation samples processed during evaluation. This is useful for quick testing or when working with limited computational resources.

Example:

```bash
python scripts/evaluate.py --checkpoint checkpoints/checkpoint_best.pth --test_limit 100
```

This will limit the evaluation to only the first 100 samples in the validation dataset.

## Output

The evaluation script produces:

1. Overall metrics (mAP, mAP50, mAP75)
2. Per-class metrics (AP50 for each class)
3. A JSON file with detailed evaluation results

Example output location:
```
eval_results/evaluation_results.json
```

## Examples

### Basic Evaluation

```bash
python scripts/evaluate.py --checkpoint output_iter_training/checkpoint_best_total.pth --coco_path /home/georgepearse/data/cmr/annotations --coco_val 2025-05-15_12:38:38.270134_val_ordered.json --coco_img_path /home/georgepearse/data/images
```

### Fast Evaluation with Limited Samples

```bash
python scripts/evaluate.py --checkpoint output_iter_training/checkpoint_best_total.pth --coco_path /home/georgepearse/data/cmr/annotations --coco_val 2025-05-15_12:38:38.270134_val_ordered.json --coco_img_path /home/georgepearse/data/images --test_limit 50 --fp16_eval
```

This will evaluate on only 50 validation samples and use FP16 precision for faster evaluation.