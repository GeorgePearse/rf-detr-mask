#!/bin/bash

# Run evaluation with test predictions
python scripts/evaluate.py \
  --checkpoint lightning_logs/version_4/checkpoints/last.ckpt \
  --coco_path /tmp/test_annotations \
  --coco_val test_val.json \
  --create_test_predictions \
  --flip_ratio 0.3 \
  --output_dir test_eval_results