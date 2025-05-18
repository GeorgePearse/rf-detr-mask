#!/bin/bash
# Script to train RF-DETR-Mask with PyTorch Lightning using fixed-size images

# Navigate to the repository root
cd "$(dirname "$0")"

# Set variables
CONFIG_PATH="configs/fixed_size_config.yaml"
OUTPUT_DIR="output_lightning_fixed_size"
BATCH_SIZE=1  # Adjust based on GPU memory
GRAD_ACCUM_STEPS=4  # Effectively gives batch size of 4

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python scripts/train_lightning.py \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE"

echo "Training complete. Results saved in $OUTPUT_DIR"