#!/bin/bash
# Script to train RF-DETR-Mask with fixed size images using PyTorch Lightning

# Navigate to the repository root
cd "$(dirname "$0")"

# Set variables
CONFIG_PATH="configs/fixed_size_config.yaml"
OUTPUT_DIR="output_lightning_fixed_size"
FIXED_WIDTH=1232
FIXED_HEIGHT=896

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training using the fixed_size.py script with Lightning
python scripts/train_fixed_size.py \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --fixed_width $FIXED_WIDTH \
    --fixed_height $FIXED_HEIGHT

echo "Training complete. Results saved in $OUTPUT_DIR"