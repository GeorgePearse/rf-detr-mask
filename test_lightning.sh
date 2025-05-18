#!/bin/bash
# Script to test RF-DETR-Mask with PyTorch Lightning using a small number of epochs

# Navigate to the repository root
cd "$(dirname "$0")"

# Set variables
CONFIG_PATH="configs/fixed_size_config.yaml"
OUTPUT_DIR="output_lightning_test"
BATCH_SIZE=1
EPOCHS=2  # Use small number of epochs for testing

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training for just 2 epochs
python scripts/train_lightning.py \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS"

echo "Test training complete. Results saved in $OUTPUT_DIR"