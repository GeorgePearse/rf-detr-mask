#!/bin/bash
# Script to test PyTorch Lightning training with RF-DETR-Mask

# Navigate to the repository root
cd "$(dirname "$0")"

# Set variables
OUTPUT_DIR="output_lightning_test"
EPOCHS=2
BATCH_SIZE=1
VAL_BATCH_SIZE=1

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training using the new Lightning integration
python scripts/train.py \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \
    --amp \
    --steps_per_validation 20

echo "Test training completed. Results saved in $OUTPUT_DIR"