#!/usr/bin/env python3
"""Quick test to verify training and evaluation work with fp16_eval"""

import sys
import os
from pathlib import Path

# Run a minimal training with fp16_eval to test the fix
cmd = f"""cd {Path(__file__).parent} && python scripts/train_coco_segmentation.py \\
    --masks \\
    --num_classes 2 \\
    --coco_path /home/georgepearse/data/cmr \\
    --train_anno /home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json \\
    --val_anno /home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json \\
    --image_dir /home/georgepearse/data/images \\
    --epochs 1 \\
    --lr_drop 1 \\
    --batch_size 2 \\
    --eval_batch_size 2 \\
    --num_workers 2 \\
    --output_dir ./test_output \\
    --eval_every_epoch 1 \\
    --lr 2e-4 \\
    --warmup_epochs 0 \\
    --fp16_eval \\
    --save_checkpoint_every_epoch 1 \\
    --amp
"""

print("Running quick training test with fp16_eval enabled...")
print("This will test the dtype fix during evaluation...")
print("=" * 50)

os.system(cmd)