# Checkpoint Metrics Demo

This document shows what the per-class metrics output will look like when checkpoints are saved during training.

## When Metrics Are Printed

Per-class COCO mAP metrics are now printed automatically whenever a checkpoint is saved:

1. **Every 100 training steps** - Intermediate checkpoints with step-based metrics
2. **Every 10 epochs** - Periodic checkpoints with epoch-based evaluation  
3. **When a new best model is found** - Best model checkpoints with detailed metrics

## Example Output

When a checkpoint is saved, you'll see output like this:

```
Saved checkpoint at step 100

Evaluating checkpoint for per-class metrics...

================================================================================
Per-Class AP @ IoU=[0.5:0.95] | area=all | maxDets=100
================================================================================
 class_id                       class_name     AP
       14          AA_AAA Alkaline Battery  0.823
       15              9V Alkaline Battery  0.756
        4          C_D_Carbon Zinc Battery  0.745
       68              Lithium Ion Battery  0.689
       17          CR_AA_AAA Lithium Metal  0.634
       21 AA_AAA_or_2-3V_NiCd NiMH Battery  0.612
       39       AA_AAA Carbon Zinc Battery  0.598
       28 3V_3.6V_3.7V_Lithium Ion Battery  0.567
       30              C NiCd_NiMH Battery  0.534
       43   Small 3.7V Lithium Ion Battery  0.489
       23              Conveyor Belt Metal  0.456
       52          Button Zinc Air Battery  0.423
       38                  9V NiCd Battery  0.389
       61    AAAA_23A_LR1 Alkaline Battery  0.334

--------------------------------------------------------------------------------
Summary Statistics:
Mean                          : 0.575
Std                           : 0.147
Min                           : 0.334
Max                           : 0.823
Number of classes             : 14
================================================================================
```

## Benefits

1. **Track Per-Class Performance**: See which classes are improving and which need more attention
2. **Early Detection of Issues**: Identify if certain classes are not learning properly
3. **Better Model Selection**: Choose checkpoints based on performance on specific important classes
4. **Debugging Aid**: Understand model behavior across different object categories

## Implementation Details

- Uses pandas DataFrames for clean, tabular output
- Sorts classes by AP value (highest first) for easy identification of best/worst performing classes
- Includes summary statistics for quick overall assessment
- Supports both detection (bbox) and segmentation (segm) metrics
- Minimal performance impact as evaluation only runs when checkpoints are saved