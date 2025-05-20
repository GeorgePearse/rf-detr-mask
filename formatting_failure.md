rfdetr/deploy/_onnx/optimizer.py:388:9: F841 Local variable `n_layer_norm_plugin` is assigned to but never used
    |
387 |     def insert_layernorm_plugin(self):
388 |         n_layer_norm_plugin = 0
    |         ^^^^^^^^^^^^^^^^^^^ F841
389 |         for node in self.graph.nodes:
390 |             if (
    |
    = help: Remove assignment to unused variable `n_layer_norm_plugin`

rfdetr/deploy/_onnx/optimizer.py:405:17: N806 Variable `inputTensor` in function should be lowercase
    |
403 |                 and len(node.o().o(0).o().o().o().o().o().inputs[1].values.shape) == 1
404 |             ):
405 |                 inputTensor = (
    |                 ^^^^^^^^^^^ N806
406 |                     node.inputs[0] if node.i().op == "Add" else node.i().inputs[0]
407 |                 )  # CLIP or UNet and VAE
    |

rfdetr/deploy/_onnx/optimizer.py:409:17: N806 Variable `gammaNode` in function should be lowercase
    |
407 |                 )  # CLIP or UNet and VAE
408 |
409 |                 gammaNode = node.o().o().o().o().o().o().o()
    |                 ^^^^^^^^^ N806
410 |                 index = [isinstance(i, gs.ir.tensor.Constant) for i in gammaNode.inputs].index(True)
411 |                 gamma = np.array(
    |

rfdetr/deploy/_onnx/optimizer.py:414:17: N806 Variable `constantGamma` in function should be lowercase
    |
412 |                     deepcopy(gammaNode.inputs[index].values.tolist()), dtype=np.float32
413 |                 )
414 |                 constantGamma = gs.Constant(
    |                 ^^^^^^^^^^^^^ N806
415 |                     "LayerNormGamma-" + str(nLayerNormPlugin),
416 |                     np.ascontiguousarray(gamma.reshape(-1)),
    |

rfdetr/deploy/_onnx/optimizer.py:415:45: F821 Undefined name `nLayerNormPlugin`
    |
413 |                 )
414 |                 constantGamma = gs.Constant(
415 |                     "LayerNormGamma-" + str(nLayerNormPlugin),
    |                                             ^^^^^^^^^^^^^^^^ F821
416 |                     np.ascontiguousarray(gamma.reshape(-1)),
417 |                 )  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!
    |

rfdetr/deploy/_onnx/optimizer.py:419:17: N806 Variable `betaNode` in function should be lowercase
    |
417 |                 )  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!
418 |
419 |                 betaNode = gammaNode.o()
    |                 ^^^^^^^^ N806
420 |                 index = [isinstance(i, gs.ir.tensor.Constant) for i in betaNode.inputs].index(True)
421 |                 beta = np.array(deepcopy(betaNode.inputs[index].values.tolist()), dtype=np.float32)
    |

rfdetr/deploy/_onnx/optimizer.py:422:17: N806 Variable `constantBeta` in function should be lowercase
    |
420 |                 index = [isinstance(i, gs.ir.tensor.Constant) for i in betaNode.inputs].index(True)
421 |                 beta = np.array(deepcopy(betaNode.inputs[index].values.tolist()), dtype=np.float32)
422 |                 constantBeta = gs.Constant(
    |                 ^^^^^^^^^^^^ N806
423 |                     "LayerNormBeta-" + str(nLayerNormPlugin), np.ascontiguousarray(beta.reshape(-1))
424 |                 )
    |

rfdetr/deploy/_onnx/optimizer.py:423:44: F821 Undefined name `nLayerNormPlugin`
    |
421 |                 beta = np.array(deepcopy(betaNode.inputs[index].values.tolist()), dtype=np.float32)
422 |                 constantBeta = gs.Constant(
423 |                     "LayerNormBeta-" + str(nLayerNormPlugin), np.ascontiguousarray(beta.reshape(-1))
    |                                            ^^^^^^^^^^^^^^^^ F821
424 |                 )
    |

rfdetr/deploy/_onnx/optimizer.py:426:17: N806 Variable `inputList` in function should be lowercase
    |
424 |                 )
425 |
426 |                 inputList = [inputTensor, constantGamma, constantBeta]
    |                 ^^^^^^^^^ N806
427 |                 layerNormV = gs.Variable(
428 |                     "LayerNormV-" + str(nLayerNormPlugin), np.dtype(np.float32), inputTensor.shape
    |

rfdetr/deploy/_onnx/optimizer.py:427:17: N806 Variable `layerNormV` in function should be lowercase
    |
426 |                 inputList = [inputTensor, constantGamma, constantBeta]
427 |                 layerNormV = gs.Variable(
    |                 ^^^^^^^^^^ N806
428 |                     "LayerNormV-" + str(nLayerNormPlugin), np.dtype(np.float32), inputTensor.shape
429 |                 )
    |

rfdetr/deploy/_onnx/optimizer.py:428:41: F821 Undefined name `nLayerNormPlugin`
    |
426 |                 inputList = [inputTensor, constantGamma, constantBeta]
427 |                 layerNormV = gs.Variable(
428 |                     "LayerNormV-" + str(nLayerNormPlugin), np.dtype(np.float32), inputTensor.shape
    |                                         ^^^^^^^^^^^^^^^^ F821
429 |                 )
430 |                 layerNormN = gs.Node(
    |

rfdetr/deploy/_onnx/optimizer.py:430:17: N806 Variable `layerNormN` in function should be lowercase
    |
428 |                     "LayerNormV-" + str(nLayerNormPlugin), np.dtype(np.float32), inputTensor.shape
429 |                 )
430 |                 layerNormN = gs.Node(
    |                 ^^^^^^^^^^ N806
431 |                     "LayerNorm",
432 |                     "LayerNormN-" + str(nLayerNormPlugin),
    |

rfdetr/deploy/_onnx/optimizer.py:432:41: F821 Undefined name `nLayerNormPlugin`
    |
430 |                 layerNormN = gs.Node(
431 |                     "LayerNorm",
432 |                     "LayerNormN-" + str(nLayerNormPlugin),
    |                                         ^^^^^^^^^^^^^^^^ F821
433 |                     inputs=inputList,
434 |                     attrs=OrderedDict([("epsilon", 1.0e-5)]),
    |

rfdetr/deploy/_onnx/optimizer.py:438:17: N806 Variable `nLayerNormPlugin` in function should be lowercase
    |
436 |                 )
437 |                 self.graph.nodes.append(layerNormN)
438 |                 nLayerNormPlugin += 1
    |                 ^^^^^^^^^^^^^^^^ N806
439 |
440 |                 if betaNode.outputs[0] in self.graph.outputs:
    |

rfdetr/deploy/_onnx/optimizer.py:438:17: F821 Undefined name `nLayerNormPlugin`
    |
436 |                 )
437 |                 self.graph.nodes.append(layerNormN)
438 |                 nLayerNormPlugin += 1
    |                 ^^^^^^^^^^^^^^^^ F821
439 |
440 |                 if betaNode.outputs[0] in self.graph.outputs:
    |

rfdetr/deploy/_onnx/optimizer.py:444:21: N806 Variable `lastNode` in function should be lowercase
    |
442 |                     self.graph.outputs[index] = layerNormV
443 |                 else:
444 |                     lastNode = betaNode.o() if betaNode.o().op == "Cast" else betaNode
    |                     ^^^^^^^^ N806
445 |                     for subNode in self.graph.nodes:
446 |                         if lastNode.outputs[0] in subNode.inputs:
    |

rfdetr/deploy/_onnx/optimizer.py:445:25: N806 Variable `subNode` in function should be lowercase
    |
443 |                 else:
444 |                     lastNode = betaNode.o() if betaNode.o().op == "Cast" else betaNode
445 |                     for subNode in self.graph.nodes:
    |                         ^^^^^^^ N806
446 |                         if lastNode.outputs[0] in subNode.inputs:
447 |                             index = subNode.inputs.index(lastNode.outputs[0])
    |

rfdetr/deploy/_onnx/optimizer.py:460:9: N806 Variable `C` in function should be lowercase
    |
458 |         weights_v = node_v.inputs[1].values
459 |         # Input number of channels to K and V
460 |         C = weights_k.shape[0]
    |         ^ N806
461 |         # Number of heads
462 |         H = heads
    |

rfdetr/deploy/_onnx/optimizer.py:462:9: N806 Variable `H` in function should be lowercase
    |
460 |         C = weights_k.shape[0]
461 |         # Number of heads
462 |         H = heads
    |         ^ N806
463 |         # Dimension per head
464 |         D = weights_k.shape[1] // H
    |

rfdetr/deploy/_onnx/optimizer.py:464:9: N806 Variable `D` in function should be lowercase
    |
462 |         H = heads
463 |         # Dimension per head
464 |         D = weights_k.shape[1] // H
    |         ^ N806
465 |
466 |         # Concat and interleave weights such that the output of fused KV GEMM has [b, s_kv, h, 2, d] shape
    |

rfdetr/deploy/_onnx/optimizer.py:581:9: N806 Variable `C` in function should be lowercase
    |
580 |         # Input number of channels to Q, K and V
581 |         C = weights_k.shape[0]
    |         ^ N806
582 |         # Number of heads
583 |         H = heads
    |

rfdetr/deploy/_onnx/optimizer.py:583:9: N806 Variable `H` in function should be lowercase
    |
581 |         C = weights_k.shape[0]
582 |         # Number of heads
583 |         H = heads
    |         ^ N806
584 |         # Hidden dimension per head
585 |         D = weights_k.shape[1] // H
    |

rfdetr/deploy/_onnx/optimizer.py:585:9: N806 Variable `D` in function should be lowercase
    |
583 |         H = heads
584 |         # Hidden dimension per head
585 |         D = weights_k.shape[1] // H
    |         ^ N806
586 |
587 |         # Concat and interleave weights such that the output of fused QKV GEMM has [b, s, h, 3, d] shape
    |

rfdetr/deploy/benchmark.py:216:1: E402 Module level import not at top of file
    |
216 | import torchvision.transforms.functional as F
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ E402
217 |
218 | class ToTensor:
    |

rfdetr/deploy/benchmark.py:216:1: I001 [*] Import block is un-sorted or un-formatted
    |
216 | import torchvision.transforms.functional as F
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ I001
217 |
218 | class ToTensor:
    |
    = help: Organize imports

rfdetr/deploy/benchmark.py:216:8: N812 Lowercase `functional` imported as non-lowercase `F`
    |
216 | import torchvision.transforms.functional as F
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ N812
217 |
218 | class ToTensor:
    |

rfdetr/deploy/benchmark.py:533:9: N806 Variable `EXPLICIT_BATCH` in function should be lowercase
    |
531 |         http://gitlab.baidu.com/paddle-inference/benchmark/blob/main/backend_trt.py#L57
532 |         """
533 |         EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    |         ^^^^^^^^^^^^^^ N806
534 |         with (
535 |             trt.Builder(self.logger) as builder,
    |

rfdetr/deploy/export.py:14:1: I001 [*] Import block is un-sorted or un-formatted
   |
12 |   """
13 |
14 | / import os
15 | | import random
16 | | import re
17 | | import subprocess
18 | |
19 | | import numpy as np
20 | | import onnx
21 | | import onnxsim
22 | | import torch
23 | | import torch.nn as nn
24 | | from PIL import Image
25 | |
26 | | import rfdetr.util.misc as utils
27 | | from rfdetr.datasets.transforms import Compose, Normalize, SquareResize, ToTensor
28 | | from rfdetr.deploy._onnx import OnnxOptimizer
29 | | from rfdetr.models import build_model
   | |_____________________________________^ I001
30 |
31 |   # Create transforms namespace for compatibility with existing code
   |
   = help: Organize imports

rfdetr/detr.py:17:8: N812 Lowercase `functional` imported as non-lowercase `F`
   |
15 | import supervision as sv
16 | import torch
17 | import torchvision.transforms.functional as F
   |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ N812
18 | from PIL import Image
   |

rfdetr/engine.py:37:1: E402 Module level import not at top of file
   |
36 | # Import AMP utilities
37 | from torch.amp import GradScaler, autocast
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ E402
   |

rfdetr/engine.py:48:9: F841 Local variable `original_test_mode` is assigned to but never used
   |
46 |     with torch.no_grad():
47 |         # If in test_mode, set val_limit_test_mode for quicker validation
48 |         original_test_mode = getattr(args, "test_mode", False)
   |         ^^^^^^^^^^^^^^^^^^ F841
49 |         original_val_limit = getattr(args, "val_limit", None)
   |
   = help: Remove assignment to unused variable `original_test_mode`

rfdetr/engine.py:49:9: F841 Local variable `original_val_limit` is assigned to but never used
   |
47 |         # If in test_mode, set val_limit_test_mode for quicker validation
48 |         original_test_mode = getattr(args, "test_mode", False)
49 |         original_val_limit = getattr(args, "val_limit", None)
   |         ^^^^^^^^^^^^^^^^^^ F841
50 |         
51 |         if hasattr(args, "test_mode") and args.test_mode:
   |
   = help: Remove assignment to unused variable `original_val_limit`

rfdetr/engine.py:50:1: W293 [*] Blank line contains whitespace
   |
48 |         original_test_mode = getattr(args, "test_mode", False)
49 |         original_val_limit = getattr(args, "val_limit", None)
50 |         
   | ^^^^^^^^ W293
51 |         if hasattr(args, "test_mode") and args.test_mode:
52 |             logger.info(f"Using test mode with validation limit of {args.val_limit_test_mode}")
   |
   = help: Remove whitespace from blank line

rfdetr/engine.py:53:1: W293 [*] Blank line contains whitespace
   |
51 |         if hasattr(args, "test_mode") and args.test_mode:
52 |             logger.info(f"Using test mode with validation limit of {args.val_limit_test_mode}")
53 |         
   | ^^^^^^^^ W293
54 |         eval_stats, coco_evaluator = evaluate(
55 |             model,
   |
   = help: Remove whitespace from blank line

rfdetr/engine.py:63:1: W293 [*] Blank line contains whitespace
   |
61 |             args=args,
62 |         )
63 |         
   | ^^^^^^^^ W293
64 |         logger.info(
65 |             f"Step {step_counter} evaluation: mAP={eval_stats['coco_eval_bbox'][0]:.4f}"
   |
   = help: Remove whitespace from blank line

rfdetr/engine.py:82:5: C901 `train_one_epoch` is too complex (14 > 10)
   |
82 | def train_one_epoch(
   |     ^^^^^^^^^^^^^^^ C901
83 |     model: torch.nn.Module,
84 |     criterion: torch.nn.Module,
   |

rfdetr/engine.py:142:33: W291 [*] Trailing whitespace
    |
140 |             and base_ds is not None
141 |             and postprocessors is not None
142 |             and step_counter > 0 
    |                                 ^ W291
143 |             and step_counter % eval_freq == 0
144 |         ):
    |
    = help: Remove trailing whitespace

rfdetr/engine.py:244:5: C901 `evaluate` is too complex (19 > 10)
    |
244 | def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args=None):
    |     ^^^^^^^^ C901
245 |     """Evaluate model on validation dataset.
    |

rfdetr/engine.py:246:1: W293 Blank line contains whitespace
    |
244 | def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args=None):
245 |     """Evaluate model on validation dataset.
246 |     
    | ^^^^ W293
247 |     If args.test_mode is True, the validation will be limited to args.val_limit_test_mode samples.
248 |     """
    |
    = help: Remove whitespace from blank line

rfdetr/engine.py:260:1: W293 [*] Blank line contains whitespace
    |
258 |     iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors)
259 |     coco_evaluator = CocoEvaluator(base_ds, iou_types)
260 |     
    | ^^^^ W293
261 |     # If test_mode is enabled, limit evaluation to specified number of samples
262 |     max_samples = None
    |
    = help: Remove whitespace from blank line

rfdetr/engine.py:266:1: W293 [*] Blank line contains whitespace
    |
264 |         max_samples = args.val_limit_test_mode
265 |         logger.info(f"Test mode enabled: limiting validation to {max_samples} samples")
266 |     
    | ^^^^ W293
267 |     sample_count = 0
    |
    = help: Remove whitespace from blank line

rfdetr/engine.py:274:1: W293 [*] Blank line contains whitespace
    |
272 |             logger.info(f"Evaluation stopped at {sample_count} samples due to test_mode")
273 |             break
274 |         
    | ^^^^^^^^ W293
275 |         # Increment the sample counter
276 |         sample_count += len(targets)
    |
    = help: Remove whitespace from blank line

rfdetr/fabric_module.py:13:1: I001 [*] Import block is un-sorted or un-formatted
   |
11 |   """
12 |
13 | / import datetime
14 | | import math
15 | | import os
16 | | from pathlib import Path
17 | |
18 | | import torch
19 | | from lightning.fabric import Fabric
20 | | from lightning.fabric.strategies import DDPStrategy
21 | | from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
22 | |
23 | | # Import autocast for mixed precision training
24 | | from torch.amp import autocast  # GradScaler is not directly used
25 | |
26 | | import rfdetr.util.misc as utils
27 | | from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
28 | | from rfdetr.datasets.coco_eval import CocoEvaluator
29 | | from rfdetr.models import build_criterion_and_postprocessors, build_model
30 | | from rfdetr.util.get_param_dicts import get_param_dict
31 | | from rfdetr.util.utils import ModelEma
   | |______________________________________^ I001
   |
   = help: Organize imports

rfdetr/fabric_module.py:24:23: F401 [*] `torch.amp.autocast` imported but unused
   |
23 | # Import autocast for mixed precision training
24 | from torch.amp import autocast  # GradScaler is not directly used
   |                       ^^^^^^^^ F401
25 |
26 | import rfdetr.util.misc as utils
   |
   = help: Remove unused import: `torch.amp.autocast`

rfdetr/fabric_module.py:94:13: F841 Local variable `use_ema` is assigned to but never used
   |
92 |             # Dictionary access
93 |             self.ema_decay = self.config.get("ema_decay", None)
94 |             use_ema = self.config.get("use_ema", True)
   |             ^^^^^^^ F841
95 |         else:
96 |             # Object attribute access
   |
   = help: Remove assignment to unused variable `use_ema`

rfdetr/fabric_module.py:302:9: C901 `validation_step` is too complex (11 > 10)
    |
300 |         return metrics
301 |
302 |     def validation_step(self, batch):
    |         ^^^^^^^^^^^^^^^ C901
303 |         """Execute a single validation step.
    |

rfdetr/fabric_module.py:381:9: C901 `export_model` is too complex (12 > 10)
    |
379 |         return {"metrics": val_metrics, "results": res, "targets": targets}
380 |
381 |     def export_model(self, epoch):
    |         ^^^^^^^^^^^^ C901
382 |         """Export model to ONNX and save PyTorch weights.
    |

rfdetr/fabric_module.py:575:5: C901 `train_with_fabric` is too complex (20 > 10)
    |
575 | def train_with_fabric(
    |     ^^^^^^^^^^^^^^^^^ C901
576 |     config, output_dir=None, callbacks=None, precision="32-true", devices=1, accelerator="auto"
577 | ):
    |

rfdetr/hooks/onnx_checkpoint_hook.py:104:9: C901 `on_validation_epoch_start` is too complex (18 > 10)
    |
102 |         return nested_tensor
103 |
104 |     def on_validation_epoch_start(self, trainer, pl_module):
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^ C901
105 |         """
106 |         Called before validation epoch starts.
    |

rfdetr/lightning_module.py:235:9: C901 `validation_step` is too complex (12 > 10)
    |
233 |         return losses
234 |
235 |     def validation_step(self, batch, batch_idx):
    |         ^^^^^^^^^^^^^^^ C901
236 |         """Validation step logic."""
237 |         try:
    |

rfdetr/lightning_module.py:488:9: C901 `on_validation_epoch_end` is too complex (17 > 10)
    |
486 |                 print(f"Error updating COCO evaluator: {e}")
487 |
488 |     def on_validation_epoch_end(self):
    |         ^^^^^^^^^^^^^^^^^^^^^^^ C901
489 |         """Process validation epoch results."""
490 |         # Default metric values
    |

rfdetr/lightning_module.py:503:1: W293 [*] Blank line contains whitespace
    |
501 |                 # Check if we have enough data to evaluate
502 |                 eval_imgs_valid = False
503 |                 
    | ^^^^^^^^^^^^^^^^ W293
504 |                 if hasattr(self.coco_evaluator, "eval_imgs") and self.coco_evaluator.eval_imgs:
505 |                     for iou_type, imgs in self.coco_evaluator.eval_imgs.items():
    |
    = help: Remove whitespace from blank line

rfdetr/lightning_module.py:505:25: B007 Loop control variable `iou_type` not used within loop body
    |
504 |                 if hasattr(self.coco_evaluator, "eval_imgs") and self.coco_evaluator.eval_imgs:
505 |                     for iou_type, imgs in self.coco_evaluator.eval_imgs.items():
    |                         ^^^^^^^^ B007
506 |                         if isinstance(imgs, list) and len(imgs) > 0:
507 |                             eval_imgs_valid = True
    |
    = help: Rename unused `iou_type` to `_iou_type`

rfdetr/lightning_module.py:506:25: SIM114 [*] Combine `if` branches using logical `or` operator
    |
504 |                   if hasattr(self.coco_evaluator, "eval_imgs") and self.coco_evaluator.eval_imgs:
505 |                       for iou_type, imgs in self.coco_evaluator.eval_imgs.items():
506 | /                         if isinstance(imgs, list) and len(imgs) > 0:
507 | |                             eval_imgs_valid = True
508 | |                             break
509 | |                         elif isinstance(imgs, np.ndarray) and imgs.size > 0:
510 | |                             eval_imgs_valid = True
511 | |                             break
    | |_________________________________^ SIM114
512 |                   
513 |                   if eval_imgs_valid:
    |
    = help: Combine `if` branches

rfdetr/lightning_module.py:512:1: W293 [*] Blank line contains whitespace
    |
510 |                             eval_imgs_valid = True
511 |                             break
512 |                 
    | ^^^^^^^^^^^^^^^^ W293
513 |                 if eval_imgs_valid:
514 |                     # Synchronize if distributed
    |
    = help: Remove whitespace from blank line

rfdetr/lightning_module.py:532:1: W293 [*] Blank line contains whitespace
    |
530 |                         and hasattr(self.coco_evaluator.coco_eval["bbox"], "stats")
531 |                     )
532 |                     
    | ^^^^^^^^^^^^^^^^^^^^ W293
533 |                     if bbox_valid:
534 |                         stats = self.coco_evaluator.coco_eval["bbox"].stats
    |
    = help: Remove whitespace from blank line

rfdetr/lightning_module.py:542:1: W293 [*] Blank line contains whitespace
    |
540 | …                     if map_value > self.best_map:
541 | …                         self.best_map = map_value
542 | …                         
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ W293
543 | …                     # Update the logged value with the computed one
544 | …                     self.log("val/mAP", map_value, on_step=False, on_epoch=True, sync_dist=True)
    |
    = help: Remove whitespace from blank line

rfdetr/lightning_module.py:554:1: W293 [*] Blank line contains whitespace
    |
552 |                         and hasattr(self.coco_evaluator.coco_eval["segm"], "stats")
553 |                     )
554 |                     
    | ^^^^^^^^^^^^^^^^^^^^ W293
555 |                     if segm_valid:
556 |                         stats = self.coco_evaluator.coco_eval["segm"].stats
    |
    = help: Remove whitespace from blank line

rfdetr/lightning_module.py:568:1: W293 [*] Blank line contains whitespace
    |
566 |         else:
567 |             print("No COCO evaluator available. Using default metrics.")
568 |             
    | ^^^^^^^^^^^^ W293
569 |         # Log any additional metrics that might be useful
570 |         if len(self.val_metrics) > 0:
    |
    = help: Remove whitespace from blank line

rfdetr/lightning_module.py:575:9: C901 `configure_optimizers` is too complex (14 > 10)
    |
573 |             self.log("val/avg_loss", avg_loss, on_step=False, on_epoch=True, sync_dist=True)
574 |
575 |     def configure_optimizers(self):
    |         ^^^^^^^^^^^^^^^^^^^^ C901
576 |         """Configure optimizers and learning rate scheduler for iteration-based training."""
    |

rfdetr/main.py:539:9: F841 Local variable `start_time` is assigned to but never used
    |
538 |         print("Start training")
539 |         start_time = time.time()
    |         ^^^^^^^^^^ F841
540 |
541 |         # Track best metrics
    |
    = help: Remove assignment to unused variable `start_time`

rfdetr/model_config.py:11:1: UP035 `typing.List` is deprecated, use `list` instead
   |
 9 | """
10 |
11 | from typing import Any, List, Literal, Optional
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP035
12 | # Use modern type annotation style
13 | from typing_extensions import ClassVar
   |

rfdetr/model_config.py:11:1: I001 [*] Import block is un-sorted or un-formatted
   |
 9 |   """
10 |
11 | / from typing import Any, List, Literal, Optional
12 | | # Use modern type annotation style
13 | | from typing_extensions import ClassVar
14 | |
15 | | import torch
16 | | from pydantic import BaseModel, Field, field_validator
   | |______________________________________________________^ I001
17 |
18 |   DEVICE = (
   |
   = help: Organize imports

rfdetr/model_config.py:13:1: UP035 [*] Import from `typing` instead: `ClassVar`
   |
11 | from typing import Any, List, Literal, Optional
12 | # Use modern type annotation style
13 | from typing_extensions import ClassVar
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP035
14 |
15 | import torch
   |
   = help: Import from `typing`

rfdetr/model_config.py:13:31: F401 [*] `typing_extensions.ClassVar` imported but unused
   |
11 | from typing import Any, List, Literal, Optional
12 | # Use modern type annotation style
13 | from typing_extensions import ClassVar
   |                               ^^^^^^^^ F401
14 |
15 | import torch
   |
   = help: Remove unused import: `typing_extensions.ClassVar`

rfdetr/model_config.py:103:36: UP006 Use `list` instead of `List` for type annotation
    |
101 |     vit_encoder_num_layers: int = Field(default=12, gt=0)
102 |     pretrained_encoder: bool = True
103 |     window_block_indexes: Optional[List[int]] = None
    |                                    ^^^^ UP006
104 |     drop_path: float = Field(default=0.0, ge=0.0, le=1.0)
    |
    = help: Replace with `list`

rfdetr/models/attention.py:96:32: RUF012 Mutable class attributes should be annotated with `typing.ClassVar`
   |
94 |     """
95 |
96 |     __constants__: list[str] = ["batch_first"]
   |                                ^^^^^^^^^^^^^^^ RUF012
97 |     bias_k: Optional[torch.Tensor]
98 |     bias_v: Optional[torch.Tensor]
   |

rfdetr/models/attention.py:307:5: C901 `multi_head_attention_forward` is too complex (28 > 10)
    |
307 | def multi_head_attention_forward(
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ C901
308 |     query: Tensor,
309 |     key: Tensor,
    |

rfdetr/models/attention.py:648:63: RUF100 [*] Unused `noqa` directive (unused: `N806`)
    |
646 |     """
647 |     # Using descriptive variable names instead of single capitals
648 |     dim_q, dim_k, dim_v = q.size(-1), k.size(-1), v.size(-1)  # noqa: N806
    |                                                               ^^^^^^^^^^^^ RUF100
649 |     assert w_q.shape == (
650 |         dim_q,
    |
    = help: Remove unused `noqa` directive

rfdetr/models/attention.py:704:5: F841 Local variable `embedding_dim` is assigned to but never used
    |
702 |             same shape as the corresponding input tensor.
703 |     """
704 |     embedding_dim = q.size(-1)  # Using descriptive name instead of single capital
    |     ^^^^^^^^^^^^^ F841
705 |     if k is v:
706 |         if q is k:
    |
    = help: Remove assignment to unused variable `embedding_dim`

rfdetr/models/attention.py:711:34: F821 Undefined name `E`
    |
709 |         else:
710 |             # encoder-decoder attention
711 |             w_q, w_kv = w.split([E, E * 2])
    |                                  ^ F821
712 |             if b is None:
713 |                 b_q = b_kv = None
    |

rfdetr/models/attention.py:711:37: F821 Undefined name `E`
    |
709 |         else:
710 |             # encoder-decoder attention
711 |             w_q, w_kv = w.split([E, E * 2])
    |                                     ^ F821
712 |             if b is None:
713 |                 b_q = b_kv = None
    |

rfdetr/models/attention.py:715:38: F821 Undefined name `E`
    |
713 |                 b_q = b_kv = None
714 |             else:
715 |                 b_q, b_kv = b.split([E, E * 2])
    |                                      ^ F821
716 |             return (F.linear(q, w_q, b_q), *F.linear(k, w_kv, b_kv).chunk(2, dim=-1))
717 |     else:
    |

rfdetr/models/attention.py:715:41: F821 Undefined name `E`
    |
713 |                 b_q = b_kv = None
714 |             else:
715 |                 b_q, b_kv = b.split([E, E * 2])
    |                                         ^ F821
716 |             return (F.linear(q, w_q, b_q), *F.linear(k, w_kv, b_kv).chunk(2, dim=-1))
717 |     else:
    |

rfdetr/models/backbone/__init__.py:10:1: I001 [*] Import block is un-sorted or un-formatted
   |
 8 |   # ------------------------------------------------------------------------
 9 |
10 | / from typing import Callable
11 | | from typing import Dict, List
12 | |
13 | | import torch
14 | | from torch import nn
15 | |
16 | | from rfdetr.models.backbone.backbone import *
17 | | from rfdetr.models.position_encoding import build_position_encoding
18 | | from rfdetr.util.misc import NestedTensor
   | |_________________________________________^ I001
   |
   = help: Organize imports

rfdetr/models/backbone/__init__.py:11:1: UP035 `typing.Dict` is deprecated, use `dict` instead
   |
10 | from typing import Callable
11 | from typing import Dict, List
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP035
12 |
13 | import torch
   |

rfdetr/models/backbone/__init__.py:11:1: UP035 `typing.List` is deprecated, use `list` instead
   |
10 | from typing import Callable
11 | from typing import Dict, List
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UP035
12 |
13 | import torch
   |

rfdetr/models/backbone/__init__.py:16:1: F403 `from rfdetr.models.backbone.backbone import *` used; unable to detect undefined names
   |
14 | from torch import nn
15 |
16 | from rfdetr.models.backbone.backbone import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ F403
17 | from rfdetr.models.position_encoding import build_position_encoding
18 | from rfdetr.util.misc import NestedTensor
   |

rfdetr/models/backbone/__init__.py:86:16: F405 `Backbone` may be undefined, or defined from star imports
   |
84 |     position_embedding = build_position_encoding(hidden_dim, position_embedding)
85 |
86 |     backbone = Backbone(
   |                ^^^^^^^^ F405
87 |         encoder,
88 |         pretrained_encoder,
   |

rfdetr/models/backbone/dinov2_with_windowed_attn.py:301:9: SIM102 Use a single `if` statement instead of nested `if` statements
    |
300 |           # Validate output dimensions if not tracing
301 | /         if not torch.jit.is_tracing():
302 | |             if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
    | |___________________________________________________________________________________________________^ SIM102
303 |                   raise ValueError(
304 |                       "Width or height does not match with the interpolated position embeddings"
    |
    = help: Combine `if` statements using `and`

rfdetr/models/backbone/dinov2_with_windowed_attn.py:692:13: N806 Variable `B` in function should be lowercase
    |
690 |         if run_full_attention:
691 |             # reshape x to remove windows
692 |             B, HW, C = hidden_states.shape
    |             ^ N806
693 |             num_windows_squared = self.num_windows**2
694 |             hidden_states = hidden_states.view(
    |

rfdetr/models/backbone/dinov2_with_windowed_attn.py:692:16: N806 Variable `HW` in function should be lowercase
    |
690 |         if run_full_attention:
691 |             # reshape x to remove windows
692 |             B, HW, C = hidden_states.shape
    |                ^^ N806
693 |             num_windows_squared = self.num_windows**2
694 |             hidden_states = hidden_states.view(
    |

rfdetr/models/backbone/dinov2_with_windowed_attn.py:692:20: N806 Variable `C` in function should be lowercase
    |
690 |         if run_full_attention:
691 |             # reshape x to remove windows
692 |             B, HW, C = hidden_states.shape
    |                    ^ N806
693 |             num_windows_squared = self.num_windows**2
694 |             hidden_states = hidden_states.view(
    |

rfdetr/models/backbone/dinov2_with_windowed_attn.py:709:13: N806 Variable `B` in function should be lowercase
    |
707 |         if run_full_attention:
708 |             # reshape x to add windows back
709 |             B, HW, C = hidden_states.shape
    |             ^ N806
710 |             num_windows_squared = self.num_windows**2
711 |             # hidden_states = hidden_states.view(B * num_windows_squared, HW // num_windows_squared, C)
    |

rfdetr/models/backbone/dinov2_with_windowed_attn.py:709:16: N806 Variable `HW` in function should be lowercase
    |
707 |         if run_full_attention:
708 |             # reshape x to add windows back
709 |             B, HW, C = hidden_states.shape
    |                ^^ N806
710 |             num_windows_squared = self.num_windows**2
711 |             # hidden_states = hidden_states.view(B * num_windows_squared, HW // num_windows_squared, C)
    |

rfdetr/models/backbone/dinov2_with_windowed_attn.py:709:20: N806 Variable `C` in function should be lowercase
    |
707 |         if run_full_attention:
708 |             # reshape x to add windows back
709 |             B, HW, C = hidden_states.shape
    |                    ^ N806
710 |             num_windows_squared = self.num_windows**2
711 |             # hidden_states = hidden_states.view(B * num_windows_squared, HW // num_windows_squared, C)
    |

rfdetr/models/backbone/dinov2_with_windowed_attn.py:809:25: RUF012 Mutable class attributes should be annotated with `typing.ClassVar`
    |
807 |     main_input_name = "pixel_values"
808 |     supports_gradient_checkpointing = True
809 |     _no_split_modules = ["Dinov2WithRegistersSwiGLUFFN"]
    |                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ RUF012
810 |     _supports_sdpa = True
    |

rfdetr/models/backbone/dinov2_with_windowed_attn.py:1205:25: N806 Variable `B` in function should be lowercase
     |
1203 |                         # undo windowing
1204 |                         num_windows_squared = self.config.num_windows**2
1205 |                         B, HW, C = hidden_state.shape
     |                         ^ N806
1206 |                         num_h_patches_per_window = num_h_patches // self.config.num_windows
1207 |                         num_w_patches_per_window = num_w_patches // self.config.num_windows
     |

rfdetr/models/backbone/dinov2_with_windowed_attn.py:1205:28: N806 Variable `HW` in function should be lowercase
     |
1203 |                         # undo windowing
1204 |                         num_windows_squared = self.config.num_windows**2
1205 |                         B, HW, C = hidden_state.shape
     |                            ^^ N806
1206 |                         num_h_patches_per_window = num_h_patches // self.config.num_windows
1207 |                         num_w_patches_per_window = num_w_patches // self.config.num_windows
     |

rfdetr/models/backbone/dinov2_with_windowed_attn.py:1205:32: N806 Variable `C` in function should be lowercase
     |
1203 |                         # undo windowing
1204 |                         num_windows_squared = self.config.num_windows**2
1205 |                         B, HW, C = hidden_state.shape
     |                                ^ N806
1206 |                         num_h_patches_per_window = num_h_patches // self.config.num_windows
1207 |                         num_w_patches_per_window = num_w_patches // self.config.num_windows
     |

rfdetr/models/backbone/projector.py:20:8: N812 Lowercase `functional` imported as non-lowercase `F`
   |
18 | import torch
19 | import torch.nn as nn
20 | import torch.nn.functional as F
   |        ^^^^^^^^^^^^^^^^^^^^^^^^ N812
   |

rfdetr/models/lwdetr.py:23:1: I001 [*] Import block is un-sorted or un-formatted
   |
21 |   """
22 |
23 | / import copy
24 | | import math
25 | | from typing import Callable
26 | |
27 | | import torch
28 | | import torch.nn.functional as transforms_f
29 | | import torch.nn.functional as F
30 | | from torch import nn
31 | |
32 | | from rfdetr.models.backbone import build_backbone
33 | | from rfdetr.models.matcher import build_matcher
34 | | from rfdetr.models.transformer import build_transformer
35 | | from rfdetr.util import box_ops
36 | | from rfdetr.util.misc import (
37 | |     NestedTensor,
38 | |     accuracy,
39 | |     get_world_size,
40 | |     is_dist_avail_and_initialized,
41 | |     nested_tensor_from_tensor_list,
42 | | )
   | |_^ I001
   |
   = help: Organize imports

rfdetr/models/lwdetr.py:28:31: F401 [*] `torch.nn.functional` imported but unused
   |
27 | import torch
28 | import torch.nn.functional as transforms_f
   |                               ^^^^^^^^^^^^ F401
29 | import torch.nn.functional as F
30 | from torch import nn
   |
   = help: Remove unused import: `torch.nn.functional`

rfdetr/models/lwdetr.py:29:8: N812 Lowercase `functional` imported as non-lowercase `F`
   |
27 | import torch
28 | import torch.nn.functional as transforms_f
29 | import torch.nn.functional as F
   |        ^^^^^^^^^^^^^^^^^^^^^^^^ N812
30 | from torch import nn
   |

rfdetr/models/lwdetr.py:612:9: C901 `forward` is too complex (12 > 10)
    |
610 |         return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
611 |
612 |     def forward(self, outputs, targets):
    |         ^^^^^^^ C901
613 |         """This performs the loss computation.
614 |         Parameters:
    |

rfdetr/models/lwdetr.py:765:20: E741 Ambiguous variable name: `l`
    |
764 |         results = []
765 |         for i, (s, l, b) in enumerate(zip(scores, labels, boxes)):
    |                    ^ E741
766 |             result = {"scores": s, "labels": l, "boxes": b}
    |

rfdetr/models/matcher.py:107:9: N806 Variable `C` in function should be lowercase
    |
106 |         # Final cost matrix
107 |         C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
    |         ^ N806
108 |         C = C.view(bs, num_queries, -1).cpu()
    |

rfdetr/models/matcher.py:108:9: N806 Variable `C` in function should be lowercase
    |
106 |         # Final cost matrix
107 |         C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
108 |         C = C.view(bs, num_queries, -1).cpu()
    |         ^ N806
109 |
110 |         sizes = [len(v["boxes"]) for v in targets]
    |

rfdetr/models/matcher.py:113:9: N806 Variable `C_list` in function should be lowercase
    |
111 |         indices = []
112 |         g_num_queries = num_queries // group_detr
113 |         C_list = C.split(g_num_queries, dim=1)
    |         ^^^^^^ N806
114 |         for g_i in range(group_detr):
115 |             C_g = C_list[g_i]
    |

rfdetr/models/matcher.py:115:13: N806 Variable `C_g` in function should be lowercase
    |
113 |         C_list = C.split(g_num_queries, dim=1)
114 |         for g_i in range(group_detr):
115 |             C_g = C_list[g_i]
    |             ^^^ N806
116 |             indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
117 |             if g_i == 0:
    |

rfdetr/models/ops/functions/ms_deform_attn_func.py:19:8: N812 Lowercase `functional` imported as non-lowercase `F`
   |
18 | import torch
19 | import torch.nn.functional as F
   |        ^^^^^^^^^^^^^^^^^^^^^^^^ N812
   |

rfdetr/models/ops/functions/ms_deform_attn_func.py:25:5: N806 Variable `B` in function should be lowercase
   |
23 |     """ "for debug and test only, need to use cuda version instead"""
24 |     # B, n_heads, head_dim, N
25 |     B, n_heads, head_dim, _ = value.shape
   |     ^ N806
26 |     _, Len_q, n_heads, L, P, _ = sampling_locations.shape
27 |     value_list = value.split([H * W for H, W in value_spatial_shapes], dim=3)
   |

rfdetr/models/ops/functions/ms_deform_attn_func.py:26:8: N806 Variable `Len_q` in function should be lowercase
   |
24 |     # B, n_heads, head_dim, N
25 |     B, n_heads, head_dim, _ = value.shape
26 |     _, Len_q, n_heads, L, P, _ = sampling_locations.shape
   |        ^^^^^ N806
27 |     value_list = value.split([H * W for H, W in value_spatial_shapes], dim=3)
28 |     sampling_grids = 2 * sampling_locations - 1
   |

rfdetr/models/ops/functions/ms_deform_attn_func.py:26:24: N806 Variable `L` in function should be lowercase
   |
24 |     # B, n_heads, head_dim, N
25 |     B, n_heads, head_dim, _ = value.shape
26 |     _, Len_q, n_heads, L, P, _ = sampling_locations.shape
   |                        ^ N806
27 |     value_list = value.split([H * W for H, W in value_spatial_shapes], dim=3)
28 |     sampling_grids = 2 * sampling_locations - 1
   |

rfdetr/models/ops/functions/ms_deform_attn_func.py:26:27: N806 Variable `P` in function should be lowercase
   |
24 |     # B, n_heads, head_dim, N
25 |     B, n_heads, head_dim, _ = value.shape
26 |     _, Len_q, n_heads, L, P, _ = sampling_locations.shape
   |                           ^ N806
27 |     value_list = value.split([H * W for H, W in value_spatial_shapes], dim=3)
28 |     sampling_grids = 2 * sampling_locations - 1
   |

rfdetr/models/ops/functions/ms_deform_attn_func.py:30:16: N806 Variable `H` in function should be lowercase
   |
28 |     sampling_grids = 2 * sampling_locations - 1
29 |     sampling_value_list = []
30 |     for lid_, (H, W) in enumerate(value_spatial_shapes):
   |                ^ N806
31 |         # B, n_heads, head_dim, H, W
32 |         value_l_ = value_list[lid_].view(B * n_heads, head_dim, H, W)
   |

rfdetr/models/ops/functions/ms_deform_attn_func.py:30:19: N806 Variable `W` in function should be lowercase
   |
28 |     sampling_grids = 2 * sampling_locations - 1
29 |     sampling_value_list = []
30 |     for lid_, (H, W) in enumerate(value_spatial_shapes):
   |                   ^ N806
31 |         # B, n_heads, head_dim, H, W
32 |         value_l_ = value_list[lid_].view(B * n_heads, head_dim, H, W)
   |

rfdetr/models/ops/modules/ms_deform_attn.py:22:8: N812 Lowercase `functional` imported as non-lowercase `F`
   |
21 | import torch
22 | import torch.nn.functional as F
   |        ^^^^^^^^^^^^^^^^^^^^^^^^ N812
23 | from torch import nn
24 | from torch.nn.init import constant_, xavier_uniform_
   |

rfdetr/models/ops/modules/ms_deform_attn.py:121:9: N806 Variable `N` in function should be lowercase
    |
119 |         :return output                     (N, Length_{query}, C)
120 |         """
121 |         N, Len_q, _ = query.shape
    |         ^ N806
122 |         N, Len_in, _ = input_flatten.shape
123 |         assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
    |

rfdetr/models/ops/modules/ms_deform_attn.py:121:12: N806 Variable `Len_q` in function should be lowercase
    |
119 |         :return output                     (N, Length_{query}, C)
120 |         """
121 |         N, Len_q, _ = query.shape
    |            ^^^^^ N806
122 |         N, Len_in, _ = input_flatten.shape
123 |         assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
    |

rfdetr/models/ops/modules/ms_deform_attn.py:122:9: N806 Variable `N` in function should be lowercase
    |
120 |         """
121 |         N, Len_q, _ = query.shape
122 |         N, Len_in, _ = input_flatten.shape
    |         ^ N806
123 |         assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
    |

rfdetr/models/ops/modules/ms_deform_attn.py:122:12: N806 Variable `Len_in` in function should be lowercase
    |
120 |         """
121 |         N, Len_q, _ = query.shape
122 |         N, Len_in, _ = input_flatten.shape
    |            ^^^^^^ N806
123 |         assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
    |

rfdetr/models/position_encoding.py:154:5: N806 Variable `N_steps` in function should be lowercase
    |
153 | def build_position_encoding(hidden_dim, position_embedding):
154 |     N_steps = hidden_dim // 2
    |     ^^^^^^^ N806
155 |     if position_embedding in ("v2", "sine"):
156 |         # TODO find a better way of exposing other arguments
    |

rfdetr/models/segmentation.py:15:8: N812 Lowercase `functional` imported as non-lowercase `F`
   |
14 | import torch.nn as nn
15 | import torch.nn.functional as F
   |        ^^^^^^^^^^^^^^^^^^^^^^^^ N812
16 | from torch import Tensor
   |

rfdetr/models/segmentation.py:210:12: F821 Undefined name `torch`
    |
209 |         # Check for inf/nan values
210 |         if torch.isnan(mask_probs).any() or torch.isinf(mask_probs).any():
    |            ^^^^^ F821
211 |             print("Warning: inf/nan in mask predictions, clamping values")
212 |             mask_probs = torch.clamp(mask_probs, min=0.0, max=1.0)
    |

rfdetr/models/segmentation.py:210:45: F821 Undefined name `torch`
    |
209 |         # Check for inf/nan values
210 |         if torch.isnan(mask_probs).any() or torch.isinf(mask_probs).any():
    |                                             ^^^^^ F821
211 |             print("Warning: inf/nan in mask predictions, clamping values")
212 |             mask_probs = torch.clamp(mask_probs, min=0.0, max=1.0)
    |

rfdetr/models/segmentation.py:212:26: F821 Undefined name `torch`
    |
210 |         if torch.isnan(mask_probs).any() or torch.isinf(mask_probs).any():
211 |             print("Warning: inf/nan in mask predictions, clamping values")
212 |             mask_probs = torch.clamp(mask_probs, min=0.0, max=1.0)
    |                          ^^^^^ F821
213 |
214 |         # Add mask predictions to the output
    |

rfdetr/models/transformer.py:24:8: N812 Lowercase `functional` imported as non-lowercase `F`
   |
23 | import torch
24 | import torch.nn.functional as F
   |        ^^^^^^^^^^^^^^^^^^^^^^^^ N812
25 | from torch import Tensor, nn
   |

rfdetr/models/transformer.py:401:9: C901 `forward` is too complex (17 > 10)
    |
399 |         return new_refpoints_unsigmoid
400 |
401 |     def forward(
    |         ^^^^^^^ C901
402 |         self,
403 |         tgt,
    |

rfdetr/training_config.py:207:1: W293 [*] Blank line contains whitespace
    |
205 |         """Convert to a dictionary compatible with the legacy argparse approach."""
206 |         args_dict = self.dict()
207 |         
    | ^^^^^^^^ W293
208 |         # Ensure num_classes is set - it should have been determined from annotation file
209 |         if args_dict.get('num_classes') is None:
    |
    = help: Remove whitespace from blank line

rfdetr/training_config.py:211:1: W293 [*] Blank line contains whitespace
    |
209 |         if args_dict.get('num_classes') is None:
210 |             raise ValueError("num_classes must be set before converting to args dict")
211 |             
    | ^^^^^^^^^^^^ W293
212 |         return args_dict
    |
    = help: Remove whitespace from blank line

rfdetr/util/benchmark.py:455:5: C901 `flop_count` is too complex (11 > 10)
    |
455 | def flop_count(
    |     ^^^^^^^^^^ C901
456 |     model: nn.Module,
457 |     inputs: tuple[object, ...],
    |

rfdetr/util/early_stopping.py:39:9: C901 `update` is too complex (13 > 10)
   |
37 |         self.model = model
38 |
39 |     def update(self, log_stats: dict[str, Any]) -> None:
   |         ^^^^^^ C901
40 |         """Update early stopping state based on epoch validation metrics"""
41 |         regular_map = None
   |

rfdetr/util/get_param_dicts.py:90:9: B007 Loop control variable `n` not used within loop body
   |
88 |     # Process other parameters
89 |     other_params = []
90 |     for n, p in all_param_names.items():
   |         ^ B007
91 |         if id(p) not in included_params:
92 |             other_params.append(p)
   |
   = help: Rename unused `n` to `_n`

rfdetr/util/metrics.py:40:9: C901 `save` is too complex (15 > 10)
   |
38 |         self.history.append(values)
39 |
40 |     def save(self):
   |         ^^^^ C901
41 |         if not self.history:
42 |             print("No data to plot.")
   |

rfdetr/util/metrics.py:177:9: C901 `update` is too complex (12 > 10)
    |
175 |             )
176 |
177 |     def update(self, values: dict):
    |         ^^^^^^ C901
178 |         if not self.writer:
179 |             return
    |

rfdetr/util/metrics.py:249:9: C901 `update` is too complex (12 > 10)
    |
247 |             )
248 |
249 |     def update(self, values: dict):
    |         ^^^^^^ C901
250 |         if not wandb or not self.run:
251 |             return
    |

rfdetr/util/misc.py:282:13: SIM113 Use `enumerate()` for index variable `i` in `for` loop
    |
280 |                         )
281 |                     )
282 |             i += 1
    |             ^^^^^^ SIM113
283 |             end = time.time()
284 |         total_time = time.time() - start_time
    |

scripts/create_test_annotations.py:93:9: B007 Loop control variable `i` not used within loop body
   |
92 |     # Randomly flip classifications for annotations
93 |     for i, ann in enumerate(test_annotations.get("annotations", [])):
   |         ^ B007
94 |         if random.random() < args.flip_ratio:
95 |             # Get a random category ID different from the current one
   |
   = help: Rename unused `i` to `_i`

scripts/evaluate.py:94:5: C901 `evaluate_with_per_class_metrics` is too complex (28 > 10)
   |
94 | def evaluate_with_per_class_metrics(
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ C901
95 |     model, criterion, postprocessors, data_loader, base_ds, device, args=None
96 | ):
   |

scripts/evaluate.py:218:32: C416 Unnecessary dict comprehension (rewrite using `dict()`)
    |
216 |                           }
217 |                       ]
218 |                       mock_res = {
    |  ________________________________^
219 | |                         image_id: result for image_id, result in zip(image_ids, mock_results)
220 | |                     }
    | |_____________________^ C416
221 |                       if coco_evaluator is not None:
222 |                           coco_evaluator.update(mock_res)
    |
    = help: Rewrite using `dict()`

scripts/evaluate.py:271:5: C901 `extract_per_class_metrics` is too complex (11 > 10)
    |
271 | def extract_per_class_metrics(coco_evaluator, base_ds):
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^ C901
272 |     """
273 |     Extract per-class metrics from COCO evaluator.
    |

scripts/evaluate.py:368:5: C901 `main` is too complex (30 > 10)
    |
368 | def main(args):
    |     ^^^^ C901
369 |     # Setup device
370 |     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    |

scripts/train.py:36:5: C901 `main` is too complex (18 > 10)
   |
36 | def main():
   |     ^^^^ C901
37 |     """Main training function."""
38 |     parser = argparse.ArgumentParser(description="Train RF-DETR with iteration-based approach")
   |

scripts/train.py:64:1: W293 [*] Blank line contains whitespace
   |
62 |     if args.output_dir:
63 |         config.output_dir = args.output_dir
64 |         
   | ^^^^^^^^ W293
65 |     # Determine the number of classes from annotation file
66 |     coco_path = Path(config.coco_path)
   |
   = help: Remove whitespace from blank line

scripts/train.py:68:1: W293 [*] Blank line contains whitespace
   |
66 |     coco_path = Path(config.coco_path)
67 |     annotation_file = coco_path / config.coco_train if not Path(config.coco_train).is_absolute() else Path(config.coco_train)
68 |     
   | ^^^^ W293
69 |     # Get number of classes from annotation file - fail if can't determine
70 |     import json
   |
   = help: Remove whitespace from blank line

scripts/train.py:73:1: W293 [*] Blank line contains whitespace
   |
71 |     if not annotation_file.exists():
72 |         raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
73 |         
   | ^^^^^^^^ W293
74 |     with open(annotation_file) as f:
75 |         annotations = json.load(f)
   |
   = help: Remove whitespace from blank line

scripts/train.py:79:1: W293 [*] Blank line contains whitespace
   |
77 |         if not categories:
78 |             raise ValueError(f"No categories found in annotation file: {annotation_file}")
79 |         
   | ^^^^^^^^ W293
80 |         num_classes = len(categories)
81 |         logger.info(f"Detected {num_classes} classes from annotation file")
   |
   = help: Remove whitespace from blank line

scripts/train.py:82:1: W293 [*] Blank line contains whitespace
   |
80 |         num_classes = len(categories)
81 |         logger.info(f"Detected {num_classes} classes from annotation file")
82 |     
   | ^^^^ W293
83 |     # Set the num_classes in config
84 |     config.num_classes = num_classes
   |
   = help: Remove whitespace from blank line

Found 139 errors.
[*] 32 fixable with the `--fix` option (11 hidden fixes can be enabled with the `--unsafe-fixes` option).
rfdetr/util/obj365_to_coco_model.py:81: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/files.py:1: error: Library stubs not installed for "requests"  [import-untyped]
rfdetr/deploy/_onnx/symbolic.py:21: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/symbolic.py:25: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/symbolic.py:26: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/symbolic.py:27: error: Call to untyped function "optimizer" of "CustomOpSymbolicRegistry" in typed context  [no-untyped-call]
build/lib/rfdetr/util/obj365_to_coco_model.py:81: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/init_logs.py:12: error: Incompatible default for argument "base_dir" (default has type "None", argument has type "str")  [assignment]
build/lib/rfdetr/util/init_logs.py:12: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
build/lib/rfdetr/util/init_logs.py:12: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
build/lib/rfdetr/util/init_logs.py:25: error: Statement is unreachable  [unreachable]
build/lib/rfdetr/util/files.py:1: error: Library stubs not installed for "requests"  [import-untyped]
build/lib/rfdetr/util/files.py:1: note: Hint: "python3 -m pip install types-requests"
build/lib/rfdetr/util/files.py:1: note: (or run "mypy --install-types" to install all missing stub packages)
build/lib/rfdetr/util/files.py:1: note: See https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-imports
build/lib/rfdetr/util/files.py:5: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/symbolic.py:16: error: Need type annotation for "_OPTIMIZER" (hint: "_OPTIMIZER: list[<type>] = ...")  [var-annotated]
build/lib/rfdetr/deploy/_onnx/symbolic.py:19: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/symbolic.py:23: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/symbolic.py:24: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/symbolic.py:25: error: Call to untyped function "optimizer" of "CustomOpSymbolicRegistry" in typed context  [no-untyped-call]
example_custom_annotations.py:10: error: Function is missing a return type annotation  [no-untyped-def]
example_custom_annotations.py:47: error: Function is missing a return type annotation  [no-untyped-def]
example_custom_annotations.py:47: note: Use "-> None" if function does not return a value
example_custom_annotations.py:48: error: Call to untyped function "create_custom_args" in typed context  [no-untyped-call]
example_custom_annotations.py:71: error: Call to untyped function "main" in typed context  [no-untyped-call]
build/lib/rfdetr/util/early_stopping.py:22: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/early_stopping.py:31: error: Function is missing a type annotation  [no-untyped-def]
scripts/create_test_annotations.py:110: error: Call to untyped function "get_args_parser" in typed context  [no-untyped-call]
scripts/create_test_annotations.py:112: error: Call to untyped function "main" in typed context  [no-untyped-call]
rfdetr/util/logging_config.py:126: error: Incompatible return value type (got "dict[str, object]", expected "dict[str, Handler]")  [return-value]
build/lib/rfdetr/util/logging_config.py:107: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/logging_config.py:126: error: Incompatible return value type (got "dict[str, object]", expected "dict[str, Handler]")  [return-value]
rfdetr/util/error_handling.py:174: error: Name "traceback.TracebackType" is not defined  [name-defined]
rfdetr/util/error_handling.py:187: error: Argument 2 to "log_exception" has incompatible type "BaseException"; expected "Exception"  [arg-type]
rfdetr/util/error_handling.py:190: error: Incompatible types in assignment (expression has type "BaseException", variable has type "Optional[Exception]")  [assignment]
rfdetr/training_config.py:14: error: Library stubs not installed for "yaml"  [import-untyped]
rfdetr/util/drop_scheduler.py:11: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/drop_scheduler.py:11: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:30: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:34: error: Call to untyped function "set_severity" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:36: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:39: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:46: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:51: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:56: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:61: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:67: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:74: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:81: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:83: error: "str" not callable  [operator]
rfdetr/deploy/_onnx/optimizer.py:84: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:93: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:97: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:185: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:188: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:198: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:201: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:315: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:318: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:384: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:387: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:415: error: Name "nLayerNormPlugin" is not defined  [name-defined]
rfdetr/deploy/_onnx/optimizer.py:423: error: Name "nLayerNormPlugin" is not defined  [name-defined]
rfdetr/deploy/_onnx/optimizer.py:428: error: Name "nLayerNormPlugin" is not defined  [name-defined]
rfdetr/deploy/_onnx/optimizer.py:432: error: Name "nLayerNormPlugin" is not defined  [name-defined]
rfdetr/deploy/_onnx/optimizer.py:438: error: Name "nLayerNormPlugin" is not defined  [name-defined]
rfdetr/deploy/_onnx/optimizer.py:451: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:452: error: Name "nLayerNormPlugin" is not defined  [name-defined]
rfdetr/deploy/_onnx/optimizer.py:454: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:502: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:505: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:570: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:572: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:627: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:630: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:686: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:688: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:747: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:757: error: Call to untyped function "mha_mhca_detected" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:765: error: Call to untyped function "fuse_kv" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:767: error: Call to untyped function "insert_fmhca" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:771: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:781: error: Call to untyped function "mha_mhca_detected" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:786: error: Call to untyped function "fuse_qkv" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:788: error: Call to untyped function "insert_fmha" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:792: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:794: error: Call to untyped function "fuse_kv_insert_fmhca" in typed context  [no-untyped-call]
rfdetr/deploy/_onnx/optimizer.py:798: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/_onnx/optimizer.py:800: error: Call to untyped function "fuse_qkv_insert_fmha" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:30: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:34: error: Call to untyped function "set_severity" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:36: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:39: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:46: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:51: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:56: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:61: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:67: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:74: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:81: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:84: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:93: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:97: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:185: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:188: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:198: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:201: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:315: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:318: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:384: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:387: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:452: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:455: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:503: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:506: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:571: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:573: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:628: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:631: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:687: error: Call to untyped function "cleanup" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:689: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:748: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:758: error: Call to untyped function "mha_mhca_detected" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:766: error: Call to untyped function "fuse_kv" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:768: error: Call to untyped function "insert_fmhca" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:772: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:782: error: Call to untyped function "mha_mhca_detected" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:787: error: Call to untyped function "fuse_qkv" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:789: error: Call to untyped function "insert_fmha" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:793: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:795: error: Call to untyped function "fuse_kv_insert_fmhca" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/_onnx/optimizer.py:799: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/_onnx/optimizer.py:801: error: Call to untyped function "fuse_qkv_insert_fmha" in typed context  [no-untyped-call]
rfdetr/model_config.py:98: error: Incompatible types in assignment (expression has type "str", variable has type "Literal['cpu', 'cuda', 'mps']")  [assignment]
rfdetr/model_config.py:124: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/config.py:33: error: Incompatible types in assignment (expression has type "str", variable has type "Literal['cpu', 'cuda', 'mps']")  [assignment]
rfdetr/config.py:96: error: Incompatible types in assignment (expression has type "None", variable has type "list[str]")  [assignment]
rfdetr/util/utils.py:12: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/utils.py:25: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/utils.py:32: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/utils.py:41: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/utils.py:42: error: Call to untyped function "_get_decay" in typed context  [no-untyped-call]
rfdetr/util/utils.py:43: error: Call to untyped function "_update" in typed context  [no-untyped-call]
rfdetr/util/utils.py:46: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/utils.py:47: error: Call to untyped function "_update" in typed context  [no-untyped-call]
rfdetr/util/utils.py:51: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/util/utils.py:59: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/utils.py:65: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/utils.py:66: error: Call to untyped function "isbetter" in typed context  [no-untyped-call]
rfdetr/util/utils.py:78: error: Missing type parameters for generic type "dict"  [type-arg]
rfdetr/util/utils.py:86: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/util/utils.py:93: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/utils.py:98: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/util/utils.py:101: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/util/utils.py:102: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/util/utils.py:104: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/util/utils.py:105: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/util/utils.py:107: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/utils.py:118: error: Call to untyped function "summary" in typed context  [no-untyped-call]
rfdetr/util/utils.py:124: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:45: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:48: error: Need type annotation for "deque"  [var-annotated]
rfdetr/util/misc.py:53: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:58: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:58: note: Use "-> None" if function does not return a value
rfdetr/util/misc.py:62: error: Call to untyped function "is_dist_avail_and_initialized" in typed context  [no-untyped-call]
rfdetr/util/misc.py:67: error: Incompatible types in assignment (expression has type "list[Any]", variable has type "Tensor")  [assignment]
rfdetr/util/misc.py:69: error: Incompatible types in assignment (expression has type "Tensor", variable has type "float")  [assignment]
rfdetr/util/misc.py:72: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:77: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:82: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:86: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:90: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:93: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:103: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:111: error: Call to untyped function "get_world_size" in typed context  [no-untyped-call]
rfdetr/util/misc.py:117: error: Call to untyped function "from_buffer" of "TypedStorage" in typed context  [no-untyped-call]
rfdetr/util/misc.py:124: error: List comprehension has incompatible type List[int]; expected List[Tensor]  [misc]
rfdetr/util/misc.py:125: error: Value of type variable "SupportsRichComparisonT" of "max" cannot be "Tensor"  [type-var]
rfdetr/util/misc.py:132: error: Argument 1 to "empty" has incompatible type "tuple[Tensor]"; expected "Sequence[Union[int, SymInt]]"  [arg-type]
rfdetr/util/misc.py:134: error: Argument "size" to "empty" has incompatible type "tuple[Tensor]"; expected "Sequence[Union[int, SymInt]]"  [arg-type]
rfdetr/util/misc.py:146: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:155: error: Call to untyped function "get_world_size" in typed context  [no-untyped-call]
rfdetr/util/misc.py:165: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment]
rfdetr/util/misc.py:174: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:184: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:191: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:198: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:204: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:204: note: Use "-> None" if function does not return a value
rfdetr/util/misc.py:208: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:211: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:222: error: Call to untyped function "SmoothedValue" in typed context  [no-untyped-call]
rfdetr/util/misc.py:223: error: Call to untyped function "SmoothedValue" in typed context  [no-untyped-call]
rfdetr/util/misc.py:250: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/util/misc.py:252: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/util/misc.py:256: error: Call to untyped function "is_main_process" in typed context  [no-untyped-call]
rfdetr/util/misc.py:291: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:294: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:301: error: Call to untyped function "_run" in typed context  [no-untyped-call]
rfdetr/util/misc.py:303: error: Call to untyped function "_run" in typed context  [no-untyped-call]
rfdetr/util/misc.py:305: error: Call to untyped function "_run" in typed context  [no-untyped-call]
rfdetr/util/misc.py:312: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:318: error: Name "List" is not defined  [name-defined]
rfdetr/util/misc.py:318: note: Did you forget to import it from "typing"? (Suggestion: "from typing import List")
rfdetr/util/misc.py:328: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/util/misc.py:332: error: Name "Device" is not defined  [name-defined]
rfdetr/util/misc.py:343: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:346: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:350: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:377: error: Untyped decorator makes function "_onnx_nested_tensor_from_tensor_list" untyped  [misc]
rfdetr/util/misc.py:382: error: List comprehension has incompatible type List[int]; expected List[Tensor]  [misc]
rfdetr/util/misc.py:385: error: Incompatible types in assignment (expression has type "tuple[Tensor, ...]", variable has type "list[Tensor]")  [assignment]
rfdetr/util/misc.py:395: error: Argument 2 to "pad" has incompatible type "tuple[int, Tensor, int, Tensor, int, Tensor]"; expected "Sequence[int]"  [arg-type]
rfdetr/util/misc.py:399: error: Argument 2 to "pad" has incompatible type "tuple[int, Tensor, int, Tensor]"; expected "Sequence[int]"  [arg-type]
rfdetr/util/misc.py:408: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:416: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:424: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:430: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:431: error: Call to untyped function "is_dist_avail_and_initialized" in typed context  [no-untyped-call]
rfdetr/util/misc.py:436: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:437: error: Call to untyped function "is_dist_avail_and_initialized" in typed context  [no-untyped-call]
rfdetr/util/misc.py:442: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/misc.py:443: error: Call to untyped function "get_rank" in typed context  [no-untyped-call]
rfdetr/util/misc.py:446: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:450: error: Call to untyped function "is_main_process" in typed context  [no-untyped-call]
rfdetr/util/misc.py:454: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:479: error: Call to untyped function "setup_for_distributed" in typed context  [no-untyped-call]
rfdetr/util/misc.py:483: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:501: error: Name "List" is not defined  [name-defined]
rfdetr/util/misc.py:501: note: Did you forget to import it from "typing"? (Suggestion: "from typing import List")
rfdetr/util/misc.py:510: error: Returning Any from function declared to return "Tensor"  [no-any-return]
rfdetr/util/misc.py:514: error: Returning Any from function declared to return "Tensor"  [no-any-return]
rfdetr/util/misc.py:516: error: Returning Any from function declared to return "Tensor"  [no-any-return]
rfdetr/util/misc.py:519: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/misc.py:526: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/metrics.py:9: error: Cannot assign to a type  [misc]
rfdetr/util/metrics.py:9: error: Incompatible types in assignment (expression has type "None", variable has type "type[SummaryWriter]")  [assignment]
rfdetr/util/metrics.py:21: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/metrics.py:35: error: Need type annotation for "history" (hint: "history: list[<type>] = ...")  [var-annotated]
rfdetr/util/metrics.py:37: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/metrics.py:37: error: Missing type parameters for generic type "dict"  [type-arg]
rfdetr/util/metrics.py:40: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/metrics.py:45: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/metrics.py:48: error: Call to untyped function "get_array" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:49: error: Call to untyped function "get_array" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:50: error: Call to untyped function "get_array" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:55: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:58: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:61: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:68: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:71: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:74: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:166: error: Function "SummaryWriter" could always be true in boolean context  [truthy-function]
rfdetr/util/metrics.py:167: error: Call to untyped function "SummaryWriter" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:172: error: Statement is unreachable  [unreachable]
rfdetr/util/metrics.py:177: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/metrics.py:177: error: Missing type parameters for generic type "dict"  [type-arg]
rfdetr/util/metrics.py:184: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:186: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:190: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:191: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:192: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:194: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:196: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:198: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:202: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:203: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:204: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:206: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:208: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:210: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:212: error: Call to untyped function "flush" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:214: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/metrics.py:214: note: Use "-> None" if function does not return a value
rfdetr/util/metrics.py:218: error: Call to untyped function "close" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:237: error: Missing type parameters for generic type "dict"  [type-arg]
rfdetr/util/metrics.py:249: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/metrics.py:249: error: Missing type parameters for generic type "dict"  [type-arg]
rfdetr/util/metrics.py:263: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:264: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:265: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:275: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:276: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:277: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
rfdetr/util/metrics.py:287: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/metrics.py:287: note: Use "-> None" if function does not return a value
rfdetr/util/box_ops.py:22: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/box_ops.py:33: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/box_ops.py:40: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/box_ops.py:56: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/box_ops.py:67: error: Call to untyped function "box_iou" in typed context  [no-untyped-call]
rfdetr/util/box_ops.py:78: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/benchmark.py:45: error: "object" has no attribute "isCompleteTensor"  [attr-defined]
rfdetr/util/benchmark.py:46: error: "object" has no attribute "type"  [attr-defined]
rfdetr/util/benchmark.py:49: error: Returning Any from function declared to return "list[int]"  [no-any-return]
rfdetr/util/benchmark.py:50: error: "object" has no attribute "type"  [attr-defined]
rfdetr/util/benchmark.py:52: error: "object" has no attribute "type"  [attr-defined]
rfdetr/util/benchmark.py:54: error: "object" has no attribute "type"  [attr-defined]
rfdetr/util/benchmark.py:56: error: "object" has no attribute "type"  [attr-defined]
rfdetr/util/benchmark.py:88: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/benchmark.py:103: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/benchmark.py:113: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/benchmark.py:120: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/benchmark.py:127: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/benchmark.py:135: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/benchmark.py:215: error: "object" has no attribute "toIValue"  [attr-defined]
rfdetr/util/benchmark.py:223: error: "object" has no attribute "node"  [attr-defined]
rfdetr/util/benchmark.py:307: error: Incompatible return value type (got "Counter[str]", expected "Number")  [return-value]
rfdetr/util/benchmark.py:327: error: Incompatible return value type (got "Counter[str]", expected "Number")  [return-value]
rfdetr/util/benchmark.py:346: error: Incompatible types in assignment (expression has type "floating[Union[_64Bit, Any]]", variable has type "int")  [assignment]
rfdetr/util/benchmark.py:349: error: Incompatible types in assignment (expression has type "floating[Union[_64Bit, Any]]", variable has type "int")  [assignment]
rfdetr/util/benchmark.py:351: error: Incompatible return value type (got "Counter[str]", expected "Number")  [return-value]
rfdetr/util/benchmark.py:357: error: Missing type parameters for generic type "typing.Callable"  [type-arg]
rfdetr/util/benchmark.py:459: error: Missing type parameters for generic type "typing.Callable"  [type-arg]
rfdetr/util/benchmark.py:505: error: Module "torch.jit" does not explicitly export attribute "_get_trace_graph"  [attr-defined]
rfdetr/util/benchmark.py:505: error: Call to untyped function "_get_trace_graph" in typed context  [no-untyped-call]
rfdetr/util/benchmark.py:508: error: Need type annotation for "skipped_ops"  [var-annotated]
rfdetr/util/benchmark.py:509: error: Need type annotation for "total_flop_counter"  [var-annotated]
rfdetr/util/benchmark.py:523: error: Incompatible types in assignment (expression has type "list[Any]", variable has type "tuple[object, ...]")  [assignment]
rfdetr/util/benchmark.py:541: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/benchmark.py:547: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/benchmark.py:548: error: Call to untyped function "warmup" in typed context  [no-untyped-call]
rfdetr/util/benchmark.py:557: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/benchmark.py:567: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/benchmark.py:591: error: Call to untyped function "measure_time" in typed context  [no-untyped-call]
rfdetr/util/benchmark.py:596: error: Call to untyped function "fmt_res" in typed context  [no-untyped-call]
rfdetr/util/benchmark.py:598: error: Call to untyped function "fmt_res" in typed context  [no-untyped-call]
rfdetr/util/benchmark.py:601: error: Incompatible types in assignment (expression has type "dict[str, Any]", variable has type "defaultdict[str, float]")  [assignment]
rfdetr/util/benchmark.py:601: error: Call to untyped function "fmt_res" in typed context  [no-untyped-call]
rfdetr/models/attention.py:60: error: Incompatible return value type (got "tuple[Tensor, Tensor, Tensor, Union[int, float]]", expected "tuple[Tensor, Tensor, Tensor, int]")  [return-value]
rfdetr/models/attention.py:100: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/attention.py:159: error: Incompatible types in assignment (expression has type "None", variable has type "Parameter")  [assignment]
rfdetr/models/attention.py:160: error: Incompatible types in assignment (expression has type "None", variable has type "Parameter")  [assignment]
rfdetr/models/attention.py:162: error: Incompatible types in assignment (expression has type "None", variable has type "Parameter")  [assignment]
rfdetr/models/attention.py:163: error: Incompatible types in assignment (expression has type "None", variable has type "Parameter")  [assignment]
rfdetr/models/attention.py:164: error: Incompatible types in assignment (expression has type "None", variable has type "Parameter")  [assignment]
rfdetr/models/attention.py:168: error: Call to untyped function "_reset_parameters" in typed context  [no-untyped-call]
rfdetr/models/attention.py:170: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/attention.py:170: note: Use "-> None" if function does not return a value
rfdetr/models/attention.py:187: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/attention.py:192: error: Call to untyped function "__setstate__" in typed context  [no-untyped-call]
rfdetr/models/attention.py:398: error: Subclass of "int" and "Tensor" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/models/attention.py:400: error: Statement is unreachable  [unreachable]
rfdetr/models/attention.py:406: error: Unsupported operand types for // ("None" and "int")  [operator]
rfdetr/models/attention.py:406: error: Incompatible types in assignment (expression has type "int", variable has type "Tensor")  [assignment]
rfdetr/models/attention.py:526: error: No overload variant of "view" of "TensorBase" matches argument types "int", "int", "int", "Tensor"  [call-overload]
rfdetr/models/attention.py:526: note: Possible overload variants:
rfdetr/models/attention.py:526: note:     def view(self, dtype: dtype) -> Tensor
rfdetr/models/attention.py:526: note:     def view(self, size: Sequence[Union[int, SymInt]]) -> Tensor
rfdetr/models/attention.py:526: note:     def view(self, *size: Union[int, SymInt]) -> Tensor
rfdetr/models/attention.py:588: error: No overload variant of "view" of "TensorBase" matches argument types "int", "int", "Optional[Tensor]"  [call-overload]
rfdetr/models/attention.py:588: note: Possible overload variants:
rfdetr/models/attention.py:588: note:     def view(self, dtype: dtype) -> Tensor
rfdetr/models/attention.py:588: note:     def view(self, size: Sequence[Union[int, SymInt]]) -> Tensor
rfdetr/models/attention.py:588: note:     def view(self, *size: Union[int, SymInt]) -> Tensor
rfdetr/models/attention.py:708: error: Incompatible return value type (got "tuple[Tensor, ...]", expected "list[Tensor]")  [return-value]
rfdetr/models/attention.py:711: error: Call to untyped function "split" in typed context  [no-untyped-call]
rfdetr/models/attention.py:711: error: Name "E" is not defined  [name-defined]
rfdetr/models/attention.py:715: error: Call to untyped function "split" in typed context  [no-untyped-call]
rfdetr/models/attention.py:715: error: Name "E" is not defined  [name-defined]
rfdetr/models/attention.py:716: error: Incompatible return value type (got "tuple[Tensor, ...]", expected "list[Tensor]")  [return-value]
rfdetr/models/attention.py:723: error: Incompatible return value type (got "tuple[Tensor, Tensor, Tensor]", expected "list[Tensor]")  [return-value]
rfdetr/models/attention.py:772: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/attention.py:788: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/ops/functions/ms_deform_attn_func.py:43: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Tensor]")  [assignment]
rfdetr/models/backbone/projector.py:31: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/projector.py:38: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/projector.py:52: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/projector.py:67: error: Call to untyped function "LayerNorm" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:72: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/projector.py:77: error: Incompatible types in assignment (expression has type "ReLU", variable has type "SiLU")  [assignment]
rfdetr/models/backbone/projector.py:79: error: Incompatible types in assignment (expression has type "LeakyReLU", variable has type "SiLU")  [assignment]
rfdetr/models/backbone/projector.py:81: error: Incompatible types in assignment (expression has type "Identity", variable has type "SiLU")  [assignment]
rfdetr/models/backbone/projector.py:90: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/projector.py:119: error: Call to untyped function "get_norm" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:119: error: Incompatible types in assignment (expression has type "Union[Any, BatchNorm2d]", variable has type "RMSNorm")  [assignment]
rfdetr/models/backbone/projector.py:120: error: Call to untyped function "get_activation" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:122: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/projector.py:131: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/projector.py:146: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:147: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:152: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/projector.py:160: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/projector.py:166: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:167: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:171: error: Call to untyped function "Bottleneck" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:185: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/projector.py:198: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/projector.py:223: error: Need type annotation for "stages_sampling" (hint: "stages_sampling: list[<type>] = ...")  [var-annotated]
rfdetr/models/backbone/projector.py:240: error: Call to untyped function "get_norm" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:266: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:274: error: Incompatible types in assignment (expression has type "Sequential", variable has type "list[Any]")  [assignment]
rfdetr/models/backbone/projector.py:280: error: Call to untyped function "C2f" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:281: error: Call to untyped function "get_norm" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:283: error: Incompatible types in assignment (expression has type "Sequential", variable has type "list[Any]")  [assignment]
rfdetr/models/backbone/projector.py:287: error: Argument 1 to "ModuleList" has incompatible type "list[list[Any]]"; expected "Optional[Iterable[Module]]"  [arg-type]
rfdetr/models/backbone/projector.py:289: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/projector.py:317: error: Need type annotation for "stage_sampling"  [var-annotated]
rfdetr/models/backbone/projector.py:317: error: Argument 1 to "enumerate" has incompatible type Module; expected "Iterable[Never]"  [arg-type]
rfdetr/models/backbone/projector.py:319: error: Incompatible types in assignment (expression has type "Union[Tensor, Any]", variable has type "list[Any]")  [assignment]
rfdetr/models/backbone/projector.py:327: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/projector.py:330: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:331: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:333: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:334: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:335: error: Call to untyped function "get_norm" in typed context  [no-untyped-call]
rfdetr/models/backbone/projector.py:337: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:27: error: Module "transformers.utils" does not explicitly export attribute "add_code_sample_docstrings"  [attr-defined]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:27: error: Module "transformers.utils" does not explicitly export attribute "add_start_docstrings"  [attr-defined]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:27: error: Module "transformers.utils" does not explicitly export attribute "add_start_docstrings_to_model_forward"  [attr-defined]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:27: error: Module "transformers.utils" does not explicitly export attribute "replace_return_docstrings"  [attr-defined]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:27: error: Module "transformers.utils" does not explicitly export attribute "torch_int"  [attr-defined]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:130: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:158: error: Call to untyped function "__init__" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:197: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:230: error: Returning Any from function declared to return "Tensor"  [no-any-return]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:248: error: Call to untyped function "Dinov2WithRegistersPatchEmbeddings" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:271: error: Module "torch.jit" does not explicitly export attribute "is_tracing"  [attr-defined]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:271: error: Call to untyped function "is_tracing" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:284: error: Call to untyped function "torch_int" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:294: error: Call to untyped function "torch_int" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:301: error: Module "torch.jit" does not explicitly export attribute "is_tracing"  [attr-defined]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:301: error: Call to untyped function "is_tracing" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:373: error: Item "None" of "Optional[Parameter]" has no attribute "expand"  [union-attr]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:384: error: Returning Any from function declared to return "Tensor"  [no-any-return]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:413: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:457: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:465: error: "Logger" has no attribute "warning_once"  [attr-defined]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:495: error: Incompatible return value type (got "tuple[Tensor, None]", expected "Union[tuple[Tensor, Tensor], tuple[Tensor]]")  [return-value]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:521: error: Need type annotation for "pruned_heads" (hint: "pruned_heads: set[<type>] = ...")  [var-annotated]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:527: error: Argument 1 to "find_pruneable_heads_and_indices" has incompatible type "set[int]"; expected "list[int]"  [arg-type]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:557: error: Returning Any from function declared to return "Union[tuple[Tensor, Tensor], tuple[Tensor]]"  [no-any-return]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:567: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:605: error: Argument 2 to "drop_path" has incompatible type "Optional[float]"; expected "float"  [arg-type]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:612: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:631: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:644: error: Returning Any from function declared to return "Tensor"  [no-any-return]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:677: error: Incompatible types in assignment (expression has type "Dinov2WithRegistersMLP", variable has type "Dinov2WithRegistersSwiGLUFFN")  [assignment]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:732: error: Returning Any from function declared to return "Union[tuple[Tensor, Tensor], tuple[Tensor]]"  [no-any-return]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:751: error: Missing type parameters for generic type "tuple"  [type-arg]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:757: error: Expected iterable as variadic argument  [misc]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:768: error: "Tensor" not callable  [operator]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:783: error: Expected iterable as variadic argument  [misc]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:786: error: Expected iterable as variadic argument  [misc]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:793: error: Argument "last_hidden_state" to "BaseModelOutput" has incompatible type "Tensor"; expected "Optional[FloatTensor]"  [arg-type]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:805: error: Incompatible types in assignment (expression has type "type[WindowedDinov2WithRegistersConfig]", base class "PreTrainedModel" defined the type as "None")  [assignment]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:809: error: Incompatible types in assignment (expression has type "list[str]", base class "PreTrainedModel" defined the type as "None")  [assignment]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:825: error: Statement is unreachable  [unreachable]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:880: error: Call to untyped function "add_start_docstrings" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:895: error: Call to untyped function "post_init" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:906: error: Item "Tensor" of "Union[Tensor, Module]" has no attribute "prune_heads"  [union-attr]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:906: error: "Tensor" not callable  [operator]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:908: error: Call to untyped function "add_start_docstrings_to_model_forward" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:908: error: Untyped decorator makes function "forward" untyped  [misc]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:909: error: Call to untyped function "add_code_sample_docstrings" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:909: error: Untyped decorator makes function "forward" untyped  [misc]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:924: error: Missing type parameters for generic type "tuple"  [type-arg]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:960: error: Returning Any from function declared to return "Union[tuple[Any, ...], BaseModelOutputWithPooling]"  [no-any-return]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:997: error: Call to untyped function "add_start_docstrings" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1019: error: Call to untyped function "post_init" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1021: error: Call to untyped function "add_start_docstrings_to_model_forward" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1021: error: Untyped decorator makes function "forward" untyped  [misc]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1022: error: Call to untyped function "add_code_sample_docstrings" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1022: error: Untyped decorator makes function "forward" untyped  [misc]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1036: error: Missing type parameters for generic type "tuple"  [type-arg]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1083: error: Incompatible types in assignment (expression has type "CrossEntropyLoss", variable has type "MSELoss")  [assignment]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1086: error: Incompatible types in assignment (expression has type "BCEWithLogitsLoss", variable has type "MSELoss")  [assignment]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1101: error: Call to untyped function "add_start_docstrings" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1122: error: Call to untyped function "post_init" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1127: error: Call to untyped function "add_start_docstrings_to_model_forward" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1127: error: Untyped decorator makes function "forward" untyped  [misc]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1128: error: Call to untyped function "replace_return_docstrings" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1128: error: Untyped decorator makes function "forward" untyped  [misc]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1226: error: Incompatible types in assignment (expression has type "tuple[Any]", variable has type "tuple[()]")  [assignment]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1233: error: Returning Any from function declared to return "BackboneOutput"  [no-any-return]
rfdetr/models/backbone/dinov2_with_windowed_attn.py:1236: error: Argument "feature_maps" to "BackboneOutput" has incompatible type "tuple[()]"; expected "Optional[tuple[FloatTensor]]"  [arg-type]
rfdetr/models/backbone/base.py:14: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/backbone/base.py:14: note: Use "-> None" if function does not return a value
rfdetr/models/backbone/base.py:17: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/backbone/base.py:17: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/config.py:34: error: Incompatible types in assignment (expression has type "str", variable has type "Literal['cpu', 'cuda', 'mps']")  [assignment]
build/lib/rfdetr/config.py:98: error: Incompatible types in assignment (expression has type "None", variable has type "list[str]")  [assignment]
build/lib/rfdetr/util/utils.py:12: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/utils.py:25: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/utils.py:32: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/utils.py:41: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/utils.py:42: error: Call to untyped function "_get_decay" in typed context  [no-untyped-call]
build/lib/rfdetr/util/utils.py:43: error: Call to untyped function "_update" in typed context  [no-untyped-call]
build/lib/rfdetr/util/utils.py:46: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/utils.py:47: error: Call to untyped function "_update" in typed context  [no-untyped-call]
build/lib/rfdetr/util/utils.py:51: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/util/utils.py:59: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/utils.py:65: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/utils.py:66: error: Call to untyped function "isbetter" in typed context  [no-untyped-call]
build/lib/rfdetr/util/utils.py:78: error: Missing type parameters for generic type "dict"  [type-arg]
build/lib/rfdetr/util/utils.py:86: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/util/utils.py:93: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/utils.py:98: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/util/utils.py:101: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/util/utils.py:102: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/util/utils.py:104: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/util/utils.py:105: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/util/utils.py:107: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/utils.py:118: error: Call to untyped function "summary" in typed context  [no-untyped-call]
build/lib/rfdetr/util/utils.py:124: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:45: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:48: error: Need type annotation for "deque"  [var-annotated]
build/lib/rfdetr/util/misc.py:53: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:58: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:58: note: Use "-> None" if function does not return a value
build/lib/rfdetr/util/misc.py:62: error: Call to untyped function "is_dist_avail_and_initialized" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:67: error: Incompatible types in assignment (expression has type "list[Any]", variable has type "Tensor")  [assignment]
build/lib/rfdetr/util/misc.py:69: error: Incompatible types in assignment (expression has type "Tensor", variable has type "float")  [assignment]
build/lib/rfdetr/util/misc.py:72: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:77: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:82: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:86: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:90: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:93: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:103: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:111: error: Call to untyped function "get_world_size" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:117: error: Call to untyped function "from_buffer" of "TypedStorage" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:124: error: List comprehension has incompatible type List[int]; expected List[Tensor]  [misc]
build/lib/rfdetr/util/misc.py:125: error: Value of type variable "SupportsRichComparisonT" of "max" cannot be "Tensor"  [type-var]
build/lib/rfdetr/util/misc.py:132: error: Argument 1 to "empty" has incompatible type "tuple[Tensor]"; expected "Sequence[Union[int, SymInt]]"  [arg-type]
build/lib/rfdetr/util/misc.py:134: error: Argument "size" to "empty" has incompatible type "tuple[Tensor]"; expected "Sequence[Union[int, SymInt]]"  [arg-type]
build/lib/rfdetr/util/misc.py:146: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:155: error: Call to untyped function "get_world_size" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:165: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment]
build/lib/rfdetr/util/misc.py:174: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:184: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:191: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:198: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:204: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:204: note: Use "-> None" if function does not return a value
build/lib/rfdetr/util/misc.py:208: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:211: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:222: error: Call to untyped function "SmoothedValue" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:223: error: Call to untyped function "SmoothedValue" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:250: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:252: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:256: error: Call to untyped function "is_main_process" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:291: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:294: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:301: error: Call to untyped function "_run" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:303: error: Call to untyped function "_run" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:305: error: Call to untyped function "_run" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:312: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:318: error: Name "List" is not defined  [name-defined]
build/lib/rfdetr/util/misc.py:318: note: Did you forget to import it from "typing"? (Suggestion: "from typing import List")
build/lib/rfdetr/util/misc.py:328: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/util/misc.py:332: error: Name "Device" is not defined  [name-defined]
build/lib/rfdetr/util/misc.py:343: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:346: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:350: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:377: error: Untyped decorator makes function "_onnx_nested_tensor_from_tensor_list" untyped  [misc]
build/lib/rfdetr/util/misc.py:382: error: List comprehension has incompatible type List[int]; expected List[Tensor]  [misc]
build/lib/rfdetr/util/misc.py:385: error: Incompatible types in assignment (expression has type "tuple[Tensor, ...]", variable has type "list[Tensor]")  [assignment]
build/lib/rfdetr/util/misc.py:395: error: Argument 2 to "pad" has incompatible type "tuple[int, Tensor, int, Tensor, int, Tensor]"; expected "Sequence[int]"  [arg-type]
build/lib/rfdetr/util/misc.py:399: error: Argument 2 to "pad" has incompatible type "tuple[int, Tensor, int, Tensor]"; expected "Sequence[int]"  [arg-type]
build/lib/rfdetr/util/misc.py:408: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:416: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:424: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:430: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:431: error: Call to untyped function "is_dist_avail_and_initialized" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:436: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:437: error: Call to untyped function "is_dist_avail_and_initialized" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:442: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:443: error: Call to untyped function "get_rank" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:446: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:450: error: Call to untyped function "is_main_process" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:454: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:479: error: Call to untyped function "setup_for_distributed" in typed context  [no-untyped-call]
build/lib/rfdetr/util/misc.py:483: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:501: error: Name "List" is not defined  [name-defined]
build/lib/rfdetr/util/misc.py:501: note: Did you forget to import it from "typing"? (Suggestion: "from typing import List")
build/lib/rfdetr/util/misc.py:510: error: Returning Any from function declared to return "Tensor"  [no-any-return]
build/lib/rfdetr/util/misc.py:514: error: Returning Any from function declared to return "Tensor"  [no-any-return]
build/lib/rfdetr/util/misc.py:516: error: Returning Any from function declared to return "Tensor"  [no-any-return]
build/lib/rfdetr/util/misc.py:519: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/misc.py:526: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/metrics.py:9: error: Cannot assign to a type  [misc]
build/lib/rfdetr/util/metrics.py:9: error: Incompatible types in assignment (expression has type "None", variable has type "type[SummaryWriter]")  [assignment]
build/lib/rfdetr/util/metrics.py:21: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/metrics.py:35: error: Need type annotation for "history" (hint: "history: list[<type>] = ...")  [var-annotated]
build/lib/rfdetr/util/metrics.py:37: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/metrics.py:37: error: Missing type parameters for generic type "dict"  [type-arg]
build/lib/rfdetr/util/metrics.py:40: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/metrics.py:45: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/metrics.py:48: error: Call to untyped function "get_array" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:49: error: Call to untyped function "get_array" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:50: error: Call to untyped function "get_array" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:55: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:58: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:61: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:68: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:71: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:74: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:166: error: Function "SummaryWriter" could always be true in boolean context  [truthy-function]
build/lib/rfdetr/util/metrics.py:167: error: Call to untyped function "SummaryWriter" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:172: error: Statement is unreachable  [unreachable]
build/lib/rfdetr/util/metrics.py:177: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/metrics.py:177: error: Missing type parameters for generic type "dict"  [type-arg]
build/lib/rfdetr/util/metrics.py:184: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:186: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:190: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:191: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:192: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:194: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:196: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:198: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:202: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:203: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:204: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:206: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:208: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:210: error: Call to untyped function "add_scalar" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:212: error: Call to untyped function "flush" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:214: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/metrics.py:214: note: Use "-> None" if function does not return a value
build/lib/rfdetr/util/metrics.py:218: error: Call to untyped function "close" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:237: error: Missing type parameters for generic type "dict"  [type-arg]
build/lib/rfdetr/util/metrics.py:249: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/metrics.py:249: error: Missing type parameters for generic type "dict"  [type-arg]
build/lib/rfdetr/util/metrics.py:263: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:264: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:265: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:275: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:276: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:277: error: Call to untyped function "safe_index" in typed context  [no-untyped-call]
build/lib/rfdetr/util/metrics.py:287: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/metrics.py:287: note: Use "-> None" if function does not return a value
build/lib/rfdetr/util/box_ops.py:22: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/box_ops.py:33: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/box_ops.py:40: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/box_ops.py:56: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/box_ops.py:67: error: Call to untyped function "box_iou" in typed context  [no-untyped-call]
build/lib/rfdetr/util/box_ops.py:78: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/benchmark.py:45: error: "object" has no attribute "isCompleteTensor"  [attr-defined]
build/lib/rfdetr/util/benchmark.py:46: error: "object" has no attribute "type"  [attr-defined]
build/lib/rfdetr/util/benchmark.py:49: error: Returning Any from function declared to return "list[int]"  [no-any-return]
build/lib/rfdetr/util/benchmark.py:50: error: "object" has no attribute "type"  [attr-defined]
build/lib/rfdetr/util/benchmark.py:52: error: "object" has no attribute "type"  [attr-defined]
build/lib/rfdetr/util/benchmark.py:54: error: "object" has no attribute "type"  [attr-defined]
build/lib/rfdetr/util/benchmark.py:56: error: "object" has no attribute "type"  [attr-defined]
build/lib/rfdetr/util/benchmark.py:88: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/benchmark.py:103: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/benchmark.py:113: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/benchmark.py:120: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/benchmark.py:127: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/benchmark.py:135: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/benchmark.py:215: error: "object" has no attribute "toIValue"  [attr-defined]
build/lib/rfdetr/util/benchmark.py:223: error: "object" has no attribute "node"  [attr-defined]
build/lib/rfdetr/util/benchmark.py:307: error: Incompatible return value type (got "Counter[str]", expected "Number")  [return-value]
build/lib/rfdetr/util/benchmark.py:327: error: Incompatible return value type (got "Counter[str]", expected "Number")  [return-value]
build/lib/rfdetr/util/benchmark.py:346: error: Incompatible types in assignment (expression has type "floating[Union[_64Bit, Any]]", variable has type "int")  [assignment]
build/lib/rfdetr/util/benchmark.py:349: error: Incompatible types in assignment (expression has type "floating[Union[_64Bit, Any]]", variable has type "int")  [assignment]
build/lib/rfdetr/util/benchmark.py:351: error: Incompatible return value type (got "Counter[str]", expected "Number")  [return-value]
build/lib/rfdetr/util/benchmark.py:357: error: Missing type parameters for generic type "typing.Callable"  [type-arg]
build/lib/rfdetr/util/benchmark.py:459: error: Missing type parameters for generic type "typing.Callable"  [type-arg]
build/lib/rfdetr/util/benchmark.py:505: error: Module "torch.jit" does not explicitly export attribute "_get_trace_graph"  [attr-defined]
build/lib/rfdetr/util/benchmark.py:505: error: Call to untyped function "_get_trace_graph" in typed context  [no-untyped-call]
build/lib/rfdetr/util/benchmark.py:508: error: Need type annotation for "skipped_ops"  [var-annotated]
build/lib/rfdetr/util/benchmark.py:509: error: Need type annotation for "total_flop_counter"  [var-annotated]
build/lib/rfdetr/util/benchmark.py:523: error: Incompatible types in assignment (expression has type "list[Any]", variable has type "tuple[object, ...]")  [assignment]
build/lib/rfdetr/util/benchmark.py:541: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/benchmark.py:547: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/benchmark.py:548: error: Call to untyped function "warmup" in typed context  [no-untyped-call]
build/lib/rfdetr/util/benchmark.py:557: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/benchmark.py:567: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/benchmark.py:591: error: Call to untyped function "measure_time" in typed context  [no-untyped-call]
build/lib/rfdetr/util/benchmark.py:596: error: Call to untyped function "fmt_res" in typed context  [no-untyped-call]
build/lib/rfdetr/util/benchmark.py:598: error: Call to untyped function "fmt_res" in typed context  [no-untyped-call]
build/lib/rfdetr/util/benchmark.py:601: error: Incompatible types in assignment (expression has type "dict[str, Any]", variable has type "defaultdict[str, float]")  [assignment]
build/lib/rfdetr/util/benchmark.py:601: error: Call to untyped function "fmt_res" in typed context  [no-untyped-call]
build/lib/rfdetr/models/attention.py:66: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/attention.py:125: error: Incompatible types in assignment (expression has type "None", variable has type "Parameter")  [assignment]
build/lib/rfdetr/models/attention.py:126: error: Incompatible types in assignment (expression has type "None", variable has type "Parameter")  [assignment]
build/lib/rfdetr/models/attention.py:128: error: Incompatible types in assignment (expression has type "None", variable has type "Parameter")  [assignment]
build/lib/rfdetr/models/attention.py:129: error: Incompatible types in assignment (expression has type "None", variable has type "Parameter")  [assignment]
build/lib/rfdetr/models/attention.py:130: error: Incompatible types in assignment (expression has type "None", variable has type "Parameter")  [assignment]
build/lib/rfdetr/models/attention.py:134: error: Call to untyped function "_reset_parameters" in typed context  [no-untyped-call]
build/lib/rfdetr/models/attention.py:136: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/attention.py:136: note: Use "-> None" if function does not return a value
build/lib/rfdetr/models/attention.py:153: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/attention.py:158: error: Call to untyped function "__setstate__" in typed context  [no-untyped-call]
build/lib/rfdetr/models/attention.py:364: error: Subclass of "int" and "Tensor" cannot exist: would have incompatible method signatures  [unreachable]
build/lib/rfdetr/models/attention.py:366: error: Statement is unreachable  [unreachable]
build/lib/rfdetr/models/attention.py:372: error: Unsupported operand types for // ("None" and "int")  [operator]
build/lib/rfdetr/models/attention.py:372: error: Incompatible types in assignment (expression has type "int", variable has type "Tensor")  [assignment]
build/lib/rfdetr/models/attention.py:492: error: No overload variant of "view" of "TensorBase" matches argument types "int", "int", "int", "Tensor"  [call-overload]
build/lib/rfdetr/models/attention.py:492: note: Possible overload variants:
build/lib/rfdetr/models/attention.py:492: note:     def view(self, dtype: dtype) -> Tensor
build/lib/rfdetr/models/attention.py:492: note:     def view(self, size: Sequence[Union[int, SymInt]]) -> Tensor
build/lib/rfdetr/models/attention.py:492: note:     def view(self, *size: Union[int, SymInt]) -> Tensor
build/lib/rfdetr/models/attention.py:554: error: No overload variant of "view" of "TensorBase" matches argument types "int", "int", "Optional[Tensor]"  [call-overload]
build/lib/rfdetr/models/attention.py:554: note: Possible overload variants:
build/lib/rfdetr/models/attention.py:554: note:     def view(self, dtype: dtype) -> Tensor
build/lib/rfdetr/models/attention.py:554: note:     def view(self, size: Sequence[Union[int, SymInt]]) -> Tensor
build/lib/rfdetr/models/attention.py:554: note:     def view(self, *size: Union[int, SymInt]) -> Tensor
build/lib/rfdetr/models/attention.py:673: error: Incompatible return value type (got "tuple[Tensor, ...]", expected "list[Tensor]")  [return-value]
build/lib/rfdetr/models/attention.py:676: error: Call to untyped function "split" in typed context  [no-untyped-call]
build/lib/rfdetr/models/attention.py:680: error: Call to untyped function "split" in typed context  [no-untyped-call]
build/lib/rfdetr/models/attention.py:681: error: Incompatible return value type (got "tuple[Tensor, ...]", expected "list[Tensor]")  [return-value]
build/lib/rfdetr/models/attention.py:688: error: Incompatible return value type (got "tuple[Tensor, Tensor, Tensor]", expected "list[Tensor]")  [return-value]
build/lib/rfdetr/models/attention.py:737: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/attention.py:747: error: Name "rearrange" is not defined  [name-defined]
build/lib/rfdetr/models/attention.py:753: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/attention.py:754: error: Name "rearrange" is not defined  [name-defined]
build/lib/rfdetr/models/attention.py:757: error: Name "unpad_input" is not defined  [name-defined]
build/lib/rfdetr/models/attention.py:758: error: Name "rearrange" is not defined  [name-defined]
build/lib/rfdetr/models/attention.py:759: error: Name "unpad_input" is not defined  [name-defined]
build/lib/rfdetr/models/attention.py:760: error: Name "rearrange" is not defined  [name-defined]
build/lib/rfdetr/models/attention.py:762: error: Name "rearrange" is not defined  [name-defined]
build/lib/rfdetr/models/attention.py:763: error: Name "rearrange" is not defined  [name-defined]
build/lib/rfdetr/models/ops/functions/ms_deform_attn_func.py:22: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/ops/functions/ms_deform_attn_func.py:43: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Tensor]")  [assignment]
build/lib/rfdetr/models/backbone/projector.py:31: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/projector.py:38: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/projector.py:52: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/projector.py:67: error: Call to untyped function "LayerNorm" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:72: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/projector.py:77: error: Incompatible types in assignment (expression has type "ReLU", variable has type "SiLU")  [assignment]
build/lib/rfdetr/models/backbone/projector.py:79: error: Incompatible types in assignment (expression has type "LeakyReLU", variable has type "SiLU")  [assignment]
build/lib/rfdetr/models/backbone/projector.py:81: error: Incompatible types in assignment (expression has type "Identity", variable has type "SiLU")  [assignment]
build/lib/rfdetr/models/backbone/projector.py:90: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/projector.py:119: error: Call to untyped function "get_norm" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:119: error: Incompatible types in assignment (expression has type "Union[Any, BatchNorm2d]", variable has type "RMSNorm")  [assignment]
build/lib/rfdetr/models/backbone/projector.py:120: error: Call to untyped function "get_activation" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:122: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/projector.py:131: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/projector.py:146: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:147: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:152: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/projector.py:160: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/projector.py:166: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:167: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:171: error: Call to untyped function "Bottleneck" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:185: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/projector.py:198: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/projector.py:223: error: Need type annotation for "stages_sampling" (hint: "stages_sampling: list[<type>] = ...")  [var-annotated]
build/lib/rfdetr/models/backbone/projector.py:240: error: Call to untyped function "get_norm" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:266: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:274: error: Incompatible types in assignment (expression has type "Sequential", variable has type "list[Any]")  [assignment]
build/lib/rfdetr/models/backbone/projector.py:280: error: Call to untyped function "C2f" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:281: error: Call to untyped function "get_norm" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:283: error: Incompatible types in assignment (expression has type "Sequential", variable has type "list[Any]")  [assignment]
build/lib/rfdetr/models/backbone/projector.py:287: error: Argument 1 to "ModuleList" has incompatible type "list[list[Any]]"; expected "Optional[Iterable[Module]]"  [arg-type]
build/lib/rfdetr/models/backbone/projector.py:289: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/projector.py:317: error: Need type annotation for "stage_sampling"  [var-annotated]
build/lib/rfdetr/models/backbone/projector.py:317: error: Argument 1 to "enumerate" has incompatible type Module; expected "Iterable[Never]"  [arg-type]
build/lib/rfdetr/models/backbone/projector.py:319: error: Incompatible types in assignment (expression has type "Union[Tensor, Any]", variable has type "list[Any]")  [assignment]
build/lib/rfdetr/models/backbone/projector.py:327: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/projector.py:330: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:331: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:333: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:334: error: Call to untyped function "ConvX" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:335: error: Call to untyped function "get_norm" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/projector.py:337: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:27: error: Module "transformers.utils" does not explicitly export attribute "add_code_sample_docstrings"  [attr-defined]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:27: error: Module "transformers.utils" does not explicitly export attribute "add_start_docstrings"  [attr-defined]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:27: error: Module "transformers.utils" does not explicitly export attribute "add_start_docstrings_to_model_forward"  [attr-defined]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:27: error: Module "transformers.utils" does not explicitly export attribute "replace_return_docstrings"  [attr-defined]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:27: error: Module "transformers.utils" does not explicitly export attribute "torch_int"  [attr-defined]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:130: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:158: error: Call to untyped function "__init__" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:197: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:230: error: Returning Any from function declared to return "Tensor"  [no-any-return]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:248: error: Call to untyped function "Dinov2WithRegistersPatchEmbeddings" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:271: error: Module "torch.jit" does not explicitly export attribute "is_tracing"  [attr-defined]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:271: error: Call to untyped function "is_tracing" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:284: error: Call to untyped function "torch_int" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:294: error: Call to untyped function "torch_int" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:301: error: Module "torch.jit" does not explicitly export attribute "is_tracing"  [attr-defined]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:301: error: Call to untyped function "is_tracing" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:370: error: Item "None" of "Optional[Parameter]" has no attribute "expand"  [union-attr]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:381: error: Returning Any from function declared to return "Tensor"  [no-any-return]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:410: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:454: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:462: error: "Logger" has no attribute "warning_once"  [attr-defined]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:492: error: Incompatible return value type (got "tuple[Tensor, None]", expected "Union[tuple[Tensor, Tensor], tuple[Tensor]]")  [return-value]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:518: error: Need type annotation for "pruned_heads" (hint: "pruned_heads: set[<type>] = ...")  [var-annotated]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:524: error: Argument 1 to "find_pruneable_heads_and_indices" has incompatible type "set[int]"; expected "list[int]"  [arg-type]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:554: error: Returning Any from function declared to return "Union[tuple[Tensor, Tensor], tuple[Tensor]]"  [no-any-return]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:564: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:602: error: Argument 2 to "drop_path" has incompatible type "Optional[float]"; expected "float"  [arg-type]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:609: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:628: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:641: error: Returning Any from function declared to return "Tensor"  [no-any-return]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:674: error: Incompatible types in assignment (expression has type "Dinov2WithRegistersMLP", variable has type "Dinov2WithRegistersSwiGLUFFN")  [assignment]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:729: error: Returning Any from function declared to return "Union[tuple[Tensor, Tensor], tuple[Tensor]]"  [no-any-return]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:748: error: Missing type parameters for generic type "tuple"  [type-arg]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:754: error: Expected iterable as variadic argument  [misc]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:765: error: "Tensor" not callable  [operator]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:780: error: Expected iterable as variadic argument  [misc]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:783: error: Expected iterable as variadic argument  [misc]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:790: error: Argument "last_hidden_state" to "BaseModelOutput" has incompatible type "Tensor"; expected "Optional[FloatTensor]"  [arg-type]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:802: error: Incompatible types in assignment (expression has type "type[WindowedDinov2WithRegistersConfig]", base class "PreTrainedModel" defined the type as "None")  [assignment]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:806: error: Incompatible types in assignment (expression has type "list[str]", base class "PreTrainedModel" defined the type as "None")  [assignment]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:822: error: Statement is unreachable  [unreachable]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:877: error: Call to untyped function "add_start_docstrings" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:892: error: Call to untyped function "post_init" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:903: error: Item "Tensor" of "Union[Tensor, Module]" has no attribute "prune_heads"  [union-attr]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:903: error: "Tensor" not callable  [operator]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:905: error: Call to untyped function "add_start_docstrings_to_model_forward" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:905: error: Untyped decorator makes function "forward" untyped  [misc]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:906: error: Call to untyped function "add_code_sample_docstrings" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:906: error: Untyped decorator makes function "forward" untyped  [misc]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:921: error: Missing type parameters for generic type "tuple"  [type-arg]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:957: error: Returning Any from function declared to return "Union[tuple[Any, ...], BaseModelOutputWithPooling]"  [no-any-return]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:994: error: Call to untyped function "add_start_docstrings" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1016: error: Call to untyped function "post_init" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1018: error: Call to untyped function "add_start_docstrings_to_model_forward" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1018: error: Untyped decorator makes function "forward" untyped  [misc]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1019: error: Call to untyped function "add_code_sample_docstrings" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1019: error: Untyped decorator makes function "forward" untyped  [misc]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1033: error: Missing type parameters for generic type "tuple"  [type-arg]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1080: error: Incompatible types in assignment (expression has type "CrossEntropyLoss", variable has type "MSELoss")  [assignment]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1083: error: Incompatible types in assignment (expression has type "BCEWithLogitsLoss", variable has type "MSELoss")  [assignment]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1098: error: Call to untyped function "add_start_docstrings" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1119: error: Call to untyped function "post_init" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1124: error: Call to untyped function "add_start_docstrings_to_model_forward" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1124: error: Untyped decorator makes function "forward" untyped  [misc]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1125: error: Call to untyped function "replace_return_docstrings" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1125: error: Untyped decorator makes function "forward" untyped  [misc]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1223: error: Incompatible types in assignment (expression has type "tuple[Any]", variable has type "tuple[()]")  [assignment]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1230: error: Returning Any from function declared to return "BackboneOutput"  [no-any-return]
build/lib/rfdetr/models/backbone/dinov2_with_windowed_attn.py:1233: error: Argument "feature_maps" to "BackboneOutput" has incompatible type "tuple[()]"; expected "Optional[tuple[FloatTensor]]"  [arg-type]
build/lib/rfdetr/models/backbone/base.py:14: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/base.py:14: note: Use "-> None" if function does not return a value
build/lib/rfdetr/models/backbone/base.py:17: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/base.py:17: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:39: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:52: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:64: error: Need type annotation for "eval_imgs"  [var-annotated]
build/lib/rfdetr/deploy/benchmark.py:66: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:71: error: Call to untyped function "prepare" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:80: error: Call to untyped function "evaluate" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:84: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:84: note: Use "-> None" if function does not return a value
build/lib/rfdetr/deploy/benchmark.py:87: error: Call to untyped function "create_common_coco_eval" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:91: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:91: note: Use "-> None" if function does not return a value
build/lib/rfdetr/deploy/benchmark.py:95: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:95: note: Use "-> None" if function does not return a value
build/lib/rfdetr/deploy/benchmark.py:100: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:102: error: Call to untyped function "prepare_for_coco_detection" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:106: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:113: error: Call to untyped function "convert_to_xywh" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:131: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:140: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:177: error: Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[Any]")  [assignment]
build/lib/rfdetr/deploy/benchmark.py:182: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:187: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:193: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:198: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:201: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:206: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:216: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:221: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:225: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:233: error: Name "box_xyxy_to_cxcywh" is not defined  [name-defined]
build/lib/rfdetr/deploy/benchmark.py:240: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:244: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:271: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:272: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:272: error: Call to untyped function "Normalize" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:273: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:275: error: Call to untyped function "SquareResize" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:281: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:292: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:303: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:316: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:319: error: Call to untyped function "load_image" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:322: error: Call to untyped function "infer_transforms" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:336: error: Call to untyped function "post_process" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:353: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:356: error: Call to untyped function "load_image" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:359: error: Call to untyped function "infer_transforms" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:376: error: Call to untyped function "post_process" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:395: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:410: error: Call to untyped function "load_engine" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:414: error: Call to untyped function "get_bindings" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:419: error: Call to untyped function "get_input_names" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:420: error: Call to untyped function "get_output_names" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:428: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:436: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:442: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:451: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:460: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:473: error: Statement is unreachable  [unreachable]
build/lib/rfdetr/deploy/benchmark.py:488: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:494: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:502: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:504: error: Call to untyped function "run_sync" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:506: error: Call to untyped function "run_async" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:508: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:508: note: Use "-> None" if function does not return a value
build/lib/rfdetr/deploy/benchmark.py:516: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:517: error: "None" has no attribute "reset"  [attr-defined]
build/lib/rfdetr/deploy/benchmark.py:518: error: "None" has no attribute "__enter__"  [attr-defined]
build/lib/rfdetr/deploy/benchmark.py:518: error: "None" has no attribute "__exit__"  [attr-defined]
build/lib/rfdetr/deploy/benchmark.py:521: error: "None" has no attribute "total"  [attr-defined]
build/lib/rfdetr/deploy/benchmark.py:523: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:552: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:552: note: Use "-> None" if function does not return a value
build/lib/rfdetr/deploy/benchmark.py:557: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:560: error: Call to untyped function "time" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:563: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:564: error: Call to untyped function "time" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:566: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:566: note: Use "-> None" if function does not return a value
build/lib/rfdetr/deploy/benchmark.py:571: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:579: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/benchmark.py:583: error: Call to untyped function "get_image_list" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:594: error: Call to untyped function "CocoEvaluator" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:596: error: Call to untyped function "TimeProfiler" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:600: error: Call to untyped function "infer_onnx" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:611: error: Call to untyped function "infer_engine" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:627: error: Call to untyped function "parser_args" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/benchmark.py:628: error: Call to untyped function "main" in typed context  [no-untyped-call]
rfdetr/models/position_encoding.py:34: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/position_encoding.py:46: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/position_encoding.py:46: note: Use "-> None" if function does not return a value
rfdetr/models/position_encoding.py:49: error: Cannot assign to a method  [method-assign]
rfdetr/models/position_encoding.py:49: error: Incompatible types in assignment (expression has type "Callable[[Tensor, Any], Any]", variable has type "Callable[[NestedTensor, Any], Any]")  [assignment]
rfdetr/models/position_encoding.py:51: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/position_encoding.py:51: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/position_encoding.py:82: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/position_encoding.py:82: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/position_encoding.py:117: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/position_encoding.py:121: error: Call to untyped function "reset_parameters" in typed context  [no-untyped-call]
rfdetr/models/position_encoding.py:124: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/position_encoding.py:124: note: Use "-> None" if function does not return a value
rfdetr/models/position_encoding.py:127: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/position_encoding.py:127: note: Use "-> None" if function does not return a value
rfdetr/models/position_encoding.py:131: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/position_encoding.py:153: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/position_encoding.py:157: error: Call to untyped function "PositionEmbeddingSine" in typed context  [no-untyped-call]
rfdetr/models/position_encoding.py:159: error: Call to untyped function "PositionEmbeddingLearned" in typed context  [no-untyped-call]
rfdetr/models/matcher.py:61: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/matcher.py:92: error: Call to untyped function "generalized_box_iou" in typed context  [no-untyped-call]
rfdetr/models/matcher.py:92: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
rfdetr/models/matcher.py:133: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/dinov2.py:42: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/dinov2.py:53: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/dinov2.py:90: error: Incompatible types in assignment (expression has type "list[int]", variable has type "set[int]")  [assignment]
rfdetr/models/backbone/dinov2.py:92: error: Call to untyped function "get_config" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2.py:119: error: Call to untyped function "WindowedDinov2WithRegistersConfig" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2.py:126: error: Call to untyped function "WindowedDinov2WithRegistersConfig" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2.py:145: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/backbone/dinov2.py:151: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/dinov2.py:182: error: Call to untyped function "make_new_interpolated_pos_encoding" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2.py:191: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/dinov2.py:203: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/dinov2.py:241: error: Call to untyped function "DinoV2" in typed context  [no-untyped-call]
rfdetr/models/backbone/dinov2.py:242: error: Call to untyped function "export" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:40: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:53: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:65: error: Need type annotation for "eval_imgs"  [var-annotated]
rfdetr/deploy/benchmark.py:67: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:72: error: Call to untyped function "prepare" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:81: error: Call to untyped function "evaluate" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:85: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:85: note: Use "-> None" if function does not return a value
rfdetr/deploy/benchmark.py:88: error: Call to untyped function "create_common_coco_eval" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:92: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:92: note: Use "-> None" if function does not return a value
rfdetr/deploy/benchmark.py:96: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:96: note: Use "-> None" if function does not return a value
rfdetr/deploy/benchmark.py:101: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:103: error: Call to untyped function "prepare_for_coco_detection" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:107: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:114: error: Call to untyped function "convert_to_xywh" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:132: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:141: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:178: error: Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[Any]")  [assignment]
rfdetr/deploy/benchmark.py:183: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:188: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:194: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:199: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:202: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:207: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:219: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:224: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:228: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:236: error: Call to untyped function "box_xyxy_to_cxcywh" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:243: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:247: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:274: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:275: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:275: error: Call to untyped function "Normalize" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:276: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:278: error: Call to untyped function "SquareResize" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:284: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:295: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:306: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:322: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:325: error: Call to untyped function "load_image" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:328: error: Call to untyped function "infer_transforms" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:342: error: Call to untyped function "post_process" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:359: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:362: error: Call to untyped function "load_image" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:365: error: Call to untyped function "infer_transforms" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:382: error: Call to untyped function "post_process" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:401: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/deploy/benchmark.py:416: error: Call to untyped function "load_engine" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:420: error: Call to untyped function "get_bindings" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:425: error: Call to untyped function "get_input_names" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:426: error: Call to untyped function "get_output_names" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:434: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:442: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:448: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:457: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:466: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:479: error: Statement is unreachable  [unreachable]
rfdetr/deploy/benchmark.py:494: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:500: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:508: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:510: error: Call to untyped function "run_sync" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:512: error: Call to untyped function "run_async" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:514: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:514: note: Use "-> None" if function does not return a value
rfdetr/deploy/benchmark.py:522: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:523: error: "None" has no attribute "reset"  [attr-defined]
rfdetr/deploy/benchmark.py:524: error: "None" has no attribute "__enter__"  [attr-defined]
rfdetr/deploy/benchmark.py:524: error: "None" has no attribute "__exit__"  [attr-defined]
rfdetr/deploy/benchmark.py:527: error: "None" has no attribute "total"  [attr-defined]
rfdetr/deploy/benchmark.py:529: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:558: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:558: note: Use "-> None" if function does not return a value
rfdetr/deploy/benchmark.py:563: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:566: error: Call to untyped function "time" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:569: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:570: error: Call to untyped function "time" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:572: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:572: note: Use "-> None" if function does not return a value
rfdetr/deploy/benchmark.py:577: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:585: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/benchmark.py:589: error: Call to untyped function "get_image_list" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:600: error: Call to untyped function "CocoEvaluator" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:602: error: Call to untyped function "TimeProfiler" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:606: error: Call to untyped function "infer_onnx" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:617: error: Call to untyped function "infer_engine" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:633: error: Call to untyped function "parser_args" in typed context  [no-untyped-call]
rfdetr/deploy/benchmark.py:634: error: Call to untyped function "main" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:41: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:84: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:103: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:124: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:128: error: Call to untyped function "get_size_with_aspect_ratio" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:131: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:162: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:165: error: Call to untyped function "get_size" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:171: error: Call to untyped function "update_target_for_resize" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:175: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:189: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:192: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:194: error: Call to untyped function "crop" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:202: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:202: error: Missing type parameters for generic type "dict"  [type-arg]
rfdetr/datasets/transforms.py:206: error: Call to untyped function "crop" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:210: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:213: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:218: error: Call to untyped function "crop" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:222: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:225: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:227: error: Call to untyped function "hflip" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:232: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:237: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:239: error: Call to untyped function "resize" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:243: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:247: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:275: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:278: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:281: error: Call to untyped function "pad" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:285: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:290: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:295: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:328: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:331: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:340: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:365: error: Call to untyped function "apply_image" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:372: error: Call to untyped function "apply_bbox" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:386: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:399: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:413: error: Call to untyped function "Pad" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:424: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:429: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:436: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:441: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:444: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:449: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:453: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:461: error: Call to untyped function "box_xyxy_to_cxcywh" in typed context  [no-untyped-call]
rfdetr/datasets/transforms.py:468: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:471: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/transforms.py:476: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:38: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:49: error: Need type annotation for "eval_imgs"  [var-annotated]
rfdetr/datasets/coco_eval.py:51: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:56: error: Call to untyped function "prepare" in typed context  [no-untyped-call]
rfdetr/datasets/coco_eval.py:77: error: Call to untyped function "evaluate" in typed context  [no-untyped-call]
rfdetr/datasets/coco_eval.py:82: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:82: note: Use "-> None" if function does not return a value
rfdetr/datasets/coco_eval.py:85: error: Call to untyped function "create_common_coco_eval" in typed context  [no-untyped-call]
rfdetr/datasets/coco_eval.py:89: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:89: note: Use "-> None" if function does not return a value
rfdetr/datasets/coco_eval.py:93: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:93: note: Use "-> None" if function does not return a value
rfdetr/datasets/coco_eval.py:98: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:100: error: Call to untyped function "prepare_for_coco_detection" in typed context  [no-untyped-call]
rfdetr/datasets/coco_eval.py:102: error: Call to untyped function "prepare_for_coco_segmentation" in typed context  [no-untyped-call]
rfdetr/datasets/coco_eval.py:104: error: Call to untyped function "prepare_for_coco_keypoint" in typed context  [no-untyped-call]
rfdetr/datasets/coco_eval.py:108: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:115: error: Call to untyped function "convert_to_xywh" in typed context  [no-untyped-call]
rfdetr/datasets/coco_eval.py:132: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:186: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:193: error: Call to untyped function "convert_to_xywh" in typed context  [no-untyped-call]
rfdetr/datasets/coco_eval.py:213: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:218: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:219: error: Call to untyped function "all_gather" in typed context  [no-untyped-call]
rfdetr/datasets/coco_eval.py:220: error: Call to untyped function "all_gather" in typed context  [no-untyped-call]
rfdetr/datasets/coco_eval.py:230: error: Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[Any]")  [assignment]
rfdetr/datasets/coco_eval.py:235: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[ellipsis, Any]"  [call-overload]
rfdetr/datasets/coco_eval.py:235: note: Possible overload variants:
rfdetr/datasets/coco_eval.py:235: note:     def __getitem__(self, SupportsIndex, /) -> Any
rfdetr/datasets/coco_eval.py:235: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Any]
rfdetr/datasets/coco_eval.py:240: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:241: error: Call to untyped function "merge" in typed context  [no-untyped-call]
rfdetr/datasets/coco_eval.py:256: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/datasets/coco_eval.py:296: error: Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[Any]")  [assignment]
build/lib/rfdetr/models/position_encoding.py:34: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/position_encoding.py:46: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/position_encoding.py:46: note: Use "-> None" if function does not return a value
build/lib/rfdetr/models/position_encoding.py:49: error: Cannot assign to a method  [method-assign]
build/lib/rfdetr/models/position_encoding.py:49: error: Incompatible types in assignment (expression has type "Callable[[Tensor, Any], Any]", variable has type "Callable[[NestedTensor, Any], Any]")  [assignment]
build/lib/rfdetr/models/position_encoding.py:51: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/position_encoding.py:51: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/position_encoding.py:82: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/position_encoding.py:82: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/position_encoding.py:117: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/position_encoding.py:121: error: Call to untyped function "reset_parameters" in typed context  [no-untyped-call]
build/lib/rfdetr/models/position_encoding.py:124: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/position_encoding.py:124: note: Use "-> None" if function does not return a value
build/lib/rfdetr/models/position_encoding.py:127: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/position_encoding.py:127: note: Use "-> None" if function does not return a value
build/lib/rfdetr/models/position_encoding.py:131: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/position_encoding.py:153: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/position_encoding.py:157: error: Call to untyped function "PositionEmbeddingSine" in typed context  [no-untyped-call]
build/lib/rfdetr/models/position_encoding.py:159: error: Call to untyped function "PositionEmbeddingLearned" in typed context  [no-untyped-call]
build/lib/rfdetr/models/matcher.py:61: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/matcher.py:92: error: Call to untyped function "generalized_box_iou" in typed context  [no-untyped-call]
build/lib/rfdetr/models/matcher.py:92: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
build/lib/rfdetr/models/matcher.py:133: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2.py:42: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2.py:53: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2.py:90: error: Incompatible types in assignment (expression has type "list[int]", variable has type "set[int]")  [assignment]
build/lib/rfdetr/models/backbone/dinov2.py:92: error: Call to untyped function "get_config" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2.py:98: error: Call to untyped function "WindowedDinov2WithRegistersConfig" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2.py:105: error: Call to untyped function "WindowedDinov2WithRegistersConfig" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2.py:124: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2.py:130: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2.py:161: error: Call to untyped function "make_new_interpolated_pos_encoding" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2.py:170: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2.py:182: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/dinov2.py:198: error: Call to untyped function "DinoV2" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/dinov2.py:199: error: Call to untyped function "export" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/transforms.py:41: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:84: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:103: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:106: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:126: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:130: error: Call to untyped function "get_size_with_aspect_ratio" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/transforms.py:132: error: Call to untyped function "get_size" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/transforms.py:165: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:179: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:182: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:184: error: Call to untyped function "crop" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/transforms.py:192: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:192: error: Missing type parameters for generic type "dict"  [type-arg]
build/lib/rfdetr/datasets/transforms.py:196: error: Call to untyped function "crop" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/transforms.py:200: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:203: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:208: error: Call to untyped function "crop" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/transforms.py:212: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:215: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:217: error: Call to untyped function "hflip" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/transforms.py:222: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:227: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:229: error: Call to untyped function "resize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/transforms.py:233: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:237: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:265: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:268: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:271: error: Call to untyped function "pad" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/transforms.py:275: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:280: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:285: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:318: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:321: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:330: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:355: error: Call to untyped function "apply_image" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/transforms.py:362: error: Call to untyped function "apply_bbox" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/transforms.py:376: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:389: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:403: error: Call to untyped function "Pad" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/transforms.py:414: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:419: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:426: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:431: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:434: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:439: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:443: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:451: error: Call to untyped function "box_xyxy_to_cxcywh" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/transforms.py:458: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:461: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/transforms.py:466: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:38: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:49: error: Need type annotation for "eval_imgs"  [var-annotated]
build/lib/rfdetr/datasets/coco_eval.py:51: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:56: error: Call to untyped function "prepare" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco_eval.py:65: error: Call to untyped function "evaluate" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco_eval.py:69: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:69: note: Use "-> None" if function does not return a value
build/lib/rfdetr/datasets/coco_eval.py:72: error: Call to untyped function "create_common_coco_eval" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco_eval.py:76: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:76: note: Use "-> None" if function does not return a value
build/lib/rfdetr/datasets/coco_eval.py:80: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:80: note: Use "-> None" if function does not return a value
build/lib/rfdetr/datasets/coco_eval.py:85: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:87: error: Call to untyped function "prepare_for_coco_detection" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco_eval.py:89: error: Call to untyped function "prepare_for_coco_segmentation" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco_eval.py:91: error: Call to untyped function "prepare_for_coco_keypoint" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco_eval.py:95: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:102: error: Call to untyped function "convert_to_xywh" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco_eval.py:119: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:171: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:178: error: Call to untyped function "convert_to_xywh" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco_eval.py:198: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:203: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:204: error: Call to untyped function "all_gather" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco_eval.py:205: error: Call to untyped function "all_gather" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco_eval.py:215: error: Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[Any]")  [assignment]
build/lib/rfdetr/datasets/coco_eval.py:220: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[ellipsis, Any]"  [call-overload]
build/lib/rfdetr/datasets/coco_eval.py:220: note: Possible overload variants:
build/lib/rfdetr/datasets/coco_eval.py:220: note:     def __getitem__(self, SupportsIndex, /) -> Any
build/lib/rfdetr/datasets/coco_eval.py:220: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Any]
build/lib/rfdetr/datasets/coco_eval.py:225: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:226: error: Call to untyped function "merge" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco_eval.py:241: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco_eval.py:281: error: Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[Any]")  [assignment]
rfdetr/models/ops/modules/ms_deform_attn.py:26: error: Module "rfdetr.models.ops.functions" does not explicitly export attribute "ms_deform_attn_core_pytorch"  [attr-defined]
rfdetr/models/ops/modules/ms_deform_attn.py:53: error: Call to untyped function "_is_power_of_2" in typed context  [no-untyped-call]
rfdetr/models/ops/modules/ms_deform_attn.py:73: error: Call to untyped function "_reset_parameters" in typed context  [no-untyped-call]
rfdetr/models/ops/modules/ms_deform_attn.py:161: error: Call to untyped function "ms_deform_attn_core_pytorch" in typed context  [no-untyped-call]
rfdetr/models/backbone/backbone.py:37: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/backbone/backbone.py:41: error: Missing type parameters for generic type "list"  [type-arg]
rfdetr/models/backbone/backbone.py:44: error: Missing type parameters for generic type "list"  [type-arg]
rfdetr/models/backbone/backbone.py:45: error: Missing type parameters for generic type "list"  [type-arg]
rfdetr/models/backbone/backbone.py:55: error: Call to untyped function "__init__" in typed context  [no-untyped-call]
rfdetr/models/backbone/backbone.py:75: error: Call to untyped function "DinoV2" in typed context  [no-untyped-call]
rfdetr/models/backbone/backbone.py:90: error: Argument 1 to "len" has incompatible type "Optional[list[Any]]"; expected "Sized"  [arg-type]
rfdetr/models/backbone/backbone.py:93: error: Argument 1 to "sorted" has incompatible type "Optional[list[Any]]"; expected "Iterable[Any]"  [arg-type]
rfdetr/models/backbone/backbone.py:98: error: Call to untyped function "MultiScaleProjector" in typed context  [no-untyped-call]
rfdetr/models/backbone/backbone.py:108: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/backbone/backbone.py:108: note: Use "-> None" if function does not return a value
rfdetr/models/backbone/backbone.py:111: error: Cannot assign to a method  [method-assign]
rfdetr/models/backbone/backbone.py:111: error: Incompatible types in assignment (expression has type "Callable[[Tensor], Any]", variable has type "Callable[[NestedTensor], Any]")  [assignment]
rfdetr/models/backbone/backbone.py:117: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/backbone/backbone.py:131: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/backbone/backbone.py:143: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/backbone/backbone.py:143: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/backbone/backbone.py:152: error: Call to untyped function "get_dinov2_lr_decay_rate" in typed context  [no-untyped-call]
rfdetr/models/backbone/backbone.py:159: error: Call to untyped function "get_dinov2_weight_decay_rate" in typed context  [no-untyped-call]
rfdetr/models/backbone/backbone.py:168: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/backbone.py:188: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/fixed_size_transforms.py:23: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/fixed_size_transforms.py:29: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/fixed_size_transforms.py:101: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/fixed_size_transforms.py:106: error: Call to untyped function "FixedSizeTransform" in typed context  [no-untyped-call]
rfdetr/datasets/fixed_size_transforms.py:108: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/fixed_size_transforms.py:119: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/fixed_size_transforms.py:135: error: Call to untyped function "Normalize" in typed context  [no-untyped-call]
rfdetr/datasets/fixed_size_transforms.py:136: error: Call to untyped function "FixedSizeTransform" in typed context  [no-untyped-call]
rfdetr/datasets/fixed_size_transforms.py:143: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/datasets/fixed_size_transforms.py:144: error: Call to untyped function "RandomHorizontalFlip" in typed context  [no-untyped-call]
rfdetr/datasets/fixed_size_transforms.py:145: error: Call to untyped function "RandomSelect" in typed context  [no-untyped-call]
rfdetr/datasets/fixed_size_transforms.py:146: error: Call to untyped function "RandomResize" in typed context  [no-untyped-call]
rfdetr/datasets/fixed_size_transforms.py:147: error: Call to untyped function "RandomResize" in typed context  [no-untyped-call]
rfdetr/datasets/fixed_size_transforms.py:154: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/models/ops/modules/ms_deform_attn.py:26: error: Module "rfdetr.models.ops.functions" does not explicitly export attribute "ms_deform_attn_core_pytorch"  [attr-defined]
build/lib/rfdetr/models/ops/modules/ms_deform_attn.py:29: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/ops/modules/ms_deform_attn.py:38: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/ops/modules/ms_deform_attn.py:53: error: Call to untyped function "_is_power_of_2" in typed context  [no-untyped-call]
build/lib/rfdetr/models/ops/modules/ms_deform_attn.py:73: error: Call to untyped function "_reset_parameters" in typed context  [no-untyped-call]
build/lib/rfdetr/models/ops/modules/ms_deform_attn.py:77: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/ops/modules/ms_deform_attn.py:77: note: Use "-> None" if function does not return a value
build/lib/rfdetr/models/ops/modules/ms_deform_attn.py:81: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/ops/modules/ms_deform_attn.py:81: note: Use "-> None" if function does not return a value
build/lib/rfdetr/models/ops/modules/ms_deform_attn.py:101: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/ops/modules/ms_deform_attn.py:161: error: Call to untyped function "ms_deform_attn_core_pytorch" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/backbone.py:37: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/backbone/backbone.py:41: error: Missing type parameters for generic type "list"  [type-arg]
build/lib/rfdetr/models/backbone/backbone.py:44: error: Missing type parameters for generic type "list"  [type-arg]
build/lib/rfdetr/models/backbone/backbone.py:45: error: Missing type parameters for generic type "list"  [type-arg]
build/lib/rfdetr/models/backbone/backbone.py:55: error: Call to untyped function "__init__" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/backbone.py:75: error: Call to untyped function "DinoV2" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/backbone.py:90: error: Argument 1 to "len" has incompatible type "Optional[list[Any]]"; expected "Sized"  [arg-type]
build/lib/rfdetr/models/backbone/backbone.py:93: error: Argument 1 to "sorted" has incompatible type "Optional[list[Any]]"; expected "Iterable[Any]"  [arg-type]
build/lib/rfdetr/models/backbone/backbone.py:98: error: Call to untyped function "MultiScaleProjector" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/backbone.py:108: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/backbone.py:108: note: Use "-> None" if function does not return a value
build/lib/rfdetr/models/backbone/backbone.py:111: error: Cannot assign to a method  [method-assign]
build/lib/rfdetr/models/backbone/backbone.py:111: error: Incompatible types in assignment (expression has type "Callable[[Tensor], Any]", variable has type "Callable[[NestedTensor], Any]")  [assignment]
build/lib/rfdetr/models/backbone/backbone.py:117: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/backbone.py:131: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/backbone.py:143: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/backbone.py:143: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/backbone/backbone.py:152: error: Call to untyped function "get_dinov2_lr_decay_rate" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/backbone.py:159: error: Call to untyped function "get_dinov2_weight_decay_rate" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/backbone.py:168: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/backbone.py:188: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/fixed_size_transforms.py:32: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/fixed_size_transforms.py:75: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/fixed_size_transforms.py:90: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/fixed_size_transforms.py:92: error: Call to untyped function "Normalize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/fixed_size_transforms.py:96: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/fixed_size_transforms.py:97: error: Call to untyped function "RandomHorizontalFlip" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/fixed_size_transforms.py:103: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/models/backbone/__init__.py:22: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/__init__.py:26: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/backbone/__init__.py:34: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/backbone/__init__.py:34: note: Use "-> None" if function does not return a value
rfdetr/models/backbone/__init__.py:37: error: Cannot assign to a method  [method-assign]
rfdetr/models/backbone/__init__.py:37: error: Incompatible types in assignment (expression has type "Callable[[Tensor], Any]", variable has type "Callable[[NestedTensor], Any]")  [assignment]
rfdetr/models/backbone/__init__.py:41: error: Argument 2 to "isinstance" has incompatible type "<typing special form>"; expected "_ClassInfo"  [arg-type]
rfdetr/models/backbone/__init__.py:47: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/backbone/__init__.py:55: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/backbone/__init__.py:84: error: Call to untyped function "build_position_encoding" in typed context  [no-untyped-call]
rfdetr/models/backbone/__init__.py:104: error: Call to untyped function "Joiner" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:33: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco.py:60: error: Class cannot subclass "CocoDetection" (has type "Any")  [misc]
rfdetr/datasets/coco.py:61: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco.py:68: error: Cannot determine type of "ids"  [has-type]
rfdetr/datasets/coco.py:70: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco.py:81: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco.py:93: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment]
rfdetr/datasets/coco.py:94: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[slice[None, None, None], slice[int, None, None]]"  [call-overload]
rfdetr/datasets/coco.py:94: note: Possible overload variants:
rfdetr/datasets/coco.py:94: note:     def __getitem__(self, SupportsIndex, /) -> Any
rfdetr/datasets/coco.py:94: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Any]
rfdetr/datasets/coco.py:94: error: No overload variant of "__setitem__" of "list" matches argument types "tuple[slice[None, None, None], slice[int, None, None]]", "Any"  [call-overload]
rfdetr/datasets/coco.py:94: note:     def __setitem__(self, SupportsIndex, Any, /) -> None
rfdetr/datasets/coco.py:94: note:     def __setitem__(self, slice[Any, Any, Any], Iterable[Any], /) -> None
rfdetr/datasets/coco.py:94: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[slice[None, None, None], slice[None, int, None]]"  [call-overload]
rfdetr/datasets/coco.py:95: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[slice[None, None, None], slice[int, None, int]]"  [call-overload]
rfdetr/datasets/coco.py:95: note: Possible overload variants:
rfdetr/datasets/coco.py:95: note:     def __getitem__(self, SupportsIndex, /) -> Any
rfdetr/datasets/coco.py:95: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Any]
rfdetr/datasets/coco.py:96: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[slice[None, None, None], slice[int, None, int]]"  [call-overload]
rfdetr/datasets/coco.py:96: note: Possible overload variants:
rfdetr/datasets/coco.py:96: note:     def __getitem__(self, SupportsIndex, /) -> Any
rfdetr/datasets/coco.py:96: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Any]
rfdetr/datasets/coco.py:99: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment]
rfdetr/datasets/coco.py:121: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment]
rfdetr/datasets/coco.py:123: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[slice[None, None, None], int]"  [call-overload]
rfdetr/datasets/coco.py:123: note: Possible overload variants:
rfdetr/datasets/coco.py:123: note:     def __getitem__(self, SupportsIndex, /) -> Any
rfdetr/datasets/coco.py:123: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Any]
rfdetr/datasets/coco.py:150: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco.py:151: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:154: error: Call to untyped function "Normalize" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:170: error: Call to untyped function "compute_multi_scale_scales" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:174: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:176: error: Call to untyped function "RandomHorizontalFlip" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:177: error: Call to untyped function "RandomSelect" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:178: error: Call to untyped function "RandomResize" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:179: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:181: error: Call to untyped function "RandomResize" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:183: error: Call to untyped function "RandomResize" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:192: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:194: error: Call to untyped function "RandomResize" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:199: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:201: error: Call to untyped function "SquareResize" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:209: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco.py:214: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:217: error: Call to untyped function "Normalize" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:233: error: Call to untyped function "compute_multi_scale_scales" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:237: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:239: error: Call to untyped function "RandomHorizontalFlip" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:240: error: Call to untyped function "RandomSelect" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:241: error: Call to untyped function "SquareResize" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:242: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:244: error: Call to untyped function "RandomResize" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:246: error: Call to untyped function "SquareResize" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:255: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:257: error: Call to untyped function "SquareResize" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:262: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:264: error: Call to untyped function "SquareResize" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:272: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco.py:313: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:316: error: Call to untyped function "make_coco_transforms_square_div_64" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:325: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:328: error: Call to untyped function "make_coco_transforms" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:339: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/coco.py:356: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:359: error: Call to untyped function "make_coco_transforms_square_div_64" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:365: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
rfdetr/datasets/coco.py:368: error: Call to untyped function "make_coco_transforms" in typed context  [no-untyped-call]
rfdetr/datasets/__init__.py:21: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/__init__.py:29: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/datasets/__init__.py:87: error: Call to untyped function "make_training_dimensions_transforms" in typed context  [no-untyped-call]
rfdetr/datasets/__init__.py:88: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
rfdetr/datasets/__init__.py:96: error: Call to untyped function "build" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/__init__.py:21: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/__init__.py:25: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/__init__.py:33: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/__init__.py:33: note: Use "-> None" if function does not return a value
build/lib/rfdetr/models/backbone/__init__.py:36: error: Cannot assign to a method  [method-assign]
build/lib/rfdetr/models/backbone/__init__.py:36: error: Incompatible types in assignment (expression has type "Callable[[Tensor], Any]", variable has type "Callable[[NestedTensor], Any]")  [assignment]
build/lib/rfdetr/models/backbone/__init__.py:40: error: Argument 2 to "isinstance" has incompatible type "<typing special form>"; expected "_ClassInfo"  [arg-type]
build/lib/rfdetr/models/backbone/__init__.py:46: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/__init__.py:54: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/backbone/__init__.py:83: error: Call to untyped function "build_position_encoding" in typed context  [no-untyped-call]
build/lib/rfdetr/models/backbone/__init__.py:103: error: Call to untyped function "Joiner" in typed context  [no-untyped-call]
modified_coco_build.py:5: error: Module "rfdetr.datasets.transforms" has no attribute "make_coco_transforms"  [attr-defined]
modified_coco_build.py:5: error: Module "rfdetr.datasets.transforms" has no attribute "make_coco_transforms_square_div_64"  [attr-defined]
modified_coco_build.py:8: error: Function is missing a type annotation  [no-untyped-def]
modified_coco_build.py:62: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
modified_coco_build.py:70: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
tests/test_resize_div_56.py:43: error: Call to untyped function "build" in typed context  [no-untyped-call]
tests/test_resize_div_56.py:43: error: Too many arguments for "build"  [call-arg]
tests/test_resize_div_56.py:61: error: Call to untyped function "main" in typed context  [no-untyped-call]
tests/test_custom_coco_annotations.py:37: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_custom_coco_annotations.py:37: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/test_custom_coco_annotations.py:57: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_custom_coco_annotations.py:57: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/test_custom_coco_annotations.py:80: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_custom_coco_annotations.py:80: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/test_custom_coco_annotations.py:97: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_custom_coco_annotations.py:97: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/check_coco_mask_loading.py:74: error: Module "matplotlib.pyplot" does not explicitly export attribute "Rectangle"  [attr-defined]
tests/check_coco_mask_loading.py:185: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
tests/check_coco_mask_loading.py:198: error: Call to untyped function "examine_original_annotations" in typed context  [no-untyped-call]
tests/check_coco_mask_loading.py:206: error: Call to untyped function "visualize_masks" in typed context  [no-untyped-call]
tests/check_coco_mask_loading.py:238: error: Call to untyped function "parse_args" in typed context  [no-untyped-call]
tests/check_coco_mask_loading.py:239: error: Call to untyped function "convert_coco_check" in typed context  [no-untyped-call]
rfdetr/util/get_param_dicts.py:14: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/get_param_dicts.py:36: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/util/get_param_dicts.py:50: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/util/get_param_dicts.py:50: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/transformer.py:28: error: Module "rfdetr.models.ops.modules" does not explicitly export attribute "MSDeformAttn"  [attr-defined]
rfdetr/models/transformer.py:34: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/transformer.py:42: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/transformer.py:48: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/transformer.py:77: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/transformer.py:145: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/transformer.py:167: error: Call to untyped function "TransformerDecoderLayer" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:185: error: Call to untyped function (unknown) in typed context  [no-untyped-call]
rfdetr/models/transformer.py:187: error: Call to untyped function "TransformerDecoder" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:204: error: Call to untyped function "MLP" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:207: error: Call to untyped function "_reset_parameters" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:218: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/transformer.py:218: note: Use "-> None" if function does not return a value
rfdetr/models/transformer.py:221: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/transformer.py:221: note: Use "-> None" if function does not return a value
rfdetr/models/transformer.py:227: error: Call to untyped function "_reset_parameters" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:229: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/transformer.py:238: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/transformer.py:240: error: Need type annotation for "mask_flatten"  [var-annotated]
rfdetr/models/transformer.py:243: error: Need type annotation for "valid_ratios"  [var-annotated]
rfdetr/models/transformer.py:255: error: Item "None" of "Optional[list[Any]]" has no attribute "append"  [union-attr]
rfdetr/models/transformer.py:258: error: Incompatible types in assignment (expression has type "Tensor", variable has type "Optional[list[Any]]")  [assignment]
rfdetr/models/transformer.py:259: error: Incompatible types in assignment (expression has type "Tensor", variable has type "Optional[list[Any]]")  [assignment]
rfdetr/models/transformer.py:259: error: Call to untyped function "get_valid_ratio" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:260: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment]
rfdetr/models/transformer.py:261: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[tuple[Any, Any]]")  [assignment]
rfdetr/models/transformer.py:263: error: "list[tuple[Any, Any]]" has no attribute "new_zeros"  [attr-defined]
rfdetr/models/transformer.py:263: error: "list[tuple[Any, Any]]" has no attribute "prod"  [attr-defined]
rfdetr/models/transformer.py:267: error: Call to untyped function "gen_encoder_output_proposals" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:318: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Tensor]")  [assignment]
rfdetr/models/transformer.py:320: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Tensor]")  [assignment]
rfdetr/models/transformer.py:321: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Tensor]")  [assignment]
rfdetr/models/transformer.py:327: error: "list[Tensor]" has no attribute "shape"  [attr-defined]
rfdetr/models/transformer.py:332: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[ellipsis, slice[int, None, None]]"  [call-overload]
rfdetr/models/transformer.py:332: note: Possible overload variants:
rfdetr/models/transformer.py:332: note:     def __getitem__(self, SupportsIndex, /) -> Tensor
rfdetr/models/transformer.py:332: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Tensor]
rfdetr/models/transformer.py:333: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[ellipsis, slice[None, int, None]]"  [call-overload]
rfdetr/models/transformer.py:333: note: Possible overload variants:
rfdetr/models/transformer.py:333: note:     def __getitem__(self, SupportsIndex, /) -> Tensor
rfdetr/models/transformer.py:333: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Tensor]
rfdetr/models/transformer.py:334: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[ellipsis, slice[int, None, None]]"  [call-overload]
rfdetr/models/transformer.py:334: note: Possible overload variants:
rfdetr/models/transformer.py:334: note:     def __getitem__(self, SupportsIndex, /) -> Tensor
rfdetr/models/transformer.py:334: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Tensor]
rfdetr/models/transformer.py:351: error: "list[Any]" has no attribute "to"  [attr-defined]
rfdetr/models/transformer.py:359: error: "list[Tensor]" has no attribute "sigmoid"  [attr-defined]
rfdetr/models/transformer.py:363: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/transformer.py:374: error: Call to untyped function "_get_clones" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:382: error: Call to untyped function "MLP" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:386: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/transformer.py:386: note: Use "-> None" if function does not return a value
rfdetr/models/transformer.py:389: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/transformer.py:401: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/transformer.py:401: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/transformer.py:421: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/transformer.py:426: error: Call to untyped function "gen_sineembed_for_position" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:432: error: List item 0 has incompatible type "Optional[Tensor]"; expected "Tensor"  [list-item]
rfdetr/models/transformer.py:432: error: List item 1 has incompatible type "Optional[Tensor]"; expected "Tensor"  [list-item]
rfdetr/models/transformer.py:434: error: Call to untyped function "gen_sineembed_for_position" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:443: error: Call to untyped function "get_reference" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:447: error: Call to untyped function "get_reference" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:448: error: Item "None" of "Optional[Tensor]" has no attribute "sigmoid"  [union-attr]
rfdetr/models/transformer.py:455: error: Call to untyped function "get_reference" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:459: error: Call to untyped function "get_reference" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:460: error: Item "None" of "Optional[Tensor]" has no attribute "sigmoid"  [union-attr]
rfdetr/models/transformer.py:486: error: "Tensor" not callable  [operator]
rfdetr/models/transformer.py:487: error: Call to untyped function "refpoints_refine" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:510: error: Statement is unreachable  [unreachable]
rfdetr/models/transformer.py:516: error: Argument 1 to "stack" has incompatible type "list[Optional[Tensor]]"; expected "Union[tuple[Tensor, ...], list[Tensor], None]"  [arg-type]
rfdetr/models/transformer.py:519: error: Statement is unreachable  [unreachable]
rfdetr/models/transformer.py:525: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/transformer.py:548: error: Call to untyped function "MSDeformAttn" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:565: error: Call to untyped function "_get_activation_fn" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:569: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/transformer.py:569: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/transformer.py:572: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/transformer.py:572: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/transformer.py:627: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/transformer.py:627: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/transformer.py:660: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/transformer.py:664: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/transformer.py:665: error: Call to untyped function "Transformer" in typed context  [no-untyped-call]
rfdetr/models/transformer.py:683: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/get_param_dicts.py:14: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/get_param_dicts.py:35: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/util/get_param_dicts.py:48: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/util/get_param_dicts.py:48: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:28: error: Module "rfdetr.models.ops.modules" does not explicitly export attribute "MSDeformAttn"  [attr-defined]
build/lib/rfdetr/models/transformer.py:34: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:42: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:48: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:77: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:143: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:166: error: Call to untyped function "TransformerDecoderLayer" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:184: error: Call to untyped function (unknown) in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:186: error: Call to untyped function "TransformerDecoder" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:203: error: Call to untyped function "_reset_parameters" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:214: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:214: note: Use "-> None" if function does not return a value
build/lib/rfdetr/models/transformer.py:217: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:217: note: Use "-> None" if function does not return a value
build/lib/rfdetr/models/transformer.py:223: error: Call to untyped function "_reset_parameters" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:225: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:234: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:236: error: Need type annotation for "mask_flatten"  [var-annotated]
build/lib/rfdetr/models/transformer.py:239: error: Need type annotation for "valid_ratios"  [var-annotated]
build/lib/rfdetr/models/transformer.py:251: error: Item "None" of "Optional[list[Any]]" has no attribute "append"  [union-attr]
build/lib/rfdetr/models/transformer.py:254: error: Incompatible types in assignment (expression has type "Tensor", variable has type "Optional[list[Any]]")  [assignment]
build/lib/rfdetr/models/transformer.py:255: error: Incompatible types in assignment (expression has type "Tensor", variable has type "Optional[list[Any]]")  [assignment]
build/lib/rfdetr/models/transformer.py:255: error: Call to untyped function "get_valid_ratio" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:256: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment]
build/lib/rfdetr/models/transformer.py:257: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[tuple[Any, Any]]")  [assignment]
build/lib/rfdetr/models/transformer.py:259: error: "list[tuple[Any, Any]]" has no attribute "new_zeros"  [attr-defined]
build/lib/rfdetr/models/transformer.py:259: error: "list[tuple[Any, Any]]" has no attribute "prod"  [attr-defined]
build/lib/rfdetr/models/transformer.py:263: error: Call to untyped function "gen_encoder_output_proposals" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:274: error: Value of type "Union[Tensor, Module]" is not indexable  [index]
build/lib/rfdetr/models/transformer.py:274: error: "Tensor" not callable  [operator]
build/lib/rfdetr/models/transformer.py:278: error: Value of type "Union[Tensor, Module]" is not indexable  [index]
build/lib/rfdetr/models/transformer.py:278: error: "Tensor" not callable  [operator]
build/lib/rfdetr/models/transformer.py:293: error: Value of type "Union[Tensor, Module]" is not indexable  [index]
build/lib/rfdetr/models/transformer.py:293: error: "Tensor" not callable  [operator]
build/lib/rfdetr/models/transformer.py:320: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Tensor]")  [assignment]
build/lib/rfdetr/models/transformer.py:322: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Tensor]")  [assignment]
build/lib/rfdetr/models/transformer.py:323: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Tensor]")  [assignment]
build/lib/rfdetr/models/transformer.py:328: error: "list[Tensor]" has no attribute "shape"  [attr-defined]
build/lib/rfdetr/models/transformer.py:333: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[ellipsis, slice[int, None, None]]"  [call-overload]
build/lib/rfdetr/models/transformer.py:333: note: Possible overload variants:
build/lib/rfdetr/models/transformer.py:333: note:     def __getitem__(self, SupportsIndex, /) -> Tensor
build/lib/rfdetr/models/transformer.py:333: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Tensor]
build/lib/rfdetr/models/transformer.py:334: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[ellipsis, slice[None, int, None]]"  [call-overload]
build/lib/rfdetr/models/transformer.py:334: note: Possible overload variants:
build/lib/rfdetr/models/transformer.py:334: note:     def __getitem__(self, SupportsIndex, /) -> Tensor
build/lib/rfdetr/models/transformer.py:334: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Tensor]
build/lib/rfdetr/models/transformer.py:336: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[ellipsis, slice[int, None, None]]"  [call-overload]
build/lib/rfdetr/models/transformer.py:336: note: Possible overload variants:
build/lib/rfdetr/models/transformer.py:336: note:     def __getitem__(self, SupportsIndex, /) -> Tensor
build/lib/rfdetr/models/transformer.py:336: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Tensor]
build/lib/rfdetr/models/transformer.py:354: error: "list[Any]" has no attribute "to"  [attr-defined]
build/lib/rfdetr/models/transformer.py:362: error: "list[Tensor]" has no attribute "sigmoid"  [attr-defined]
build/lib/rfdetr/models/transformer.py:367: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:378: error: Call to untyped function "_get_clones" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:386: error: Call to untyped function "MLP" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:390: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:390: note: Use "-> None" if function does not return a value
build/lib/rfdetr/models/transformer.py:393: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:405: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:405: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:425: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:430: error: Call to untyped function "gen_sineembed_for_position" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:436: error: List item 0 has incompatible type "Optional[Tensor]"; expected "Tensor"  [list-item]
build/lib/rfdetr/models/transformer.py:436: error: List item 1 has incompatible type "Optional[Tensor]"; expected "Tensor"  [list-item]
build/lib/rfdetr/models/transformer.py:438: error: Call to untyped function "gen_sineembed_for_position" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:447: error: Call to untyped function "get_reference" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:451: error: Call to untyped function "get_reference" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:452: error: Item "None" of "Optional[Tensor]" has no attribute "sigmoid"  [union-attr]
build/lib/rfdetr/models/transformer.py:459: error: Call to untyped function "get_reference" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:463: error: Call to untyped function "get_reference" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:464: error: Item "None" of "Optional[Tensor]" has no attribute "sigmoid"  [union-attr]
build/lib/rfdetr/models/transformer.py:490: error: "Tensor" not callable  [operator]
build/lib/rfdetr/models/transformer.py:491: error: Call to untyped function "refpoints_refine" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:514: error: Statement is unreachable  [unreachable]
build/lib/rfdetr/models/transformer.py:520: error: Argument 1 to "stack" has incompatible type "list[Optional[Tensor]]"; expected "Union[tuple[Tensor, ...], list[Tensor], None]"  [arg-type]
build/lib/rfdetr/models/transformer.py:523: error: Statement is unreachable  [unreachable]
build/lib/rfdetr/models/transformer.py:529: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:552: error: Call to untyped function "MSDeformAttn" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:569: error: Call to untyped function "_get_activation_fn" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:573: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:573: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:576: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:576: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:631: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:631: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:664: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:668: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/transformer.py:674: error: Call to untyped function "Transformer" in typed context  [no-untyped-call]
build/lib/rfdetr/models/transformer.py:693: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco.py:33: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco.py:60: error: Class cannot subclass "CocoDetection" (has type "Any")  [misc]
build/lib/rfdetr/datasets/coco.py:61: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco.py:68: error: Cannot determine type of "ids"  [has-type]
build/lib/rfdetr/datasets/coco.py:70: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco.py:81: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco.py:93: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment]
build/lib/rfdetr/datasets/coco.py:94: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[slice[None, None, None], slice[int, None, None]]"  [call-overload]
build/lib/rfdetr/datasets/coco.py:94: note: Possible overload variants:
build/lib/rfdetr/datasets/coco.py:94: note:     def __getitem__(self, SupportsIndex, /) -> Any
build/lib/rfdetr/datasets/coco.py:94: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Any]
build/lib/rfdetr/datasets/coco.py:94: error: No overload variant of "__setitem__" of "list" matches argument types "tuple[slice[None, None, None], slice[int, None, None]]", "Any"  [call-overload]
build/lib/rfdetr/datasets/coco.py:94: note:     def __setitem__(self, SupportsIndex, Any, /) -> None
build/lib/rfdetr/datasets/coco.py:94: note:     def __setitem__(self, slice[Any, Any, Any], Iterable[Any], /) -> None
build/lib/rfdetr/datasets/coco.py:94: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[slice[None, None, None], slice[None, int, None]]"  [call-overload]
build/lib/rfdetr/datasets/coco.py:95: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[slice[None, None, None], slice[int, None, int]]"  [call-overload]
build/lib/rfdetr/datasets/coco.py:95: note: Possible overload variants:
build/lib/rfdetr/datasets/coco.py:95: note:     def __getitem__(self, SupportsIndex, /) -> Any
build/lib/rfdetr/datasets/coco.py:95: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Any]
build/lib/rfdetr/datasets/coco.py:96: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[slice[None, None, None], slice[int, None, int]]"  [call-overload]
build/lib/rfdetr/datasets/coco.py:96: note: Possible overload variants:
build/lib/rfdetr/datasets/coco.py:96: note:     def __getitem__(self, SupportsIndex, /) -> Any
build/lib/rfdetr/datasets/coco.py:96: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Any]
build/lib/rfdetr/datasets/coco.py:99: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment]
build/lib/rfdetr/datasets/coco.py:121: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment]
build/lib/rfdetr/datasets/coco.py:123: error: No overload variant of "__getitem__" of "list" matches argument type "tuple[slice[None, None, None], int]"  [call-overload]
build/lib/rfdetr/datasets/coco.py:123: note: Possible overload variants:
build/lib/rfdetr/datasets/coco.py:123: note:     def __getitem__(self, SupportsIndex, /) -> Any
build/lib/rfdetr/datasets/coco.py:123: note:     def __getitem__(self, slice[Any, Any, Any], /) -> list[Any]
build/lib/rfdetr/datasets/coco.py:150: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco.py:151: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:154: error: Call to untyped function "Normalize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:161: error: Call to untyped function "compute_multi_scale_scales" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:165: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:167: error: Call to untyped function "RandomHorizontalFlip" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:168: error: Call to untyped function "RandomSelect" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:169: error: Call to untyped function "RandomResize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:170: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:172: error: Call to untyped function "RandomResize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:174: error: Call to untyped function "RandomResize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:183: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:185: error: Call to untyped function "RandomResize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:190: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:192: error: Call to untyped function "SquareResize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:200: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco.py:205: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:208: error: Call to untyped function "Normalize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:215: error: Call to untyped function "compute_multi_scale_scales" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:219: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:221: error: Call to untyped function "RandomHorizontalFlip" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:222: error: Call to untyped function "RandomSelect" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:223: error: Call to untyped function "SquareResize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:224: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:226: error: Call to untyped function "RandomResize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:228: error: Call to untyped function "SquareResize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:237: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:239: error: Call to untyped function "SquareResize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:244: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:246: error: Call to untyped function "SquareResize" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:254: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco.py:295: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:298: error: Call to untyped function "make_coco_transforms_square_div_64" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:307: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:310: error: Call to untyped function "make_coco_transforms" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:321: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/coco.py:338: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:341: error: Call to untyped function "make_coco_transforms_square_div_64" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:347: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/coco.py:350: error: Call to untyped function "make_coco_transforms" in typed context  [no-untyped-call]
tests/test_mixed_precision_eval.py:21: error: Call to untyped function "MLP" in typed context  [no-untyped-call]
tests/test_mixed_precision_eval.py:43: error: Call to untyped function "gen_sineembed_for_position" in typed context  [no-untyped-call]
tests/test_mixed_precision_eval.py:55: error: Call to untyped function "gen_sineembed_for_position" in typed context  [no-untyped-call]
tests/test_mixed_precision_eval.py:67: error: Call to untyped function "MLP" in typed context  [no-untyped-call]
tests/test_mixed_precision_eval.py:72: error: Call to untyped function "gen_sineembed_for_position" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:48: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:75: error: Call to untyped function "MLP" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:101: error: Argument 1 to "constant_" has incompatible type "Union[Tensor, Module]"; expected "Tensor"  [arg-type]
rfdetr/models/lwdetr.py:102: error: Argument 1 to "constant_" has incompatible type "Union[Tensor, Module]"; expected "Tensor"  [arg-type]
rfdetr/models/lwdetr.py:105: error: Call to untyped function "MLP" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:106: error: Argument 1 to "constant_" has incompatible type "Union[Tensor, Module]"; expected "Tensor"  [arg-type]
rfdetr/models/lwdetr.py:107: error: Argument 1 to "constant_" has incompatible type "Union[Tensor, Module]"; expected "Tensor"  [arg-type]
rfdetr/models/lwdetr.py:119: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:136: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:136: note: Use "-> None" if function does not return a value
rfdetr/models/lwdetr.py:139: error: Cannot assign to a method  [method-assign]
rfdetr/models/lwdetr.py:139: error: Incompatible types in assignment (expression has type "Callable[[Any], Any]", variable has type "Callable[[NestedTensor, Any], Any]")  [assignment]
rfdetr/models/lwdetr.py:143: error: Argument 2 to "isinstance" has incompatible type "<typing special form>"; expected "_ClassInfo"  [arg-type]
rfdetr/models/lwdetr.py:149: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:149: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/lwdetr.py:220: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment]
rfdetr/models/lwdetr.py:224: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:250: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:265: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:276: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:289: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:323: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:330: error: Call to untyped function "_get_src_permutation_idx" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:340: error: Call to untyped function "box_iou" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:341: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:342: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:371: error: Call to untyped function "box_iou" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:372: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:373: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:408: error: Call to untyped function "box_iou" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:409: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:410: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:455: error: Call to untyped function "accuracy" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:459: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:472: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:478: error: Call to untyped function "_get_src_permutation_idx" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:488: error: Call to untyped function "generalized_box_iou" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:489: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:495: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:501: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:507: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:513: error: Call to untyped function "_get_src_permutation_idx" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:514: error: Call to untyped function "_get_tgt_permutation_idx" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:602: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:610: error: Cannot call function of unknown type  [operator]
rfdetr/models/lwdetr.py:612: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:629: error: Incompatible types in assignment (expression has type "Tensor", variable has type "int")  [assignment]
rfdetr/models/lwdetr.py:632: error: Call to untyped function "is_dist_avail_and_initialized" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:634: error: Incompatible types in assignment (expression has type "Union[int, float]", variable has type "int")  [assignment]
rfdetr/models/lwdetr.py:634: error: Call to untyped function "get_world_size" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:639: error: Call to untyped function "get_loss" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:650: error: Call to untyped function "get_loss" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:668: error: Call to untyped function "get_loss" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:675: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:675: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/lwdetr.py:703: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:703: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/lwdetr.py:715: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:715: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/lwdetr.py:730: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/lwdetr.py:735: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:756: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:812: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:817: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:890: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:898: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:904: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:935: error: Call to untyped function "build_backbone" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:987: error: Call to untyped function "build_transformer" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:1001: error: Call to untyped function "LWDETR" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:1005: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/models/lwdetr.py:1007: error: Call to untyped function "build_matcher" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:1036: error: Call to untyped function "SetCriterion" in typed context  [no-untyped-call]
rfdetr/models/lwdetr.py:1052: error: Dict entry 0 has incompatible type "str": "PostProcessSegm"; expected "str": "PostProcess"  [dict-item]
rfdetr/models/lwdetr.py:1052: error: Call to untyped function "PostProcessSegm" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:47: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:75: error: Call to untyped function "MLP" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:101: error: Argument 1 to "constant_" has incompatible type "Union[Tensor, Module]"; expected "Tensor"  [arg-type]
build/lib/rfdetr/models/lwdetr.py:102: error: Argument 1 to "constant_" has incompatible type "Union[Tensor, Module]"; expected "Tensor"  [arg-type]
build/lib/rfdetr/models/lwdetr.py:105: error: Call to untyped function "MLP" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:106: error: Argument 1 to "constant_" has incompatible type "Union[Tensor, Module]"; expected "Tensor"  [arg-type]
build/lib/rfdetr/models/lwdetr.py:107: error: Argument 1 to "constant_" has incompatible type "Union[Tensor, Module]"; expected "Tensor"  [arg-type]
build/lib/rfdetr/models/lwdetr.py:121: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:138: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:138: note: Use "-> None" if function does not return a value
build/lib/rfdetr/models/lwdetr.py:141: error: Cannot assign to a method  [method-assign]
build/lib/rfdetr/models/lwdetr.py:141: error: Incompatible types in assignment (expression has type "Callable[[Any], Any]", variable has type "Callable[[NestedTensor, Any], Any]")  [assignment]
build/lib/rfdetr/models/lwdetr.py:145: error: Argument 2 to "isinstance" has incompatible type "<typing special form>"; expected "_ClassInfo"  [arg-type]
build/lib/rfdetr/models/lwdetr.py:151: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:151: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:222: error: Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment]
build/lib/rfdetr/models/lwdetr.py:226: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:252: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:267: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:278: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:291: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:325: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:332: error: Call to untyped function "_get_src_permutation_idx" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:342: error: Call to untyped function "box_iou" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:343: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:344: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:373: error: Call to untyped function "box_iou" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:374: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:375: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:410: error: Call to untyped function "box_iou" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:411: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:412: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:457: error: Call to untyped function "accuracy" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:461: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:474: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:480: error: Call to untyped function "_get_src_permutation_idx" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:490: error: Call to untyped function "generalized_box_iou" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:491: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:497: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:503: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:509: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:515: error: Call to untyped function "_get_src_permutation_idx" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:516: error: Call to untyped function "_get_tgt_permutation_idx" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:604: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:612: error: Cannot call function of unknown type  [operator]
build/lib/rfdetr/models/lwdetr.py:614: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:631: error: Incompatible types in assignment (expression has type "Tensor", variable has type "int")  [assignment]
build/lib/rfdetr/models/lwdetr.py:634: error: Call to untyped function "is_dist_avail_and_initialized" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:636: error: Incompatible types in assignment (expression has type "Union[int, float]", variable has type "int")  [assignment]
build/lib/rfdetr/models/lwdetr.py:636: error: Call to untyped function "get_world_size" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:641: error: Call to untyped function "get_loss" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:652: error: Call to untyped function "get_loss" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:670: error: Call to untyped function "get_loss" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:677: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:677: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:705: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:705: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:717: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:717: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:732: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:737: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:758: error: Call to untyped function "box_cxcywh_to_xyxy" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:814: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:819: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:892: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:900: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:906: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:918: error: Call to untyped function "build_backbone" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:949: error: Call to untyped function "build_transformer" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:951: error: Call to untyped function "LWDETR" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:965: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/models/lwdetr.py:967: error: Call to untyped function "build_matcher" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:986: error: Call to untyped function "SetCriterion" in typed context  [no-untyped-call]
build/lib/rfdetr/models/lwdetr.py:1002: error: Dict entry 0 has incompatible type "str": "PostProcessSegm"; expected "str": "PostProcess"  [dict-item]
build/lib/rfdetr/models/lwdetr.py:1002: error: Call to untyped function "PostProcessSegm" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/o365.py:21: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/o365.py:32: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/o365.py:35: error: Call to untyped function "make_coco_transforms_square_div_64" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/o365.py:43: error: Call to untyped function "CocoDetection" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/o365.py:46: error: Call to untyped function "make_coco_transforms" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/o365.py:56: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/o365.py:58: error: Call to untyped function "build_o365_raw" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/o365.py:61: error: Call to untyped function "build_o365_raw" in typed context  [no-untyped-call]
rfdetr/models/segmentation.py:105: error: Returning Any from function declared to return "Tensor"  [no-any-return]
rfdetr/models/segmentation.py:131: error: Item "Tensor" of "Union[Tensor, Module]" has no attribute "num_channels"  [union-attr]
rfdetr/models/segmentation.py:132: error: Item "Tensor" of "Union[Tensor, Module]" has no attribute "d_model"  [union-attr]
rfdetr/models/segmentation.py:141: error: Argument "dim" to "MaskHeadSmallConv" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
rfdetr/models/segmentation.py:141: error: Argument "context_dim" to "MaskHeadSmallConv" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
rfdetr/models/segmentation.py:147: error: Argument 1 to "Conv2d" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
rfdetr/models/segmentation.py:147: error: Argument 2 to "Conv2d" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
rfdetr/models/segmentation.py:148: error: Argument 1 to "Conv2d" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
rfdetr/models/segmentation.py:148: error: Argument 2 to "Conv2d" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
rfdetr/models/segmentation.py:149: error: Argument 1 to "Conv2d" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
rfdetr/models/segmentation.py:149: error: Argument 2 to "Conv2d" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
rfdetr/models/segmentation.py:153: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/segmentation.py:170: error: "Tensor" not callable  [operator]
rfdetr/models/segmentation.py:178: error: Item "Tensor" of "Union[Tensor, Module]" has no attribute "encoder"  [union-attr]
rfdetr/models/segmentation.py:178: error: "Tensor" not callable  [operator]
rfdetr/models/segmentation.py:210: error: Name "torch" is not defined  [name-defined]
rfdetr/models/segmentation.py:212: error: Name "torch" is not defined  [name-defined]
rfdetr/models/segmentation.py:228: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/models/segmentation.py:228: error: Missing type parameters for generic type "dict"  [type-arg]
rfdetr/models/segmentation.py:243: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
build/lib/rfdetr/models/segmentation.py:105: error: Returning Any from function declared to return "Tensor"  [no-any-return]
build/lib/rfdetr/models/segmentation.py:131: error: Item "Tensor" of "Union[Tensor, Module]" has no attribute "num_channels"  [union-attr]
build/lib/rfdetr/models/segmentation.py:132: error: Item "Tensor" of "Union[Tensor, Module]" has no attribute "d_model"  [union-attr]
build/lib/rfdetr/models/segmentation.py:141: error: Argument "dim" to "MaskHeadSmallConv" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
build/lib/rfdetr/models/segmentation.py:141: error: Argument "context_dim" to "MaskHeadSmallConv" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
build/lib/rfdetr/models/segmentation.py:147: error: Argument 1 to "Conv2d" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
build/lib/rfdetr/models/segmentation.py:147: error: Argument 2 to "Conv2d" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
build/lib/rfdetr/models/segmentation.py:148: error: Argument 1 to "Conv2d" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
build/lib/rfdetr/models/segmentation.py:148: error: Argument 2 to "Conv2d" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
build/lib/rfdetr/models/segmentation.py:149: error: Argument 1 to "Conv2d" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
build/lib/rfdetr/models/segmentation.py:149: error: Argument 2 to "Conv2d" has incompatible type "Union[Any, Tensor, Module]"; expected "int"  [arg-type]
build/lib/rfdetr/models/segmentation.py:153: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/segmentation.py:170: error: "Tensor" not callable  [operator]
build/lib/rfdetr/models/segmentation.py:178: error: Item "Tensor" of "Union[Tensor, Module]" has no attribute "encoder"  [union-attr]
build/lib/rfdetr/models/segmentation.py:178: error: "Tensor" not callable  [operator]
build/lib/rfdetr/models/segmentation.py:210: error: Name "torch" is not defined  [name-defined]
build/lib/rfdetr/models/segmentation.py:212: error: Name "torch" is not defined  [name-defined]
build/lib/rfdetr/models/segmentation.py:228: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/models/segmentation.py:228: error: Missing type parameters for generic type "dict"  [type-arg]
build/lib/rfdetr/models/segmentation.py:243: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/__init__.py:21: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/__init__.py:29: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/datasets/__init__.py:31: error: Call to untyped function "build" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/__init__.py:33: error: Call to untyped function "build_o365" in typed context  [no-untyped-call]
build/lib/rfdetr/datasets/__init__.py:35: error: Call to untyped function "build_roboflow" in typed context  [no-untyped-call]
tests/test_checkpoint_loading.py:14: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
tests/test_checkpoint_loading.py:46: error: Unexpected keyword argument "resolution" for "ModelConfig"  [call-arg]
tests/test_checkpoint_loading.py:84: error: Call to untyped function "_create_model_config" in typed context  [no-untyped-call]
tests/test_checkpoint_loading.py:85: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
tests/test_checkpoint_loading.py:105: error: Call to untyped function "_create_model_config" in typed context  [no-untyped-call]
tests/test_checkpoint_loading.py:106: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
rfdetr/engine.py:37: error: Module "torch.amp" does not explicitly export attribute "GradScaler"  [attr-defined]
rfdetr/engine.py:37: error: Module "torch.amp" does not explicitly export attribute "autocast"  [attr-defined]
rfdetr/engine.py:40: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/engine.py:54: error: Call to untyped function "evaluate" in typed context  [no-untyped-call]
rfdetr/engine.py:71: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/engine.py:82: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/engine.py:82: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/engine.py:86: error: Missing type parameters for generic type "Iterable"  [type-arg]
rfdetr/engine.py:92: error: Incompatible default for argument "ema_m" (default has type "None", argument has type Module)  [assignment]
rfdetr/engine.py:92: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
rfdetr/engine.py:92: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
rfdetr/engine.py:93: error: Missing type parameters for generic type "dict"  [type-arg]
rfdetr/engine.py:97: error: Missing type parameters for generic type "Callable"  [type-arg]
rfdetr/engine.py:101: error: Missing type parameters for generic type "Iterable"  [type-arg]
rfdetr/engine.py:107: error: Call to untyped function "MetricLogger" in typed context  [no-untyped-call]
rfdetr/engine.py:108: error: Call to untyped function "add_meter" in typed context  [no-untyped-call]
rfdetr/engine.py:108: error: Call to untyped function "SmoothedValue" in typed context  [no-untyped-call]
rfdetr/engine.py:109: error: Call to untyped function "add_meter" in typed context  [no-untyped-call]
rfdetr/engine.py:109: error: Call to untyped function "SmoothedValue" in typed context  [no-untyped-call]
rfdetr/engine.py:115: error: Call to untyped function "get_world_size" in typed context  [no-untyped-call]
rfdetr/engine.py:118: error: Unexpected keyword argument "device_type" for "GradScaler"  [call-arg]
/home/georgepearse/.local/lib/python3.9/site-packages/torch/amp/grad_scaler.py:123: note: "GradScaler" defined here
rfdetr/engine.py:123: error: Argument 1 to "len" has incompatible type "Iterable[Any]"; expected "Sized"  [arg-type]
rfdetr/engine.py:129: error: Call to untyped function "log_every" in typed context  [no-untyped-call]
rfdetr/engine.py:132: error: Unsupported operand types for <= ("int" and "None")  [operator]
rfdetr/engine.py:132: note: Left operand is of type "Optional[int]"
rfdetr/engine.py:142: error: Unsupported operand types for < ("int" and "None")  [operator]
rfdetr/engine.py:142: note: Left operand is of type "Optional[int]"
rfdetr/engine.py:143: error: Unsupported operand types for % ("None" and "int")  [operator]
rfdetr/engine.py:143: note: Left operand is of type "Optional[int]"
rfdetr/engine.py:145: error: Call to untyped function "do_evaluation_during_training" in typed context  [no-untyped-call]
rfdetr/engine.py:164: error: Value of type "Optional[defaultdict[str, list[Callable[..., Any]]]]" is not indexable  [index]
rfdetr/engine.py:168: error: Item "Tensor" of "Union[Tensor, Module]" has no attribute "update_drop_path"  [union-attr]
rfdetr/engine.py:168: error: "Tensor" not callable  [operator]
rfdetr/engine.py:170: error: "Tensor" not callable  [operator]
rfdetr/engine.py:173: error: Item "Tensor" of "Union[Tensor, Module]" has no attribute "update_dropout"  [union-attr]
rfdetr/engine.py:173: error: "Tensor" not callable  [operator]
rfdetr/engine.py:175: error: "Tensor" not callable  [operator]
rfdetr/engine.py:178: error: Unsupported operand types for + ("None" and "int")  [operator]
rfdetr/engine.py:178: note: Left operand is of type "Optional[int]"
rfdetr/engine.py:190: error: Call to untyped function "get_autocast_args" in typed context  [no-untyped-call]
rfdetr/engine.py:195: error: Value of type "Union[Tensor, Module]" is not indexable  [index]
rfdetr/engine.py:197: error: Unsupported right operand type for in ("Union[Tensor, Module]")  [operator]
rfdetr/engine.py:200: error: No overload variant of "scale" of "GradScaler" matches argument type "int"  [call-overload]
rfdetr/engine.py:200: note: Possible overload variants:
rfdetr/engine.py:200: note:     def scale(self, outputs: Tensor) -> Tensor
rfdetr/engine.py:200: note:     def scale(self, outputs: list[Tensor]) -> list[Tensor]
rfdetr/engine.py:200: note:     def scale(self, outputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]
rfdetr/engine.py:200: note:     def scale(self, outputs: Iterable[Tensor]) -> Iterable[Tensor]
rfdetr/engine.py:203: error: Call to untyped function "reduce_dict" in typed context  [no-untyped-call]
rfdetr/engine.py:206: error: Value of type "Union[Tensor, Module]" is not indexable  [index]
rfdetr/engine.py:206: error: Unsupported right operand type for in ("Union[Tensor, Module]")  [operator]
rfdetr/engine.py:227: error: "Tensor" not callable  [operator]
rfdetr/engine.py:228: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/engine.py:231: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/engine.py:232: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/engine.py:234: error: Call to untyped function "synchronize_between_processes" in typed context  [no-untyped-call]
rfdetr/engine.py:244: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/engine.py:254: error: Call to untyped function "MetricLogger" in typed context  [no-untyped-call]
rfdetr/engine.py:255: error: Call to untyped function "add_meter" in typed context  [no-untyped-call]
rfdetr/engine.py:255: error: Call to untyped function "SmoothedValue" in typed context  [no-untyped-call]
rfdetr/engine.py:259: error: Call to untyped function "CocoEvaluator" in typed context  [no-untyped-call]
rfdetr/engine.py:269: error: Call to untyped function "log_every" in typed context  [no-untyped-call]
rfdetr/engine.py:284: error: Call to untyped function "get_autocast_args" in typed context  [no-untyped-call]
rfdetr/engine.py:303: error: Call to untyped function "reduce_dict" in typed context  [no-untyped-call]
rfdetr/engine.py:308: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/engine.py:313: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/engine.py:319: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/engine.py:322: error: Call to untyped function "synchronize_between_processes" in typed context  [no-untyped-call]
rfdetr/engine.py:325: error: Call to untyped function "synchronize_between_processes" in typed context  [no-untyped-call]
rfdetr/engine.py:329: error: Call to untyped function "accumulate" in typed context  [no-untyped-call]
rfdetr/engine.py:330: error: Call to untyped function "summarize" in typed context  [no-untyped-call]
rfdetr/deploy/export.py:28: error: Module "rfdetr.deploy._onnx" does not explicitly export attribute "OnnxOptimizer"  [attr-defined]
rfdetr/deploy/export.py:29: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
rfdetr/deploy/export.py:39: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/deploy/export.py:46: error: Incompatible return value type (got "CompletedProcess[str]", expected "int")  [return-value]
rfdetr/deploy/export.py:53: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/export.py:60: error: Call to untyped function (unknown) in typed context  [no-untyped-call]
rfdetr/deploy/export.py:62: error: Call to untyped function (unknown) in typed context  [no-untyped-call]
rfdetr/deploy/export.py:64: error: Call to untyped function (unknown) in typed context  [no-untyped-call]
rfdetr/deploy/export.py:75: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/export.py:111: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/deploy/export.py:111: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/deploy/export.py:120: error: Call to untyped function "OnnxOptimizer" in typed context  [no-untyped-call]
rfdetr/deploy/export.py:121: error: Call to untyped function "info" in typed context  [no-untyped-call]
rfdetr/deploy/export.py:122: error: Call to untyped function "common_opt" in typed context  [no-untyped-call]
rfdetr/deploy/export.py:123: error: Call to untyped function "info" in typed context  [no-untyped-call]
rfdetr/deploy/export.py:139: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/deploy/export.py:171: error: Call to untyped function "parse_trtexec_output" in typed context  [no-untyped-call]
rfdetr/deploy/export.py:171: error: "int" has no attribute "stdout"  [attr-defined]
rfdetr/deploy/export.py:174: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/export.py:232: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/export.py:238: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/deploy/export.py:239: error: Call to untyped function "get_sha" in typed context  [no-untyped-call]
rfdetr/deploy/export.py:256: error: Call to untyped function "get_rank" in typed context  [no-untyped-call]
rfdetr/deploy/export.py:261: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
rfdetr/deploy/export.py:278: error: Call to untyped function "no_batch_norm" in typed context  [no-untyped-call]
rfdetr/deploy/export.py:282: error: Call to untyped function "make_infer_image" in typed context  [no-untyped-call]
rfdetr/deploy/export.py:282: error: Missing positional argument "batch_size" in call to "make_infer_image"  [call-arg]
rfdetr/deploy/export.py:301: error: Call to untyped function "export_onnx" in typed context  [no-untyped-call]
rfdetr/deploy/export.py:307: error: "trtexec" does not return a value (it only ever returns None)  [func-returns-value]
rfdetr/main.py:45: error: Module "rfdetr.models" does not explicitly export attribute "build_criterion_and_postprocessors"  [attr-defined]
rfdetr/main.py:45: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
rfdetr/main.py:55: error: Call to untyped function "set_sharing_strategy" in typed context  [no-untyped-call]
rfdetr/main.py:67: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/main.py:67: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/main.py:77: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:79: error: Call to untyped function "_initialize_model_structure" in typed context  [no-untyped-call]
rfdetr/main.py:80: error: Call to untyped function "_load_pretrained_weights" in typed context  [no-untyped-call]
rfdetr/main.py:81: error: Call to untyped function "_apply_lora_if_needed" in typed context  [no-untyped-call]
rfdetr/main.py:82: error: Call to untyped function "_finalize_model_setup" in typed context  [no-untyped-call]
rfdetr/main.py:84: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:88: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
rfdetr/main.py:90: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
rfdetr/main.py:93: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:99: error: Call to untyped function "_load_checkpoint" in typed context  [no-untyped-call]
rfdetr/main.py:106: error: Call to untyped function "_handle_class_mismatch" in typed context  [no-untyped-call]
rfdetr/main.py:109: error: Call to untyped function "_process_exclude_keys" in typed context  [no-untyped-call]
rfdetr/main.py:112: error: Call to untyped function "_process_keys_to_modify" in typed context  [no-untyped-call]
rfdetr/main.py:115: error: Call to untyped function "_update_query_params" in typed context  [no-untyped-call]
rfdetr/main.py:120: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:130: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:139: error: Call to untyped function "reinitialize_detection_head" in typed context  [no-untyped-call]
rfdetr/main.py:141: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:148: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:156: error: Call to untyped function "get_coco_pretrain_from_obj365" in typed context  [no-untyped-call]
rfdetr/main.py:164: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:172: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:196: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:199: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
rfdetr/main.py:201: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:204: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/main.py:204: note: Use "-> None" if function does not return a value
rfdetr/main.py:208: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/main.py:208: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/main.py:208: error: Missing type parameters for generic type "Callable"  [type-arg]
rfdetr/main.py:211: error: Call to untyped function "_validate_callbacks" in typed context  [no-untyped-call]
rfdetr/main.py:213: error: Call to untyped function "_setup_training_environment" in typed context  [no-untyped-call]
rfdetr/main.py:216: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
rfdetr/main.py:217: error: Call to untyped function "_setup_optimizer" in typed context  [no-untyped-call]
rfdetr/main.py:221: error: Call to untyped function "_setup_datasets_and_loaders" in typed context  [no-untyped-call]
rfdetr/main.py:226: error: Call to untyped function "_resume_from_checkpoint" in typed context  [no-untyped-call]
rfdetr/main.py:230: error: Call to untyped function "_run_evaluation" in typed context  [no-untyped-call]
rfdetr/main.py:236: error: Call to untyped function "_setup_drop_schedules" in typed context  [no-untyped-call]
rfdetr/main.py:239: error: Call to untyped function "_run_training_loop" in typed context  [no-untyped-call]
rfdetr/main.py:257: error: Call to untyped function "_save_final_results" in typed context  [no-untyped-call]
rfdetr/main.py:260: error: Call to untyped function "_finalize_model" in typed context  [no-untyped-call]
rfdetr/main.py:266: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:276: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:278: error: Call to untyped function "init_distributed_mode" in typed context  [no-untyped-call]
rfdetr/main.py:279: error: Call to untyped function "get_sha" in typed context  [no-untyped-call]
rfdetr/main.py:284: error: Call to untyped function "get_rank" in typed context  [no-untyped-call]
rfdetr/main.py:297: error: Call to untyped function "convert_sync_batchnorm" of "SyncBatchNorm" in typed context  [no-untyped-call]
rfdetr/main.py:305: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:312: error: Call to untyped function "_create_lr_lambda" in typed context  [no-untyped-call]
rfdetr/main.py:317: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:320: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
rfdetr/main.py:321: error: Call to untyped function "get_world_size" in typed context  [no-untyped-call]
rfdetr/main.py:329: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/main.py:350: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:352: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
rfdetr/main.py:353: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
rfdetr/main.py:357: error: Need type annotation for "sampler_train"  [var-annotated]
rfdetr/main.py:358: error: Need type annotation for "sampler_val"  [var-annotated]
rfdetr/main.py:360: error: Incompatible types in assignment (expression has type "RandomSampler", variable has type "DistributedSampler[Any]")  [assignment]
rfdetr/main.py:361: error: Incompatible types in assignment (expression has type "SequentialSampler", variable has type "DistributedSampler[Any]")  [assignment]
rfdetr/main.py:367: error: Call to untyped function "_create_train_loader" in typed context  [no-untyped-call]
rfdetr/main.py:382: error: Call to untyped function "get_coco_api_from_dataset" in typed context  [no-untyped-call]
rfdetr/main.py:386: error: Call to untyped function "ModelEma" in typed context  [no-untyped-call]
rfdetr/main.py:388: error: Incompatible types in assignment (expression has type "None", variable has type "ModelEma")  [assignment]
rfdetr/main.py:392: error: Call to untyped function "is_main_process" in typed context  [no-untyped-call]
rfdetr/main.py:394: error: Call to untyped function "benchmark" in typed context  [no-untyped-call]
rfdetr/main.py:404: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:437: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:445: error: Call to untyped function "clean_state_dict" in typed context  [no-untyped-call]
rfdetr/main.py:448: error: Call to untyped function "ModelEma" in typed context  [no-untyped-call]
rfdetr/main.py:461: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:465: error: Call to untyped function "evaluate" in typed context  [no-untyped-call]
rfdetr/main.py:470: error: Call to untyped function "save_on_master" in typed context  [no-untyped-call]
rfdetr/main.py:472: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:475: error: Call to untyped function "get_world_size" in typed context  [no-untyped-call]
rfdetr/main.py:484: error: Call to untyped function "drop_scheduler" in typed context  [no-untyped-call]
rfdetr/main.py:500: error: Call to untyped function "drop_scheduler" in typed context  [no-untyped-call]
rfdetr/main.py:516: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:557: error: Call to untyped function "_process_epoch" in typed context  [no-untyped-call]
rfdetr/main.py:581: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:641: error: Call to untyped function "_save_epoch_checkpoints" in typed context  [no-untyped-call]
rfdetr/main.py:646: error: Call to untyped function "_evaluate_model" in typed context  [no-untyped-call]
rfdetr/main.py:651: error: Call to untyped function "_update_regular_model_metrics" in typed context  [no-untyped-call]
rfdetr/main.py:664: error: Call to untyped function "_process_ema_model" in typed context  [no-untyped-call]
rfdetr/main.py:679: error: Call to untyped function "_create_log_stats" in typed context  [no-untyped-call]
rfdetr/main.py:684: error: Call to untyped function "_save_evaluation_logs" in typed context  [no-untyped-call]
rfdetr/main.py:692: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:723: error: Call to untyped function "save_on_master" in typed context  [no-untyped-call]
rfdetr/main.py:725: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:730: error: Call to untyped function "evaluate" in typed context  [no-untyped-call]
rfdetr/main.py:734: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:756: error: Call to untyped function "save_on_master" in typed context  [no-untyped-call]
rfdetr/main.py:769: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:784: error: Call to untyped function "evaluate" in typed context  [no-untyped-call]
rfdetr/main.py:806: error: Call to untyped function "save_on_master" in typed context  [no-untyped-call]
rfdetr/main.py:819: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:852: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:854: error: Call to untyped function "is_main_process" in typed context  [no-untyped-call]
rfdetr/main.py:870: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:872: error: Call to untyped function "is_main_process" in typed context  [no-untyped-call]
rfdetr/main.py:890: error: Call to untyped function "strip_checkpoint" in typed context  [no-untyped-call]
rfdetr/main.py:908: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:918: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/main.py:953: error: Call to untyped function "make_infer_image" in typed context  [no-untyped-call]
rfdetr/main.py:973: error: Call to untyped function "export_onnx" in typed context  [no-untyped-call]
rfdetr/main.py:1401: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/detr.py:32: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/detr.py:33: error: Call to untyped function "get_model_config" in typed context  [no-untyped-call]
rfdetr/detr.py:34: error: Call to untyped function "maybe_download_pretrain_weights" in typed context  [no-untyped-call]
rfdetr/detr.py:38: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/detr.py:38: note: Use "-> None" if function does not return a value
rfdetr/detr.py:41: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/detr.py:44: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/detr.py:45: error: Call to untyped function "get_train_config" in typed context  [no-untyped-call]
rfdetr/detr.py:48: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/detr.py:51: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/detr.py:51: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/detr.py:117: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/detr.py:120: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/detr.py:121: error: Call to untyped function "Model" in typed context  [no-untyped-call]
rfdetr/detr.py:125: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/detr.py:131: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/detr.py:133: error: Missing type parameters for generic type "ndarray"  [type-arg]
rfdetr/detr.py:143: error: Name "sv.Detections" is not defined  [name-defined]
rfdetr/detr.py:183: error: Item "builtins.bool" of "Union[Any, ndarray[Any, dtype[numpy.bool]], builtins.bool, Tensor]" has no attribute "any"  [union-attr]
rfdetr/detr.py:183: error: Unsupported operand types for < ("int" and "Image")  [operator]
rfdetr/detr.py:183: note: Left operand is of type "Union[str, ndarray[Any, Any], Image, Tensor]"
rfdetr/detr.py:183: error: Unsupported operand types for > ("str" and "int")  [operator]
rfdetr/detr.py:188: error: Item "str" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "shape"  [union-attr]
rfdetr/detr.py:188: error: Item "Image" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "shape"  [union-attr]
rfdetr/detr.py:191: error: Item "str" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "shape"  [union-attr]
rfdetr/detr.py:191: error: Item "Image" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "shape"  [union-attr]
rfdetr/detr.py:195: error: Item "str" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "shape"  [union-attr]
rfdetr/detr.py:195: error: Item "Image" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "shape"  [union-attr]
rfdetr/detr.py:198: error: Item "str" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "to"  [union-attr]
rfdetr/detr.py:198: error: Item "ndarray[Any, Any]" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "to"  [union-attr]
rfdetr/detr.py:198: error: Item "Image" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "to"  [union-attr]
rfdetr/detr.py:204: error: Argument 1 to "stack" has incompatible type "list[Union[str, ndarray[Any, Any], Image, Tensor]]"; expected "Union[tuple[Tensor, ...], list[Tensor], None]"  [arg-type]
rfdetr/detr.py:233: error: Module "supervision" does not explicitly export attribute "Detections"  [attr-defined]
rfdetr/detr.py:248: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/detr.py:251: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/detr.py:256: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/detr.py:259: error: Function is missing a type annotation  [no-untyped-def]
tests/trace_masks.py:10: error: Module "rfdetr" does not explicitly export attribute "RFDETRBase"  [attr-defined]
tests/trace_masks.py:25: error: Call to untyped function "RFDETRBase" in typed context  [no-untyped-call]
tests/trace_masks.py:106: error: Call to untyped function "trace_mask_flow" in typed context  [no-untyped-call]
tests/test_segmentation_integration.py:12: error: Module "rfdetr" does not explicitly export attribute "RFDETRBase"  [attr-defined]
tests/test_segmentation_integration.py:16: error: Missing type parameters for generic type "ndarray"  [type-arg]
tests/test_segmentation_integration.py:40: error: Call to untyped function "RFDETRBase" in typed context  [no-untyped-call]
tests/test_segmentation_integration.py:82: error: Call to untyped function "RFDETRBase" in typed context  [no-untyped-call]
tests/test_segmentation_integration.py:139: error: Call to untyped function "RFDETRBase" in typed context  [no-untyped-call]
tests/test_segmentation_integration.py:146: error: Argument 1 to "predict" of "RFDETR" has incompatible type "list[Image]"; expected "Union[str, Image, ndarray[Any, Any], Tensor, list[Union[str, ndarray[Any, Any], Image, Tensor]]]"  [arg-type]
tests/test_segmentation_integration.py:146: note: "List" is invariant -- see https://mypy.readthedocs.io/en/stable/common_issues.html#variance
tests/test_segmentation_integration.py:146: note: Consider using "Sequence" instead, which is covariant
tests/test_segmentation_integration.py:155: error: Argument 1 to "predict" of "RFDETR" has incompatible type "list[Image]"; expected "Union[str, Image, ndarray[Any, Any], Tensor, list[Union[str, ndarray[Any, Any], Image, Tensor]]]"  [arg-type]
tests/test_segmentation_integration.py:155: note: "List" is invariant -- see https://mypy.readthedocs.io/en/stable/common_issues.html#variance
tests/test_segmentation_integration.py:155: note: Consider using "Sequence" instead, which is covariant
tests/test_segmentation_integration.py:185: error: Call to untyped function "RFDETRBase" in typed context  [no-untyped-call]
tests/test_segmentation_integration.py:207: error: Module "supervision" does not explicitly export attribute "BoxAnnotator"  [attr-defined]
tests/test_segmentation_integration.py:208: error: Module "supervision" does not explicitly export attribute "MaskAnnotator"  [attr-defined]
tests/test_segmentation_integration.py:209: error: Module "supervision" does not explicitly export attribute "LabelAnnotator"  [attr-defined]
tests/test_segmentation_integration.py:263: error: Call to untyped function (unknown) in typed context  [no-untyped-call]
tests/test_segmentation_integration.py:290: error: Call to untyped function "main" in typed context  [no-untyped-call]
tests/test_minimal_segmentation.py:10: error: Module "rfdetr" does not explicitly export attribute "RFDETRBase"  [attr-defined]
tests/test_minimal_segmentation.py:33: error: Call to untyped function "RFDETRBase" in typed context  [no-untyped-call]
tests/test_minimal_segmentation.py:113: error: Call to untyped function "test_segmentation" in typed context  [no-untyped-call]
tests/test_minimal_model.py:12: error: Module "rfdetr.models" does not explicitly export attribute "build_criterion_and_postprocessors"  [attr-defined]
tests/test_minimal_model.py:12: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
tests/test_minimal_model.py:24: error: Missing type parameters for generic type "list"  [type-arg]
tests/test_minimal_model.py:99: error: "type[TestMinimalModel]" has no attribute "args"  [attr-defined]
tests/test_minimal_model.py:101: error: "type[TestMinimalModel]" has no attribute "model"  [attr-defined]
tests/test_minimal_model.py:101: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
tests/test_minimal_model.py:101: error: "type[TestMinimalModel]" has no attribute "args"  [attr-defined]
tests/test_minimal_model.py:103: error: "type[TestMinimalModel]" has no attribute "model"  [attr-defined]
tests/test_minimal_model.py:104: error: "type[TestMinimalModel]" has no attribute "model"  [attr-defined]
tests/test_minimal_model.py:107: error: "type[TestMinimalModel]" has no attribute "criterion"  [attr-defined]
tests/test_minimal_model.py:107: error: "type[TestMinimalModel]" has no attribute "postprocessors"  [attr-defined]
tests/test_minimal_model.py:107: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
tests/test_minimal_model.py:107: error: "type[TestMinimalModel]" has no attribute "args"  [attr-defined]
tests/test_minimal_model.py:109: error: "type[TestMinimalModel]" has no attribute "criterion"  [attr-defined]
tests/test_minimal_model.py:114: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_minimal_model.py:114: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
tests/test_minimal_model.py:114: error: "TestMinimalModel" has no attribute "args"  [attr-defined]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/test_minimal_model.py:146: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_minimal_model.py:146: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
tests/test_minimal_model.py:146: error: "TestMinimalModel" has no attribute "args"  [attr-defined]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/test_minimal_model.py:152: error: "TestMinimalModel" has no attribute "model"  [attr-defined]
tests/test_minimal_model.py:156: error: "TestMinimalModel" has no attribute "args"  [attr-defined]
tests/test_minimal_model.py:160: error: Module "torch.amp" does not explicitly export attribute "autocast"  [attr-defined]
tests/test_minimal_model.py:161: error: "TestMinimalModel" has no attribute "model"  [attr-defined]
tests/test_minimal_model.py:171: error: "TestMinimalModel" has no attribute "criterion"  [attr-defined]
tests/test_coco_segmentation.py:32: error: Module "rfdetr.models" does not explicitly export attribute "build_criterion_and_postprocessors"  [attr-defined]
tests/test_coco_segmentation.py:32: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
tests/test_coco_segmentation.py:43: error: "type[TestCocoSegmentation]" has no attribute "coco_path"  [attr-defined]
tests/test_coco_segmentation.py:44: error: "type[TestCocoSegmentation]" has no attribute "coco_path"  [attr-defined]
tests/test_coco_segmentation.py:45: error: Argument 1 to "skipTest" of "TestCase" has incompatible type "type[TestCocoSegmentation]"; expected "TestCase"  [arg-type]
tests/test_coco_segmentation.py:48: error: "type[TestCocoSegmentation]" has no attribute "output_dir"  [attr-defined]
tests/test_coco_segmentation.py:49: error: "type[TestCocoSegmentation]" has no attribute "output_dir"  [attr-defined]
tests/test_coco_segmentation.py:54: error: "TestCocoSegmentation" has no attribute "coco_path"  [attr-defined]
tests/test_coco_segmentation.py:61: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_coco_segmentation.py:61: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
tests/test_coco_segmentation.py:81: error: Module "rfdetr.datasets.transforms" has no attribute "ConvertCoco"  [attr-defined]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/test_coco_segmentation.py:100: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_coco_segmentation.py:100: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/test_coco_segmentation.py:110: error: "TestCocoSegmentation" has no attribute "coco_path"  [attr-defined]
tests/test_coco_segmentation.py:118: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
tests/test_coco_segmentation.py:143: error: "TestCocoSegmentation" has no attribute "coco_path"  [attr-defined]
tests/test_coco_segmentation.py:148: error: "TestCocoSegmentation" has no attribute "output_dir"  [attr-defined]
tests/test_coco_segmentation.py:153: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
tests/test_coco_segmentation.py:154: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
tests/test_coco_segmentation.py:160: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_coco_segmentation.py:160: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/test_coco_segmentation.py:166: error: Call to untyped function "get_coco_api_from_dataset" in typed context  [no-untyped-call]
tests/test_coco_segmentation.py:169: error: Call to untyped function "evaluate" in typed context  [no-untyped-call]
tests/test_coco_segmentation.py:184: error: Module "torch.multiprocessing" does not explicitly export attribute "spawn"  [attr-defined]
tests/test_coco_segmentation.py:184: error: Call to untyped function "spawn" in typed context  [no-untyped-call]
tests/debug_segmentation.py:9: error: Module "rfdetr" does not explicitly export attribute "RFDETRBase"  [attr-defined]
tests/debug_segmentation.py:24: error: Call to untyped function "RFDETRBase" in typed context  [no-untyped-call]
tests/debug_segmentation.py:104: error: Call to untyped function "debug_model_outputs" in typed context  [no-untyped-call]
rfdetr/config_utils.py:11: error: Library stubs not installed for "yaml"  [import-untyped]
rfdetr/config_utils.py:41: error: Incompatible types in assignment (expression has type "str", variable has type "Literal['cpu', 'cuda', 'mps']")  [assignment]
rfdetr/config_utils.py:57: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
rfdetr/config_utils.py:130: error: Incompatible types in assignment (expression has type "str", variable has type "Literal['cpu', 'cuda', 'mps']")  [assignment]
rfdetr/config_utils.py:146: error: Argument 1 has incompatible type "Callable[[type[RFDETRConfig], RFDETRConfig], RFDETRConfig]"; expected "Union[Callable[[Never, ValidationInfo], Never], Callable[[Never], Never]]"  [arg-type]
rfdetr/config_utils.py:162: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/cli/main.py:17: error: Module "rfdetr" does not explicitly export attribute "RFDETRBase"  [attr-defined]
rfdetr/cli/main.py:20: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/cli/main.py:28: error: Statement is unreachable  [unreachable]
rfdetr/cli/main.py:38: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/cli/main.py:41: error: Call to untyped function "RFDETRBase" in typed context  [no-untyped-call]
rfdetr/cli/main.py:43: error: Call to untyped function "train" in typed context  [no-untyped-call]
rfdetr/cli/main.py:50: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/cli/main.py:51: error: Call to untyped function "RFDETRBase" in typed context  [no-untyped-call]
rfdetr/cli/main.py:52: error: Call to untyped function "train" in typed context  [no-untyped-call]
rfdetr/cli/main.py:59: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/cli/main.py:59: note: Use "-> None" if function does not return a value
rfdetr/cli/main.py:88: error: Call to untyped function "trainer" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:45: error: Module "rfdetr.models" does not explicitly export attribute "build_criterion_and_postprocessors"  [attr-defined]
build/lib/rfdetr/main.py:45: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
build/lib/rfdetr/main.py:55: error: Call to untyped function "set_sharing_strategy" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:67: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/main.py:67: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/main.py:77: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/main.py:78: error: Call to untyped function "populate_args" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:80: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:82: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:108: error: Call to untyped function "reinitialize_detection_head" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:160: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:163: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/main.py:166: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/main.py:166: note: Use "-> None" if function does not return a value
build/lib/rfdetr/main.py:170: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/main.py:170: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/main.py:170: error: Missing type parameters for generic type "Callable"  [type-arg]
build/lib/rfdetr/main.py:178: error: Call to untyped function "populate_args" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:179: error: Call to untyped function "init_distributed_mode" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:180: error: Call to untyped function "get_sha" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:185: error: Call to untyped function "get_rank" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:190: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:197: error: Call to untyped function "convert_sync_batchnorm" of "SyncBatchNorm" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:212: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:212: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
build/lib/rfdetr/main.py:213: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:213: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
build/lib/rfdetr/main.py:216: error: Call to untyped function "get_world_size" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:223: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/main.py:245: error: Need type annotation for "sampler_train"  [var-annotated]
build/lib/rfdetr/main.py:246: error: Need type annotation for "sampler_val"  [var-annotated]
build/lib/rfdetr/main.py:248: error: Incompatible types in assignment (expression has type "RandomSampler", variable has type "DistributedSampler[Any]")  [assignment]
build/lib/rfdetr/main.py:249: error: Incompatible types in assignment (expression has type "SequentialSampler", variable has type "DistributedSampler[Any]")  [assignment]
build/lib/rfdetr/main.py:289: error: Call to untyped function "get_coco_api_from_dataset" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:292: error: Call to untyped function "ModelEma" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:294: error: Incompatible types in assignment (expression has type "None", variable has type "ModelEma")  [assignment]
build/lib/rfdetr/main.py:298: error: Call to untyped function "is_main_process" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:302: error: Call to untyped function "benchmark" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:311: error: Call to untyped function "clean_state_dict" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:314: error: Call to untyped function "ModelEma" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:322: error: Call to untyped function "load_state_dict" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:326: error: Call to untyped function "evaluate" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:330: error: Call to untyped function "save_on_master" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:334: error: Call to untyped function "get_world_size" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:340: error: Call to untyped function "drop_scheduler" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:355: error: Call to untyped function "drop_scheduler" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:411: error: Call to untyped function "state_dict" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:425: error: Call to untyped function "save_on_master" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:428: error: Call to untyped function "evaluate" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:433: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:439: error: Call to untyped function "save_on_master" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:443: error: Call to untyped function "state_dict" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:456: error: Call to untyped function "evaluate" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:468: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:473: error: Call to untyped function "save_on_master" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:477: error: Call to untyped function "state_dict" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:483: error: Call to untyped function "summary" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:494: error: Call to untyped function "is_main_process" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:519: error: Call to untyped function "is_main_process" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:530: error: Call to untyped function "strip_checkpoint" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:552: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/main.py:587: error: Call to untyped function "make_infer_image" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:607: error: Call to untyped function "export_onnx" in typed context  [no-untyped-call]
build/lib/rfdetr/main.py:724: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/main.py:1028: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/engine.py:33: error: Module "torch.amp" does not explicitly export attribute "GradScaler"  [attr-defined]
build/lib/rfdetr/engine.py:33: error: Module "torch.amp" does not explicitly export attribute "autocast"  [attr-defined]
build/lib/rfdetr/engine.py:37: error: Incompatible import of "GradScaler" (imported name has type "type[torch.cuda.amp.grad_scaler.GradScaler]", local name has type "type[torch.amp.grad_scaler.GradScaler]")  [assignment]
build/lib/rfdetr/engine.py:37: error: Incompatible import of "autocast" (imported name has type "type[torch.cuda.amp.autocast_mode.autocast]", local name has type "type[torch.amp.autocast_mode.autocast]")  [assignment]
build/lib/rfdetr/engine.py:46: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/engine.py:60: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/engine.py:60: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/engine.py:64: error: Missing type parameters for generic type "Iterable"  [type-arg]
build/lib/rfdetr/engine.py:70: error: Incompatible default for argument "ema_m" (default has type "None", argument has type Module)  [assignment]
build/lib/rfdetr/engine.py:70: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
build/lib/rfdetr/engine.py:70: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
build/lib/rfdetr/engine.py:71: error: Missing type parameters for generic type "dict"  [type-arg]
build/lib/rfdetr/engine.py:75: error: Missing type parameters for generic type "Callable"  [type-arg]
build/lib/rfdetr/engine.py:79: error: Call to untyped function "MetricLogger" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:80: error: Call to untyped function "add_meter" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:80: error: Call to untyped function "SmoothedValue" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:81: error: Call to untyped function "add_meter" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:81: error: Call to untyped function "SmoothedValue" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:87: error: Call to untyped function "get_world_size" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:98: error: Argument 1 to "len" has incompatible type "Iterable[Any]"; expected "Sized"  [arg-type]
build/lib/rfdetr/engine.py:100: error: Call to untyped function "log_every" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:108: error: Value of type "Optional[defaultdict[str, list[Callable[..., Any]]]]" is not indexable  [index]
build/lib/rfdetr/engine.py:112: error: Item "Tensor" of "Union[Tensor, Module]" has no attribute "update_drop_path"  [union-attr]
build/lib/rfdetr/engine.py:112: error: "Tensor" not callable  [operator]
build/lib/rfdetr/engine.py:114: error: "Tensor" not callable  [operator]
build/lib/rfdetr/engine.py:117: error: Item "Tensor" of "Union[Tensor, Module]" has no attribute "update_dropout"  [union-attr]
build/lib/rfdetr/engine.py:117: error: "Tensor" not callable  [operator]
build/lib/rfdetr/engine.py:119: error: "Tensor" not callable  [operator]
build/lib/rfdetr/engine.py:131: error: Call to untyped function "get_autocast_args" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:136: error: Value of type "Union[Tensor, Module]" is not indexable  [index]
build/lib/rfdetr/engine.py:138: error: Unsupported right operand type for in ("Union[Tensor, Module]")  [operator]
build/lib/rfdetr/engine.py:141: error: No overload variant of "scale" of "GradScaler" matches argument type "int"  [call-overload]
build/lib/rfdetr/engine.py:141: note: Possible overload variants:
build/lib/rfdetr/engine.py:141: note:     def scale(self, outputs: Tensor) -> Tensor
build/lib/rfdetr/engine.py:141: note:     def scale(self, outputs: list[Tensor]) -> list[Tensor]
build/lib/rfdetr/engine.py:141: note:     def scale(self, outputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]
build/lib/rfdetr/engine.py:141: note:     def scale(self, outputs: Iterable[Tensor]) -> Iterable[Tensor]
build/lib/rfdetr/engine.py:144: error: Call to untyped function "reduce_dict" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:147: error: Value of type "Union[Tensor, Module]" is not indexable  [index]
build/lib/rfdetr/engine.py:147: error: Unsupported right operand type for in ("Union[Tensor, Module]")  [operator]
build/lib/rfdetr/engine.py:166: error: "Tensor" not callable  [operator]
build/lib/rfdetr/engine.py:167: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:170: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:171: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:173: error: Call to untyped function "synchronize_between_processes" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:178: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/engine.py:184: error: Call to untyped function "MetricLogger" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:185: error: Call to untyped function "add_meter" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:185: error: Call to untyped function "SmoothedValue" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:189: error: Call to untyped function "CocoEvaluator" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:191: error: Call to untyped function "log_every" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:199: error: Call to untyped function "get_autocast_args" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:218: error: Call to untyped function "reduce_dict" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:223: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:228: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:234: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:237: error: Call to untyped function "synchronize_between_processes" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:240: error: Call to untyped function "synchronize_between_processes" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:244: error: Call to untyped function "accumulate" in typed context  [no-untyped-call]
build/lib/rfdetr/engine.py:245: error: Call to untyped function "summarize" in typed context  [no-untyped-call]
build/lib/rfdetr/detr.py:32: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/detr.py:33: error: Call to untyped function "get_model_config" in typed context  [no-untyped-call]
build/lib/rfdetr/detr.py:34: error: Call to untyped function "maybe_download_pretrain_weights" in typed context  [no-untyped-call]
build/lib/rfdetr/detr.py:38: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/detr.py:38: note: Use "-> None" if function does not return a value
build/lib/rfdetr/detr.py:41: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/detr.py:44: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/detr.py:45: error: Call to untyped function "get_train_config" in typed context  [no-untyped-call]
build/lib/rfdetr/detr.py:48: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/detr.py:51: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/detr.py:51: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/detr.py:117: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/detr.py:120: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/detr.py:121: error: Call to untyped function "Model" in typed context  [no-untyped-call]
build/lib/rfdetr/detr.py:125: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/detr.py:131: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/detr.py:133: error: Missing type parameters for generic type "ndarray"  [type-arg]
build/lib/rfdetr/detr.py:143: error: Name "sv.Detections" is not defined  [name-defined]
build/lib/rfdetr/detr.py:183: error: Item "builtins.bool" of "Union[Any, ndarray[Any, dtype[numpy.bool]], builtins.bool, Tensor]" has no attribute "any"  [union-attr]
build/lib/rfdetr/detr.py:183: error: Unsupported operand types for < ("int" and "Image")  [operator]
build/lib/rfdetr/detr.py:183: note: Left operand is of type "Union[str, ndarray[Any, Any], Image, Tensor]"
build/lib/rfdetr/detr.py:183: error: Unsupported operand types for > ("str" and "int")  [operator]
build/lib/rfdetr/detr.py:188: error: Item "str" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "shape"  [union-attr]
build/lib/rfdetr/detr.py:188: error: Item "Image" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "shape"  [union-attr]
build/lib/rfdetr/detr.py:191: error: Item "str" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "shape"  [union-attr]
build/lib/rfdetr/detr.py:191: error: Item "Image" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "shape"  [union-attr]
build/lib/rfdetr/detr.py:195: error: Item "str" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "shape"  [union-attr]
build/lib/rfdetr/detr.py:195: error: Item "Image" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "shape"  [union-attr]
build/lib/rfdetr/detr.py:198: error: Item "str" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "to"  [union-attr]
build/lib/rfdetr/detr.py:198: error: Item "ndarray[Any, Any]" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "to"  [union-attr]
build/lib/rfdetr/detr.py:198: error: Item "Image" of "Union[str, ndarray[Any, Any], Image, Tensor]" has no attribute "to"  [union-attr]
build/lib/rfdetr/detr.py:204: error: Argument 1 to "stack" has incompatible type "list[Union[str, ndarray[Any, Any], Image, Tensor]]"; expected "Union[tuple[Tensor, ...], list[Tensor], None]"  [arg-type]
build/lib/rfdetr/detr.py:233: error: Module "supervision" does not explicitly export attribute "Detections"  [attr-defined]
build/lib/rfdetr/detr.py:248: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/detr.py:251: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/detr.py:256: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/detr.py:259: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/config_utils.py:11: error: Library stubs not installed for "yaml"  [import-untyped]
build/lib/rfdetr/config_utils.py:11: note: Hint: "python3 -m pip install types-PyYAML"
build/lib/rfdetr/config_utils.py:37: error: Incompatible types in assignment (expression has type "str", variable has type "Literal['cpu', 'cuda', 'mps']")  [assignment]
build/lib/rfdetr/config_utils.py:45: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/config_utils.py:113: error: Incompatible types in assignment (expression has type "str", variable has type "Literal['cpu', 'cuda', 'mps']")  [assignment]
build/lib/rfdetr/config_utils.py:130: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/config_utils.py:144: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/export.py:28: error: Module "rfdetr.deploy._onnx" does not explicitly export attribute "OnnxOptimizer"  [attr-defined]
build/lib/rfdetr/deploy/export.py:29: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
build/lib/rfdetr/deploy/export.py:32: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/deploy/export.py:39: error: Incompatible return value type (got "CompletedProcess[str]", expected "int")  [return-value]
build/lib/rfdetr/deploy/export.py:46: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/export.py:53: error: Call to untyped function "Compose" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:55: error: Call to untyped function "SquareResize" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:57: error: Call to untyped function "Normalize" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:68: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/export.py:104: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/export.py:104: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/deploy/export.py:113: error: Call to untyped function "OnnxOptimizer" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:114: error: Call to untyped function "info" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:115: error: Call to untyped function "common_opt" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:116: error: Call to untyped function "info" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:132: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
build/lib/rfdetr/deploy/export.py:164: error: Call to untyped function "parse_trtexec_output" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:164: error: "int" has no attribute "stdout"  [attr-defined]
build/lib/rfdetr/deploy/export.py:167: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/export.py:225: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/export.py:231: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/deploy/export.py:232: error: Call to untyped function "get_sha" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:249: error: Call to untyped function "get_rank" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:254: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:271: error: Call to untyped function "no_batch_norm" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:275: error: Call to untyped function "make_infer_image" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:275: error: Missing positional argument "batch_size" in call to "make_infer_image"  [call-arg]
build/lib/rfdetr/deploy/export.py:294: error: Call to untyped function "export_onnx" in typed context  [no-untyped-call]
build/lib/rfdetr/deploy/export.py:300: error: "trtexec" does not return a value (it only ever returns None)  [func-returns-value]
build/lib/rfdetr/cli/main.py:17: error: Module "rfdetr" does not explicitly export attribute "RFDETRBase"  [attr-defined]
build/lib/rfdetr/cli/main.py:20: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/cli/main.py:28: error: Statement is unreachable  [unreachable]
build/lib/rfdetr/cli/main.py:38: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/cli/main.py:41: error: Call to untyped function "RFDETRBase" in typed context  [no-untyped-call]
build/lib/rfdetr/cli/main.py:43: error: Call to untyped function "train" in typed context  [no-untyped-call]
build/lib/rfdetr/cli/main.py:50: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/cli/main.py:51: error: Call to untyped function "RFDETRBase" in typed context  [no-untyped-call]
build/lib/rfdetr/cli/main.py:52: error: Call to untyped function "train" in typed context  [no-untyped-call]
build/lib/rfdetr/cli/main.py:59: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/cli/main.py:59: note: Use "-> None" if function does not return a value
build/lib/rfdetr/cli/main.py:88: error: Call to untyped function "trainer" in typed context  [no-untyped-call]
tests/test_val_limit.py:31: error: "ModelConfig" has no attribute "resolution"  [attr-defined]
tests/test_val_limit.py:38: error: Call to untyped function "to_args" in typed context  [no-untyped-call]
tests/test_val_limit.py:41: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_val_limit.py:41: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/test_val_limit.py:46: error: Call to untyped function "to_args" in typed context  [no-untyped-call]
tests/test_val_limit.py:47: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_val_limit.py:47: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/test_training_dimensions.py:37: error: "ModelConfig" has no attribute "resolution"  [attr-defined]
tests/test_training_dimensions.py:48: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_training_dimensions.py:48: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/test_quick_training.py:20: error: Module "rfdetr.models" does not explicitly export attribute "build_criterion_and_postprocessors"  [attr-defined]
tests/test_quick_training.py:20: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
tests/test_quick_training.py:51: error: "type[TestQuickTraining]" has no attribute "config_path"  [attr-defined]
tests/test_quick_training.py:52: error: "type[TestQuickTraining]" has no attribute "train_limit"  [attr-defined]
tests/test_quick_training.py:53: error: "type[TestQuickTraining]" has no attribute "val_limit"  [attr-defined]
tests/test_quick_training.py:54: error: "type[TestQuickTraining]" has no attribute "epochs"  [attr-defined]
tests/test_quick_training.py:55: error: "type[TestQuickTraining]" has no attribute "steps"  [attr-defined]
tests/test_quick_training.py:56: error: "type[TestQuickTraining]" has no attribute "validate"  [attr-defined]
tests/test_quick_training.py:61: error: "TestQuickTraining" has no attribute "config_path"  [attr-defined]
tests/test_quick_training.py:96: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
tests/test_quick_training.py:105: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
tests/test_quick_training.py:125: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_quick_training.py:125: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/test_quick_training.py:131: error: "TestQuickTraining" has no attribute "train_limit"  [attr-defined]
tests/test_quick_training.py:145: error: Call to untyped function "ModelEma" in typed context  [no-untyped-call]
tests/test_quick_training.py:148: error: "TestQuickTraining" has no attribute "validate"  [attr-defined]
tests/test_quick_training.py:149: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
tests/test_quick_training.py:149: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
tests/test_quick_training.py:154: error: "TestQuickTraining" has no attribute "val_limit"  [attr-defined]
tests/test_quick_training.py:165: error: Call to untyped function "get_coco_api_from_dataset" in typed context  [no-untyped-call]
tests/test_quick_training.py:169: error: "TestQuickTraining" has no attribute "steps"  [attr-defined]
tests/test_quick_training.py:171: error: "TestQuickTraining" has no attribute "epochs"  [attr-defined]
tests/test_quick_training.py:220: error: Call to untyped function "step" in typed context  [no-untyped-call]
tests/test_quick_training.py:224: error: Call to untyped function "update" in typed context  [no-untyped-call]
tests/test_quick_training.py:244: error: Statement is unreachable  [unreachable]
tests/test_quick_training.py:247: error: "TestQuickTraining" has no attribute "validate"  [attr-defined]
tests/test_quick_training.py:252: error: Call to untyped function "evaluate" in typed context  [no-untyped-call]
tests/test_quick_training.py:303: error: Call to untyped function "parse_args" in typed context  [no-untyped-call]
tests/test_quick_training.py:307: error: "type[TestQuickTraining]" has no attribute "config_path"  [attr-defined]
tests/test_quick_training.py:308: error: "type[TestQuickTraining]" has no attribute "train_limit"  [attr-defined]
tests/test_quick_training.py:309: error: "type[TestQuickTraining]" has no attribute "val_limit"  [attr-defined]
tests/test_quick_training.py:310: error: "type[TestQuickTraining]" has no attribute "epochs"  [attr-defined]
tests/test_quick_training.py:311: error: "type[TestQuickTraining]" has no attribute "steps"  [attr-defined]
tests/test_quick_training.py:312: error: "type[TestQuickTraining]" has no attribute "validate"  [attr-defined]
tests/test_quick_training.py:315: error: Call to untyped function "setUpClass" of "TestQuickTraining" in typed context  [no-untyped-call]
tests/test_quick_training.py:316: error: Call to untyped function "test_quick_training" in typed context  [no-untyped-call]
tests/test_quick_training.py:320: error: Call to untyped function "main" in typed context  [no-untyped-call]
tests/test_model_construction.py:11: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
tests/test_model_construction.py:58: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
tests/test_model_construction.py:74: error: Call to untyped function "test_model_construction" in typed context  [no-untyped-call]
tests/test_fixed_size.py:13: error: Module "rfdetr.models" does not explicitly export attribute "build_criterion_and_postprocessors"  [attr-defined]
tests/test_fixed_size.py:13: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
tests/test_fixed_size.py:21: error: Missing type parameters for generic type "Dataset"  [type-arg]
tests/test_fixed_size.py:72: error: "type[TestFixedSize]" has no attribute "config_path"  [attr-defined]
tests/test_fixed_size.py:73: error: "type[TestFixedSize]" has no attribute "config"  [attr-defined]
tests/test_fixed_size.py:73: error: "type[TestFixedSize]" has no attribute "config_path"  [attr-defined]
tests/test_fixed_size.py:76: error: "type[TestFixedSize]" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:78: error: "type[TestFixedSize]" has no attribute "config"  [attr-defined]
tests/test_fixed_size.py:82: error: "type[TestFixedSize]" has no attribute "config"  [attr-defined]
tests/test_fixed_size.py:83: error: "type[TestFixedSize]" has no attribute "config"  [attr-defined]
tests/test_fixed_size.py:84: error: "type[TestFixedSize]" has no attribute "config"  [attr-defined]
tests/test_fixed_size.py:85: error: "type[TestFixedSize]" has no attribute "config"  [attr-defined]
tests/test_fixed_size.py:86: error: "type[TestFixedSize]" has no attribute "config"  [attr-defined]
tests/test_fixed_size.py:87: error: "type[TestFixedSize]" has no attribute "config"  [attr-defined]
tests/test_fixed_size.py:88: error: "type[TestFixedSize]" has no attribute "config"  [attr-defined]
tests/test_fixed_size.py:89: error: "type[TestFixedSize]" has no attribute "config"  [attr-defined]
tests/test_fixed_size.py:90: error: "type[TestFixedSize]" has no attribute "config"  [attr-defined]
tests/test_fixed_size.py:91: error: "type[TestFixedSize]" has no attribute "config"  [attr-defined]
tests/test_fixed_size.py:95: error: "type[TestFixedSize]" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:96: error: "type[TestFixedSize]" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:97: error: "type[TestFixedSize]" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:98: error: "type[TestFixedSize]" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:99: error: "type[TestFixedSize]" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:100: error: "type[TestFixedSize]" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:103: error: "type[TestFixedSize]" has no attribute "device"  [attr-defined]
tests/test_fixed_size.py:103: error: "type[TestFixedSize]" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:109: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
tests/test_fixed_size.py:109: error: "TestFixedSize" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:110: error: "TestFixedSize" has no attribute "device"  [attr-defined]
tests/test_fixed_size.py:124: error: Call to untyped function "FixedSizeDataset" in typed context  [no-untyped-call]
tests/test_fixed_size.py:125: error: "TestFixedSize" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:133: error: "TestFixedSize" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:138: error: "TestFixedSize" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:149: error: Call to untyped function "test_model_construction" in typed context  [no-untyped-call]
tests/test_fixed_size.py:150: error: Call to untyped function "test_fixed_size_dataset" in typed context  [no-untyped-call]
tests/test_fixed_size.py:153: error: "TestFixedSize" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:156: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
tests/test_fixed_size.py:156: error: "TestFixedSize" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:157: error: "TestFixedSize" has no attribute "device"  [attr-defined]
tests/test_fixed_size.py:160: error: "TestFixedSize" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:162: error: "TestFixedSize" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:169: error: "TestFixedSize" has no attribute "args"  [attr-defined]
tests/test_fixed_size.py:175: error: "TestFixedSize" has no attribute "device"  [attr-defined]
tests/test_fixed_size.py:176: error: "TestFixedSize" has no attribute "device"  [attr-defined]
tests/test_fixed_size.py:227: error: Call to untyped function "main" in typed context  [no-untyped-call]
tests/test_config_validation.py:8: error: Library stubs not installed for "yaml"  [import-untyped]
scripts/evaluate.py:24: error: Module "rfdetr.models" does not explicitly export attribute "build_criterion_and_postprocessors"  [attr-defined]
scripts/evaluate.py:24: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
scripts/evaluate.py:109: error: Call to untyped function "MetricLogger" in typed context  [no-untyped-call]
scripts/evaluate.py:110: error: Call to untyped function "add_meter" in typed context  [no-untyped-call]
scripts/evaluate.py:110: error: Call to untyped function "SmoothedValue" in typed context  [no-untyped-call]
scripts/evaluate.py:114: error: Call to untyped function "CocoEvaluator" in typed context  [no-untyped-call]
scripts/evaluate.py:116: error: Call to untyped function "log_every" in typed context  [no-untyped-call]
scripts/evaluate.py:142: error: Module "torch.amp" does not explicitly export attribute "autocast"  [attr-defined]
scripts/evaluate.py:183: error: Call to untyped function "reduce_dict" in typed context  [no-untyped-call]
scripts/evaluate.py:188: error: Call to untyped function "update" in typed context  [no-untyped-call]
scripts/evaluate.py:193: error: Call to untyped function "update" in typed context  [no-untyped-call]
scripts/evaluate.py:201: error: Call to untyped function "update" in typed context  [no-untyped-call]
scripts/evaluate.py:222: error: Call to untyped function "update" in typed context  [no-untyped-call]
scripts/evaluate.py:227: error: Call to untyped function "synchronize_between_processes" in typed context  [no-untyped-call]
scripts/evaluate.py:238: error: Call to untyped function "synchronize_between_processes" in typed context  [no-untyped-call]
scripts/evaluate.py:240: error: Call to untyped function "accumulate" in typed context  [no-untyped-call]
scripts/evaluate.py:241: error: Call to untyped function "summarize" in typed context  [no-untyped-call]
scripts/evaluate.py:265: error: Call to untyped function "extract_per_class_metrics" in typed context  [no-untyped-call]
scripts/evaluate.py:308: error: Call to untyped function "create_mock_metrics" in typed context  [no-untyped-call]
scripts/evaluate.py:308: error: Call to untyped function "get_categories" in typed context  [no-untyped-call]
scripts/evaluate.py:312: error: Call to untyped function "get_categories" in typed context  [no-untyped-call]
scripts/evaluate.py:320: error: Call to untyped function "create_mock_metrics" in typed context  [no-untyped-call]
scripts/evaluate.py:332: error: Call to untyped function "create_mock_metrics" in typed context  [no-untyped-call]
scripts/evaluate.py:332: error: Call to untyped function "get_categories" in typed context  [no-untyped-call]
scripts/evaluate.py:365: error: Call to untyped function "create_mock_metrics" in typed context  [no-untyped-call]
scripts/evaluate.py:365: error: Call to untyped function "get_categories" in typed context  [no-untyped-call]
scripts/evaluate.py:383: error: Call to untyped function "main" in typed context  [no-untyped-call]
scripts/evaluate.py:407: error: Call to untyped function "to_args" in typed context  [no-untyped-call]
scripts/evaluate.py:439: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
scripts/evaluate.py:483: error: Call to untyped function "MockModel" in typed context  [no-untyped-call]
scripts/evaluate.py:487: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
scripts/evaluate.py:510: error: Call to untyped function "MockCriterion" in typed context  [no-untyped-call]
scripts/evaluate.py:530: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
scripts/evaluate.py:530: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
scripts/evaluate.py:563: error: Missing type parameters for generic type "Dataset"  [type-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
scripts/evaluate.py:544: error: Call to untyped function "get_coco_api_from_dataset" in typed context  [no-untyped-call]
scripts/evaluate.py:587: error: Call to untyped function "evaluate_with_per_class_metrics" in typed context  [no-untyped-call]
scripts/evaluate.py:648: error: Call to untyped function "get_args_parser" in typed context  [no-untyped-call]
scripts/evaluate.py:650: error: Call to untyped function "main" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:33: error: Module "rfdetr.models" does not explicitly export attribute "build_criterion_and_postprocessors"  [attr-defined]
rfdetr/lightning_module.py:33: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
rfdetr/lightning_module.py:41: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/lightning_module.py:60: error: Incompatible types in assignment (expression has type "dict[Any, Any]", variable has type "ModelConfig")  [assignment]
rfdetr/lightning_module.py:69: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:70: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:77: error: Statement is unreachable  [unreachable]
rfdetr/lightning_module.py:86: error: Call to untyped function "ModelEma" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:94: error: Call to untyped function "_setup_autocast_args" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:103: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/lightning_module.py:104: error: Statement is unreachable  [unreachable]
rfdetr/lightning_module.py:115: error: Cannot determine type of "export_onnx"  [has-type]
rfdetr/lightning_module.py:116: error: Cannot determine type of "export_torch"  [has-type]
rfdetr/lightning_module.py:117: error: Cannot determine type of "simplify_onnx"  [has-type]
rfdetr/lightning_module.py:118: error: Cannot determine type of "export_on_validation"  [has-type]
rfdetr/lightning_module.py:119: error: Cannot determine type of "max_steps"  [has-type]
rfdetr/lightning_module.py:120: error: Cannot determine type of "val_frequency"  [has-type]
rfdetr/lightning_module.py:128: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/lightning_module.py:128: note: Use "-> None" if function does not return a value
rfdetr/lightning_module.py:146: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/lightning_module.py:147: error: Statement is unreachable  [unreachable]
rfdetr/lightning_module.py:163: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/lightning_module.py:167: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/lightning_module.py:176: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "str"  [arg-type]
rfdetr/lightning_module.py:176: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "Optional[dtype]"  [arg-type]
rfdetr/lightning_module.py:176: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "bool"  [arg-type]
rfdetr/lightning_module.py:176: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "Optional[bool]"  [arg-type]
rfdetr/lightning_module.py:183: error: Call to untyped function "reduce_dict" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:213: error: Item "list[LightningOptimizer]" of "Union[LightningOptimizer, list[LightningOptimizer]]" has no attribute "param_groups"  [union-attr]
rfdetr/lightning_module.py:221: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:227: error: Item "list[LightningOptimizer]" of "Union[LightningOptimizer, list[LightningOptimizer]]" has no attribute "param_groups"  [union-attr]
rfdetr/lightning_module.py:235: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/lightning_module.py:248: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/lightning_module.py:249: error: Statement is unreachable  [unreachable]
rfdetr/lightning_module.py:258: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "str"  [arg-type]
rfdetr/lightning_module.py:258: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "Optional[dtype]"  [arg-type]
rfdetr/lightning_module.py:258: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "bool"  [arg-type]
rfdetr/lightning_module.py:258: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "Optional[bool]"  [arg-type]
rfdetr/lightning_module.py:284: error: Call to untyped function "reduce_dict" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:325: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/lightning_module.py:335: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/lightning_module.py:336: error: Statement is unreachable  [unreachable]
rfdetr/lightning_module.py:360: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/lightning_module.py:369: error: Cannot determine type of "export_onnx"  [has-type]
rfdetr/lightning_module.py:369: error: Cannot determine type of "export_torch"  [has-type]
rfdetr/lightning_module.py:395: error: Cannot determine type of "export_torch"  [has-type]
rfdetr/lightning_module.py:401: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/lightning_module.py:402: error: Statement is unreachable  [unreachable]
rfdetr/lightning_module.py:424: error: Cannot determine type of "export_onnx"  [has-type]
rfdetr/lightning_module.py:430: error: Cannot determine type of "simplify_onnx"  [has-type]
rfdetr/lightning_module.py:441: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/lightning_module.py:441: note: Use "-> None" if function does not return a value
rfdetr/lightning_module.py:452: error: Call to untyped function "get_coco_api_from_dataset" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:455: error: Call to untyped function "CocoEvaluator" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:458: error: Incompatible types in assignment (expression has type "None", variable has type "CocoEvaluator")  [assignment]
rfdetr/lightning_module.py:461: error: Incompatible types in assignment (expression has type "None", variable has type "CocoEvaluator")  [assignment]
rfdetr/lightning_module.py:466: error: Incompatible types in assignment (expression has type "None", variable has type "CocoEvaluator")  [assignment]
rfdetr/lightning_module.py:470: error: Cannot determine type of "export_on_validation"  [has-type]
rfdetr/lightning_module.py:472: error: Call to untyped function "export_model" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:477: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/lightning_module.py:477: error: Signature of "on_validation_batch_end" incompatible with supertype "ModelHooks"  [override]
rfdetr/lightning_module.py:477: note:      Superclass:
rfdetr/lightning_module.py:477: note:          def on_validation_batch_end(self, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int, dataloader_idx: int = ...) -> None
rfdetr/lightning_module.py:477: note:      Subclass:
rfdetr/lightning_module.py:477: note:          def on_validation_batch_end(self, outputs: Any, batch: Any, batch_idx: Any) -> Any
rfdetr/lightning_module.py:484: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:488: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/lightning_module.py:488: note: Use "-> None" if function does not return a value
rfdetr/lightning_module.py:509: error: Subclass of "list[Any]" and "ndarray[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/lightning_module.py:509: error: Right operand of "and" is never evaluated  [unreachable]
rfdetr/lightning_module.py:510: error: Statement is unreachable  [unreachable]
rfdetr/lightning_module.py:516: error: Call to untyped function "synchronize_between_processes" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:520: error: Call to untyped function "accumulate" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:521: error: Call to untyped function "summarize" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:567: error: Statement is unreachable  [unreachable]
rfdetr/lightning_module.py:575: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/lightning_module.py:582: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/lightning_module.py:600: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/lightning_module.py:601: error: Statement is unreachable  [unreachable]
rfdetr/lightning_module.py:617: error: Cannot determine type of "max_steps"  [has-type]
rfdetr/lightning_module.py:620: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/lightning_module.py:621: error: Statement is unreachable  [unreachable]
rfdetr/lightning_module.py:632: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/lightning_module.py:668: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/lightning_module.py:685: error: Incompatible types in assignment (expression has type "dict[Any, Any]", variable has type "ModelConfig")  [assignment]
rfdetr/lightning_module.py:694: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/lightning_module.py:695: error: Statement is unreachable  [unreachable]
rfdetr/lightning_module.py:700: error: Cannot determine type of "batch_size"  [has-type]
rfdetr/lightning_module.py:701: error: Cannot determine type of "num_workers"  [has-type]
rfdetr/lightning_module.py:702: error: Cannot determine type of "training_width"  [has-type]
rfdetr/lightning_module.py:703: error: Cannot determine type of "training_height"  [has-type]
rfdetr/lightning_module.py:705: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/lightning_module.py:708: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:710: error: Cannot determine type of "training_width"  [has-type]
rfdetr/lightning_module.py:710: error: Cannot determine type of "training_height"  [has-type]
rfdetr/lightning_module.py:712: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
rfdetr/lightning_module.py:714: error: Cannot determine type of "training_width"  [has-type]
rfdetr/lightning_module.py:714: error: Cannot determine type of "training_height"  [has-type]
rfdetr/lightning_module.py:717: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/lightning_module.py:719: error: Item "None" of "Optional[Trainer]" has no attribute "world_size"  [union-attr]
rfdetr/lightning_module.py:720: error: Need type annotation for "sampler_train"  [var-annotated]
rfdetr/lightning_module.py:722: error: Incompatible types in assignment (expression has type "RandomSampler", variable has type "DistributedSampler[Any]")  [assignment]
rfdetr/lightning_module.py:725: error: Cannot determine type of "batch_size"  [has-type]
rfdetr/lightning_module.py:732: error: Cannot determine type of "num_workers"  [has-type]
rfdetr/lightning_module.py:737: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/lightning_module.py:742: error: Item "None" of "Optional[Trainer]" has no attribute "world_size"  [union-attr]
rfdetr/lightning_module.py:743: error: Need type annotation for "sampler_val"  [var-annotated]
rfdetr/lightning_module.py:745: error: Incompatible types in assignment (expression has type "SequentialSampler", variable has type "DistributedSampler[Any]")  [assignment]
rfdetr/lightning_module.py:749: error: Cannot determine type of "batch_size"  [has-type]
rfdetr/lightning_module.py:753: error: Cannot determine type of "num_workers"  [has-type]
rfdetr/fabric_module.py:20: error: Module "lightning.fabric.strategies" does not explicitly export attribute "DDPStrategy"  [attr-defined]
rfdetr/fabric_module.py:24: error: Module "torch.amp" does not explicitly export attribute "autocast"  [attr-defined]
rfdetr/fabric_module.py:29: error: Module "rfdetr.models" does not explicitly export attribute "build_criterion_and_postprocessors"  [attr-defined]
rfdetr/fabric_module.py:29: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
rfdetr/fabric_module.py:34: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/fabric_module.py:62: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/fabric_module.py:78: error: Incompatible types in assignment (expression has type "dict[Any, Any]", variable has type "ModelConfig")  [assignment]
rfdetr/fabric_module.py:87: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:88: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:91: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/fabric_module.py:93: error: Statement is unreachable  [unreachable]
rfdetr/fabric_module.py:97: error: Cannot determine type of "ema_decay"  [has-type]
rfdetr/fabric_module.py:108: error: Call to untyped function "_setup_autocast_args" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:114: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/fabric_module.py:115: error: Statement is unreachable  [unreachable]
rfdetr/fabric_module.py:126: error: Cannot determine type of "export_onnx"  [has-type]
rfdetr/fabric_module.py:127: error: Cannot determine type of "export_torch"  [has-type]
rfdetr/fabric_module.py:128: error: Cannot determine type of "simplify_onnx"  [has-type]
rfdetr/fabric_module.py:129: error: Cannot determine type of "export_on_validation"  [has-type]
rfdetr/fabric_module.py:130: error: Cannot determine type of "max_steps"  [has-type]
rfdetr/fabric_module.py:131: error: Cannot determine type of "eval_save_frequency"  [has-type]
rfdetr/fabric_module.py:139: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/fabric_module.py:139: note: Use "-> None" if function does not return a value
rfdetr/fabric_module.py:142: error: Call to untyped function "get_autocast_args" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:144: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/fabric_module.py:154: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/fabric_module.py:155: error: Statement is unreachable  [unreachable]
rfdetr/fabric_module.py:172: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/fabric_module.py:183: error: Cannot determine type of "ema_decay"  [has-type]
rfdetr/fabric_module.py:184: error: Call to untyped function "ModelEma" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:184: error: Cannot determine type of "ema_decay"  [has-type]
rfdetr/fabric_module.py:187: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/fabric_module.py:188: error: Statement is unreachable  [unreachable]
rfdetr/fabric_module.py:206: error: Call to untyped function "_setup_lr_scheduler" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:208: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/fabric_module.py:211: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/fabric_module.py:212: error: Statement is unreachable  [unreachable]
rfdetr/fabric_module.py:221: error: Cannot determine type of "max_steps"  [has-type]
rfdetr/fabric_module.py:224: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/fabric_module.py:232: error: Cannot determine type of "max_steps"  [has-type]
rfdetr/fabric_module.py:239: error: Cannot determine type of "max_steps"  [has-type]
rfdetr/fabric_module.py:246: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/fabric_module.py:284: error: Call to untyped function "reduce_dict" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:302: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/fabric_module.py:319: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/fabric_module.py:320: error: Statement is unreachable  [unreachable]
rfdetr/fabric_module.py:363: error: Call to untyped function "reduce_dict" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:381: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/fabric_module.py:390: error: Cannot determine type of "export_onnx"  [has-type]
rfdetr/fabric_module.py:390: error: Cannot determine type of "export_torch"  [has-type]
rfdetr/fabric_module.py:416: error: Cannot determine type of "export_torch"  [has-type]
rfdetr/fabric_module.py:433: error: Cannot determine type of "export_onnx"  [has-type]
rfdetr/fabric_module.py:439: error: Call to untyped function "_make_dummy_input" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:443: error: Call to untyped function "export_onnx" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:443: error: Missing positional arguments "input_names", "output_names", "dynamic_axes" in call to "export_onnx"  [call-arg]
rfdetr/fabric_module.py:452: error: Cannot determine type of "simplify_onnx"  [has-type]
rfdetr/fabric_module.py:453: error: Missing positional argument "input_names" in call to "onnx_simplify"  [call-arg]
rfdetr/fabric_module.py:453: error: Argument "onnx_dir" to "onnx_simplify" has incompatible type "Path"; expected "str"  [arg-type]
rfdetr/fabric_module.py:465: error: Incompatible types in assignment (expression has type "Optional[Any]", variable has type "device")  [assignment]
rfdetr/fabric_module.py:473: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/fabric_module.py:489: error: Incompatible types in assignment (expression has type "dict[Any, Any]", variable has type "ModelConfig")  [assignment]
rfdetr/fabric_module.py:498: error: Subclass of "ModelConfig" and "dict[Any, Any]" cannot exist: would have incompatible method signatures  [unreachable]
rfdetr/fabric_module.py:499: error: Statement is unreachable  [unreachable]
rfdetr/fabric_module.py:504: error: Cannot determine type of "batch_size"  [has-type]
rfdetr/fabric_module.py:505: error: Cannot determine type of "num_workers"  [has-type]
rfdetr/fabric_module.py:506: error: Cannot determine type of "training_width"  [has-type]
rfdetr/fabric_module.py:507: error: Cannot determine type of "training_height"  [has-type]
rfdetr/fabric_module.py:513: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/fabric_module.py:520: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:522: error: Cannot determine type of "training_width"  [has-type]
rfdetr/fabric_module.py:522: error: Cannot determine type of "training_height"  [has-type]
rfdetr/fabric_module.py:524: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:526: error: Cannot determine type of "training_width"  [has-type]
rfdetr/fabric_module.py:526: error: Cannot determine type of "training_height"  [has-type]
rfdetr/fabric_module.py:532: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/fabric_module.py:535: error: Need type annotation for "sampler_train"  [var-annotated]
rfdetr/fabric_module.py:535: error: Argument 1 to "DistributedSampler" has incompatible type "Optional[Any]"; expected "Dataset[Any]"  [arg-type]
rfdetr/fabric_module.py:537: error: Incompatible types in assignment (expression has type "RandomSampler", variable has type "DistributedSampler[Any]")  [assignment]
rfdetr/fabric_module.py:537: error: Argument 1 to "RandomSampler" has incompatible type "Optional[Any]"; expected "Sized"  [arg-type]
rfdetr/fabric_module.py:540: error: Cannot determine type of "batch_size"  [has-type]
rfdetr/fabric_module.py:544: error: Argument 1 to "DataLoader" has incompatible type "Optional[Any]"; expected "Dataset[Any]"  [arg-type]
rfdetr/fabric_module.py:547: error: Cannot determine type of "num_workers"  [has-type]
rfdetr/fabric_module.py:554: error: Function is missing a return type annotation  [no-untyped-def]
rfdetr/fabric_module.py:557: error: Need type annotation for "sampler_val"  [var-annotated]
rfdetr/fabric_module.py:557: error: Argument 1 to "DistributedSampler" has incompatible type "Optional[Any]"; expected "Dataset[Any]"  [arg-type]
rfdetr/fabric_module.py:559: error: Incompatible types in assignment (expression has type "SequentialSampler", variable has type "DistributedSampler[Any]")  [assignment]
rfdetr/fabric_module.py:559: error: Argument 1 to "SequentialSampler" has incompatible type "Optional[Any]"; expected "Sized"  [arg-type]
rfdetr/fabric_module.py:562: error: Argument 1 to "DataLoader" has incompatible type "Optional[Any]"; expected "Dataset[Any]"  [arg-type]
rfdetr/fabric_module.py:563: error: Cannot determine type of "batch_size"  [has-type]
rfdetr/fabric_module.py:567: error: Cannot determine type of "num_workers"  [has-type]
rfdetr/fabric_module.py:575: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/fabric_module.py:610: error: Incompatible types in assignment (expression has type "DDPStrategy", variable has type "str")  [assignment]
rfdetr/fabric_module.py:621: error: Call to untyped function "RFDETRFabricModule" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:622: error: Call to untyped function "RFDETRFabricData" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:625: error: Call to untyped function "setup" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:626: error: Call to untyped function "train_dataloader" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:627: error: Call to untyped function "val_dataloader" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:630: error: Call to untyped function "setup" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:633: error: Call to untyped function "get_coco_api_from_dataset" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:637: error: Cannot determine type of "max_steps"  [has-type]
rfdetr/fabric_module.py:638: error: Cannot determine type of "eval_save_frequency"  [has-type]
rfdetr/fabric_module.py:668: error: Call to untyped function "train_step" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:681: error: Call to untyped function "CocoEvaluator" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:689: error: Call to untyped function "validation_step" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:691: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:694: error: Call to untyped function "synchronize_between_processes" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:695: error: Call to untyped function "accumulate" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:696: error: Call to untyped function "summarize" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:724: error: Call to untyped function "state_dict" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:738: error: Call to untyped function "state_dict" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:752: error: Call to untyped function "state_dict" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:766: error: Call to untyped function "state_dict" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:774: error: Cannot determine type of "export_on_validation"  [has-type]
rfdetr/fabric_module.py:775: error: Call to untyped function "export_model" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:799: error: Call to untyped function "CocoEvaluator" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:803: error: Call to untyped function "validation_step" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:804: error: Call to untyped function "update" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:806: error: Call to untyped function "synchronize_between_processes" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:807: error: Call to untyped function "accumulate" in typed context  [no-untyped-call]
rfdetr/fabric_module.py:808: error: Call to untyped function "summarize" in typed context  [no-untyped-call]
rfdetr/hooks/onnx_checkpoint_hook.py:25: error: Module "rfdetr.deploy._onnx" does not explicitly export attribute "OnnxOptimizer"  [attr-defined]
rfdetr/hooks/onnx_checkpoint_hook.py:41: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/hooks/onnx_checkpoint_hook.py:72: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/hooks/onnx_checkpoint_hook.py:104: error: Function is missing a type annotation  [no-untyped-def]
rfdetr/hooks/onnx_checkpoint_hook.py:202: error: Call to untyped function "_make_dummy_input" in typed context  [no-untyped-call]
rfdetr/hooks/onnx_checkpoint_hook.py:239: error: Call to untyped function "OnnxOptimizer" in typed context  [no-untyped-call]
rfdetr/hooks/onnx_checkpoint_hook.py:240: error: Call to untyped function "common_opt" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:26: error: Module "rfdetr.models" does not explicitly export attribute "build_criterion_and_postprocessors"  [attr-defined]
build/lib/rfdetr/lightning_module.py:26: error: Module "rfdetr.models" does not explicitly export attribute "build_model"  [attr-defined]
build/lib/rfdetr/lightning_module.py:34: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:45: error: Call to untyped function "build_model" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:46: error: Call to untyped function "build_criterion_and_postprocessors" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:51: error: Call to untyped function "ModelEma" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:62: error: Call to untyped function "_setup_autocast_args" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:80: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:80: note: Use "-> None" if function does not return a value
build/lib/rfdetr/lightning_module.py:112: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:116: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:125: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "str"  [arg-type]
build/lib/rfdetr/lightning_module.py:125: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "Optional[dtype]"  [arg-type]
build/lib/rfdetr/lightning_module.py:125: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "bool"  [arg-type]
build/lib/rfdetr/lightning_module.py:125: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "Optional[bool]"  [arg-type]
build/lib/rfdetr/lightning_module.py:132: error: Call to untyped function "reduce_dict" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:162: error: Item "list[LightningOptimizer]" of "Union[LightningOptimizer, list[LightningOptimizer]]" has no attribute "param_groups"  [union-attr]
build/lib/rfdetr/lightning_module.py:170: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:176: error: Item "list[LightningOptimizer]" of "Union[LightningOptimizer, list[LightningOptimizer]]" has no attribute "param_groups"  [union-attr]
build/lib/rfdetr/lightning_module.py:184: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:202: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "str"  [arg-type]
build/lib/rfdetr/lightning_module.py:202: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "Optional[dtype]"  [arg-type]
build/lib/rfdetr/lightning_module.py:202: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "bool"  [arg-type]
build/lib/rfdetr/lightning_module.py:202: error: Argument 1 to "autocast" has incompatible type "**dict[str, Union[dtype, str, Any, bool]]"; expected "Optional[bool]"  [arg-type]
build/lib/rfdetr/lightning_module.py:228: error: Call to untyped function "reduce_dict" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:265: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:295: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:318: error: Item "Tensor" of "Union[Tensor, Module, Any]" has no attribute "eval"  [union-attr]
build/lib/rfdetr/lightning_module.py:325: error: Item "Tensor" of "Union[Tensor, Module, Any]" has no attribute "state_dict"  [union-attr]
build/lib/rfdetr/lightning_module.py:348: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:348: note: Use "-> None" if function does not return a value
build/lib/rfdetr/lightning_module.py:355: error: "Trainer" has no attribute "datamodule"  [attr-defined]
build/lib/rfdetr/lightning_module.py:356: error: Call to untyped function "get_coco_api_from_dataset" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:358: error: Call to untyped function "CocoEvaluator" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:363: error: Incompatible types in assignment (expression has type "None", variable has type "CocoEvaluator")  [assignment]
build/lib/rfdetr/lightning_module.py:368: error: Call to untyped function "export_model" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:370: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:370: error: Signature of "on_validation_batch_end" incompatible with supertype "ModelHooks"  [override]
build/lib/rfdetr/lightning_module.py:370: note:      Superclass:
build/lib/rfdetr/lightning_module.py:370: note:          def on_validation_batch_end(self, outputs: Union[Tensor, Mapping[str, Any], None], batch: Any, batch_idx: int, dataloader_idx: int = ...) -> None
build/lib/rfdetr/lightning_module.py:370: note:      Subclass:
build/lib/rfdetr/lightning_module.py:370: note:          def on_validation_batch_end(self, outputs: Any, batch: Any, batch_idx: Any) -> Any
build/lib/rfdetr/lightning_module.py:374: error: Call to untyped function "update" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:378: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:378: note: Use "-> None" if function does not return a value
build/lib/rfdetr/lightning_module.py:394: error: Call to untyped function "synchronize_between_processes" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:397: error: Call to untyped function "accumulate" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:398: error: Call to untyped function "summarize" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:438: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:445: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:489: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:501: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:504: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:504: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
build/lib/rfdetr/lightning_module.py:507: error: Call to untyped function "build_dataset" in typed context  [no-untyped-call]
build/lib/rfdetr/lightning_module.py:507: error: Unexpected keyword argument "resolution" for "build_dataset"  [call-arg]
rfdetr/datasets/__init__.py:29: note: "build_dataset" defined here
build/lib/rfdetr/lightning_module.py:511: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:513: error: Item "None" of "Optional[Trainer]" has no attribute "world_size"  [union-attr]
build/lib/rfdetr/lightning_module.py:514: error: Need type annotation for "sampler_train"  [var-annotated]
build/lib/rfdetr/lightning_module.py:516: error: Incompatible types in assignment (expression has type "RandomSampler", variable has type "DistributedSampler[Any]")  [assignment]
build/lib/rfdetr/lightning_module.py:531: error: Function is missing a return type annotation  [no-untyped-def]
build/lib/rfdetr/lightning_module.py:533: error: Item "None" of "Optional[Trainer]" has no attribute "world_size"  [union-attr]
build/lib/rfdetr/lightning_module.py:534: error: Need type annotation for "sampler_val"  [var-annotated]
build/lib/rfdetr/lightning_module.py:536: error: Incompatible types in assignment (expression has type "SequentialSampler", variable has type "DistributedSampler[Any]")  [assignment]
build/lib/rfdetr/hooks/onnx_checkpoint_hook.py:30: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/hooks/onnx_checkpoint_hook.py:61: error: Function is missing a type annotation  [no-untyped-def]
build/lib/rfdetr/hooks/onnx_checkpoint_hook.py:93: error: Function is missing a type annotation  [no-untyped-def]
test_evaluation.py:16: error: Function is missing a return type annotation  [no-untyped-def]
test_evaluation.py:16: note: Use "-> None" if function does not return a value
test_evaluation.py:22: error: Function is missing a return type annotation  [no-untyped-def]
test_evaluation.py:22: note: Use "-> None" if function does not return a value
test_evaluation.py:133: error: Call to untyped function "Args" in typed context  [no-untyped-call]
test_evaluation.py:141: error: Call to untyped function "RFDETRLightningModule" in typed context  [no-untyped-call]
test_evaluation.py:142: error: Call to untyped function "RFDETRDataModule" in typed context  [no-untyped-call]
test_evaluation.py:146: error: Call to untyped function "setup" in typed context  [no-untyped-call]
test_evaluation.py:178: error: Call to untyped function "main" in typed context  [no-untyped-call]
scripts/train.py:124: error: Call to untyped function "RFDETRLightningModule" in typed context  [no-untyped-call]
scripts/train.py:125: error: Call to untyped function "RFDETRDataModule" in typed context  [no-untyped-call]
scripts/train.py:145: error: Argument 1 to "append" of "list" has incompatible type "WandbLogger"; expected "TensorBoardLogger"  [arg-type]
scripts/train.py:152: error: Argument 1 to "append" of "list" has incompatible type "CSVLogger"; expected "TensorBoardLogger"  [arg-type]
scripts/train.py:190: error: Call to untyped function "ONNXCheckpointHook" in typed context  [no-untyped-call]
scripts/train.py:203: error: Argument 1 to "append" of "list" has incompatible type "ONNXCheckpointHook"; expected "ModelCheckpoint"  [arg-type]
scripts/train.py:207: error: Argument 1 to "append" of "list" has incompatible type "LearningRateMonitor"; expected "ModelCheckpoint"  [arg-type]
scripts/train.py:211: error: Argument 1 to "append" of "list" has incompatible type "TQDMProgressBar"; expected "ModelCheckpoint"  [arg-type]
scripts/train.py:223: error: Argument 1 to "append" of "list" has incompatible type "EarlyStopping"; expected "ModelCheckpoint"  [arg-type]
scripts/train.py:228: error: Incompatible types in assignment (expression has type "DDPStrategy", variable has type "str")  [assignment]
scripts/train.py:236: error: Argument "callbacks" to "Trainer" has incompatible type "list[ModelCheckpoint]"; expected "Union[list[Callback], Callback, None]"  [arg-type]
scripts/train.py:236: note: "List" is invariant -- see https://mypy.readthedocs.io/en/stable/common_issues.html#variance
scripts/train.py:236: note: Consider using "Sequence" instead, which is covariant
scripts/train.py:300: error: Call to untyped function "get_sha" in typed context  [no-untyped-call]
scripts/train.py:314: error: Call to untyped function "main" in typed context  [no-untyped-call]
Found 2751 errors in 112 files (checked 139 source files)
