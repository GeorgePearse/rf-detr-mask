# RF-DETR-MASK: Instance Segmentation Extension of RF-DETR

RF-DETR-MASK is an instance segmentation extension of the RF-DETR architecture, enabling pixel-precise object delineation in addition to bounding box detection. This variant adds a mask prediction head to the original RF-DETR model while maintaining its real-time performance characteristics.

Building on the foundation of RF-DETR, which exceeds 60 AP on the [Microsoft COCO benchmark](https://cocodataset.org/#home), this extension adds instance segmentation capabilities with minimal computational overhead. Like its parent architecture, RF-DETR-MASK maintains competitive speed and accuracy, making it suitable for both detection and segmentation tasks.

The segmentation head architecture is inspired by [Facebook DETR's segmentation implementation](https://github.com/facebookresearch/detr/blob/main/models/segmentation.py).

**RF-DETR-MASK combines the edge-friendly performance of RF-DETR with the added capability of producing instance masks, ideal for applications requiring precise object boundaries.**


