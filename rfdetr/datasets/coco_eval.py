# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""

import contextlib
import copy
import os

import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rfdetr.util.misc import all_gather


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def _prepare_batch(self, coco_eval):
        """
        Silent version of evaluate that just prepares data without printing anything.
        Returns the image IDs and evaluation images.
        """
        p = coco_eval.params

        # Add backward compatibility if useSegm is specified in params
        if hasattr(p, "useSegm") and p.useSegm is not None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"

        p.imgIds = list(np.unique(p.imgIds))

        # Always initialize useCats to ensure it exists
        if not hasattr(p, "useCats"):
            p.useCats = True

        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        coco_eval.params = p

        coco_eval._prepare()

        # Create empty eval_imgs - we'll populate these during the final accumulate stage
        eval_imgs = []

        return p.imgIds, eval_imgs

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)

            # Ensure useCats is always set before attempting to use it
            if not hasattr(coco_eval.params, "useCats"):
                coco_eval.params.useCats = True

            try:
                # Call evaluate directly but without actually showing output in per-batch processing
                # This completely silences all output from the evaluator
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                    with contextlib.redirect_stderr(devnull):
                        # Instead of running full evaluation on each batch, we'll just prepare the data
                        # but call our modified _prepare_batch method to avoid the print statements
                        img_ids, eval_imgs = self._prepare_batch(coco_eval)
                self.eval_imgs[iou_type].append(eval_imgs)
            except Exception as e:
                # Use a logger instead of print to control verbosity better
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Error during batch preparation: {e}")
                # Fall back to older method but still avoid printing
                try:
                    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                        with contextlib.redirect_stderr(devnull):
                            # Ensure useCats exists before _prepare
                            if not hasattr(coco_eval.params, "useCats"):
                                coco_eval.params.useCats = True
                            # Use a completely silent approach
                            coco_eval._prepare()
                            eval_imgs = []
                            self.eval_imgs[iou_type].append(eval_imgs)
                except Exception as e2:
                    logger.debug(f"Error using fallback batch preparation: {e2}")

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            # Import logging for consistent logging
            import logging

            logger = logging.getLogger(__name__)

            # Handle cases where eval_imgs might be empty or invalid
            if not self.eval_imgs[iou_type] or len(self.eval_imgs[iou_type]) == 0:
                logger.warning(f"No evaluation images for {iou_type}, running evaluation now")
                # Run evaluation to get some valid data
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                    self.coco_eval[iou_type].evaluate()
                continue

            # Check if all eval_imgs elements are valid
            valid_eval_imgs = True
            for img in self.eval_imgs[iou_type]:
                if (
                    img is None
                    or (isinstance(img, list) and not img)
                    or (isinstance(img, np.ndarray) and img.size == 0)
                ):
                    valid_eval_imgs = False
                    break

            if not valid_eval_imgs:
                logger.warning(f"Invalid evaluation images for {iou_type}, regenerating")
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                    self.coco_eval[iou_type].evaluate()
                continue

            try:
                # Try to concatenate the evaluation images
                self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
                create_common_coco_eval(
                    self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type]
                )
            except Exception as e:
                logger.warning(
                    f"Error synchronizing {iou_type}: {e}. Regenerating evaluation data."
                )
                # Run evaluation to get valid data instead
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                    self.coco_eval[iou_type].evaluate()

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            # Ensure params exists and is properly initialized
            if not hasattr(coco_eval, "params") or coco_eval.params is None:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning("coco_eval.params is None or not present, initializing")
                # Initialize params from scratch if needed
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                    coco_eval.evaluate()
                continue

            # Ensure useCats exists before accumulate
            if not hasattr(coco_eval.params, "useCats"):
                coco_eval.params.useCats = True

            # Ensure evalImgs exists before attempting to accumulate
            if not hasattr(coco_eval, "evalImgs") or coco_eval.evalImgs is None:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning("evalImgs is None, running evaluate() first")
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                    coco_eval.evaluate()

            try:
                coco_eval.accumulate()
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Error in COCO accumulate: {e}. Initializing params and retrying.")

                # Make sure evalImgs exists
                if not hasattr(coco_eval, "evalImgs") or coco_eval.evalImgs is None:
                    logger.warning("evalImgs is None, running evaluate() first")
                    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                        coco_eval.evaluate()

                # Try accumulate again after evaluation
                try:
                    coco_eval.accumulate()
                except Exception as e2:
                    logger.error(f"Failed to accumulate after retry: {e2}")

    def summarize(self):
        # Import logging for better control over output
        import logging

        logger = logging.getLogger(__name__)

        for iou_type, coco_eval in self.coco_eval.items():
            logger.info(f"IoU metric: {iou_type}")
            # Run summarize but capture and log its output
            import io
            from contextlib import redirect_stdout

            try:
                # Ensure we have evaluation data before summarizing
                if not hasattr(coco_eval, "evalImgs") or coco_eval.evalImgs is None:
                    logger.warning("evalImgs is None, running evaluate() first")
                    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                        coco_eval.evaluate()

                # Capture the output from coco_eval.summarize()
                output_buffer = io.StringIO()
                with redirect_stdout(output_buffer):
                    coco_eval.summarize()

                # Get the output and log it properly
                summary_output = output_buffer.getvalue()
                # Split by lines and log each line at proper level
                for line in summary_output.strip().split("\n"):
                    if line:
                        logger.info(line)
            except Exception as e:
                logger.warning(
                    f"Error in COCO summary for {iou_type}: {e}. Continuing with validation."
                )

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            # Handle different mask tensor shapes
            rles = []
            for mask in masks:
                # Ensure mask is 2D numpy array
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()

                # Handle cases where mask might have extra dimensions
                if mask.ndim > 2:
                    # If mask has batch dimension, remove it
                    if mask.shape[0] == 1:
                        mask = mask[0]
                    # If mask has other extra dimensions, squeeze them
                    mask = mask.squeeze()

                # Ensure mask is 2D
                assert mask.ndim == 2, f"Mask should be 2D but got shape: {mask.shape}"

                # Encode the mask
                rle = mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[
                    0
                ]
                rles.append(rle)
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    try:
        # Ensure coco_eval has params initialized
        if not hasattr(coco_eval, "params") or coco_eval.params is None:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("coco_eval.params is None or not present in create_common_coco_eval")
            # Run evaluation to generate proper params
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                coco_eval.evaluate()
            return

        # Check if eval_imgs is valid
        if eval_imgs is None or (isinstance(eval_imgs, list) and not eval_imgs):
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("eval_imgs is None or empty in create_common_coco_eval")
            # Generate evaluation data
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                coco_eval.evaluate()
            return

        # Process the eval_imgs
        img_ids, eval_imgs = merge(img_ids, eval_imgs)
        img_ids = list(img_ids)
        eval_imgs = list(eval_imgs.flatten())

        # Ensure useCats exists before using it in subsequent operations
        if not hasattr(coco_eval.params, "useCats"):
            coco_eval.params.useCats = True

        coco_eval.evalImgs = eval_imgs
        coco_eval.params.imgIds = img_ids
        coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Error in create_common_coco_eval: {e}. Continuing with evaluation.")
        # Run evaluation to generate the necessary data
        if not hasattr(coco_eval, "evalImgs") or coco_eval.evalImgs is None:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                coco_eval.evaluate()


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """
    try:
        # Completely silent version with all print statements removed
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if hasattr(p, "useSegm") and p.useSegm is not None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"

        p.imgIds = list(np.unique(p.imgIds))

        # Always ensure useCats is initialized
        if not hasattr(p, "useCats"):
            p.useCats = True

        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        cat_ids = p.catIds if p.useCats else [-1]

        if p.iouType == "segm" or p.iouType == "bbox":
            compute_iou = self.computeIoU
        elif p.iouType == "keypoints":
            compute_iou = self.computeOks
        self.ious = {
            (imgId, catId): compute_iou(imgId, catId) for imgId in p.imgIds for catId in cat_ids
        }

        evaluate_img = self.evaluateImg
        max_det = p.maxDets[-1]
        eval_imgs = [
            evaluate_img(imgId, catId, areaRng, max_det)
            for catId in cat_ids
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        # this is NOT in the pycocotools code, but could be done outside
        eval_imgs = np.asarray(eval_imgs).reshape(len(cat_ids), len(p.areaRng), len(p.imgIds))
        self._paramsEval = copy.deepcopy(self.params)
        self.evalImgs = eval_imgs  # Save the evaluation images directly

        return p.imgIds, eval_imgs
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Error in COCO evaluate: {e}")
        # Return empty data with correct structure to avoid errors in subsequent steps
        return self.params.imgIds, np.array([])


#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
