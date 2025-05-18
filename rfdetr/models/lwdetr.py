# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
LW-DETR model and criterion classes
"""

import copy
import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from rfdetr.models.backbone import build_backbone
from rfdetr.models.matcher import build_matcher
from rfdetr.models.transformer import build_transformer
from rfdetr.util import box_ops
from rfdetr.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)


class LWDETR(nn.Module):
    """This is the Group DETR v3 module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        group_detr=1,
        two_stage=False,
        lite_refpoint_refine=False,
        bbox_reparam=False,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            group_detr: Number of groups to speed detr training. Default is 1.
            lite_refpoint_refine: TODO
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        query_dim = 4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        # iter update
        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # Add mask head (28x28 masks, similar to Mask R-CNN)
        self.mask_embed = MLP(hidden_dim, hidden_dim, 28 * 28, 3)
        nn.init.constant_(self.mask_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.mask_embed.layers[-1].bias.data, 0)

        # two_stage
        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)]
            )
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)]
            )

        self._export = False

    def reinitialize_detection_head(self, num_classes):
        # Create new classification head
        del self.class_embed
        self.add_module("class_embed", nn.Linear(self.transformer.d_model, num_classes))

        # Initialize with focal loss bias adjustment
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        if self.two_stage:
            del self.transformer.enc_out_class_embed
            self.transformer.add_module(
                "enc_out_class_embed",
                nn.ModuleList([copy.deepcopy(self.class_embed) for _ in range(self.group_detr)]),
            )

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for _name, m in self.named_modules():
            if (
                hasattr(m, "export")
                and isinstance(m.export, Callable)
                and hasattr(m, "_export")
                and not m._export
            ):
                m.export()

    def forward(self, samples: NestedTensor, targets=None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for _l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            # only use one group in inference
            refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
            query_feat_weight = self.query_feat.weight[: self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight
        )

        if self.bbox_reparam:
            outputs_coord_delta = self.bbox_embed(hs)
            outputs_coord_cxcy = (
                outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            )
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
        else:
            outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

        outputs_class = self.class_embed(hs)

        # Generate mask predictions from the same features (in FP16)
        outputs_mask = (
            self.mask_embed(hs).reshape(hs.shape[0], hs.shape[1], hs.shape[2], 28, 28).half()
        )

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_masks": outputs_mask[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord, outputs_mask)

        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                cls_enc.append(cls_enc_gidx)
            cls_enc = torch.cat(cls_enc, dim=1)
            out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
        return out

    def forward_export(self, tensors):
        srcs, _, poss = self.backbone(tensors)
        # only use one group in inference
        refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
        query_feat_weight = self.query_feat.weight[: self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, None, poss, refpoint_embed_weight, query_feat_weight
        )

        if self.bbox_reparam:
            outputs_coord_delta = self.bbox_embed(hs)
            outputs_coord_cxcy = (
                outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            )
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
        else:
            outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
        outputs_class = self.class_embed(hs)
        outputs_mask = (
            self.mask_embed(hs).reshape(hs.shape[0], hs.shape[1], hs.shape[2], 28, 28).half()
        )
        return outputs_coord, outputs_class, outputs_mask, ref_unsigmoid

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_mask is not None:
            return [
                {"pred_logits": a, "pred_boxes": b, "pred_masks": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

    def update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        """ """
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]
        for i in range(vit_encoder_num_layers):
            if hasattr(self.backbone[0].encoder, "blocks"):  # Not aimv2
                if hasattr(self.backbone[0].encoder.blocks[i].drop_path, "drop_prob"):
                    self.backbone[0].encoder.blocks[i].drop_path.drop_prob = dp_rates[i]
            else:  # aimv2
                if hasattr(self.backbone[0].encoder.trunk.blocks[i].drop_path, "drop_prob"):
                    self.backbone[0].encoder.trunk.blocks[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate):
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


class SetCriterion(nn.Module):
    """This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        focal_alpha,
        losses,
        group_detr=1,
        sum_group_losses=False,
        use_varifocal_loss=False,
        use_position_supervised_loss=False,
        ia_bce_loss=False,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            group_detr: Number of groups to speed detr training. Default is 1.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.group_detr = group_detr
        self.sum_group_losses = sum_group_losses
        self.use_varifocal_loss = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss = ia_bce_loss

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            alpha = self.focal_alpha
            gamma = 2
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets = torch.diag(
                box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_ops.box_cxcywh_to_xyxy(target_boxes),
                )[0]
            )
            pos_ious = iou_targets.clone().detach()
            prob = src_logits.sigmoid()
            # init positive weights and negative weights
            pos_weights = torch.zeros_like(src_logits)
            neg_weights = prob**gamma

            pos_ind = list(idx)
            pos_ind.append(target_classes_o)

            t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()

            pos_weights[pos_ind] = t.to(pos_weights.dtype)
            neg_weights[pos_ind] = 1 - t.to(neg_weights.dtype)
            # a reformulation of the standard loss_ce = - pos_weights * prob.log() - neg_weights * (1 - prob).log()
            # with a focus on statistical stability by using fused logsigmoid
            loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (
                pos_weights + neg_weights
            )
            loss_ce = loss_ce.sum() / num_boxes

        elif self.use_position_supervised_loss:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets = torch.diag(
                box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_ops.box_cxcywh_to_xyxy(target_boxes),
                )[0]
            )
            pos_ious = iou_targets.clone().detach()
            # pos_ious_func = pos_ious ** 2
            pos_ious_func = pos_ious

            cls_iou_func_targets = torch.zeros(
                (src_logits.shape[0], src_logits.shape[1], self.num_classes),
                dtype=src_logits.dtype,
                device=src_logits.device,
            )

            pos_ind = list(idx)
            pos_ind.append(target_classes_o)
            cls_iou_func_targets[pos_ind] = pos_ious_func
            norm_cls_iou_func_targets = cls_iou_func_targets / (
                cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(1, True) + 1e-8
            )
            loss_ce = (
                position_supervised_loss(
                    src_logits,
                    norm_cls_iou_func_targets,
                    num_boxes,
                    alpha=self.focal_alpha,
                    gamma=2,
                )
                * src_logits.shape[1]
            )

        elif self.use_varifocal_loss:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets = torch.diag(
                box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_ops.box_cxcywh_to_xyxy(target_boxes),
                )[0]
            )
            pos_ious = iou_targets.clone().detach()

            cls_iou_targets = torch.zeros(
                (src_logits.shape[0], src_logits.shape[1], self.num_classes),
                dtype=src_logits.dtype,
                device=src_logits.device,
            )

            pos_ind = list(idx)
            pos_ind.append(target_classes_o)
            cls_iou_targets[pos_ind] = pos_ious
            loss_ce = (
                sigmoid_varifocal_loss(
                    src_logits, cls_iou_targets, num_boxes, alpha=self.focal_alpha, gamma=2
                )
                * src_logits.shape[1]
            )
        else:
            target_classes = torch.full(
                src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
            )
            target_classes[idx] = target_classes_o

            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_ce = (
                sigmoid_focal_loss(
                    src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2
                )
                * src_logits.shape[1]
            )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        self._get_tgt_permutation_idx(indices)

        # Extract predicted masks for each matched query
        src_masks = outputs["pred_masks"][src_idx]

        # Check if masks are available in targets
        has_masks = all("masks" in t for t, (_, i) in zip(targets, indices))
        if not has_masks:
            # Return zero loss if masks are not available
            return {"loss_mask": torch.as_tensor(0.0, device=src_masks.device)}

        # Process each mask individually and resize to a common size before concatenation
        resized_masks = []
        for t, (_, i) in zip(targets, indices):
            if len(i) > 0:  # Skip if no indices for this batch item
                # Get masks for current target and resize them all to 28x28
                masks = t["masks"][i]  # Shape: [num_instances, h, w]
                # Resize masks to 28x28 in batches to save memory
                batch_size = 32  # Process in batches
                num_masks = masks.shape[0]
                resized_batch = []

                for j in range(0, num_masks, batch_size):
                    # Process a batch of masks
                    mask_batch = masks[j : j + batch_size]
                    # Resize this batch
                    # Use float32 for numerical stability
                    batch_resized = (
                        F.interpolate(
                            mask_batch.unsqueeze(1).float(),
                            size=(28, 28),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(1)
                        .gt(0.5)
                    )
                    resized_batch.append(batch_resized)

                # Concatenate the resized batches
                masks = (
                    torch.cat(resized_batch, dim=0)
                    if resized_batch
                    else torch.zeros((0, 28, 28), dtype=torch.bool, device=masks.device)
                )
                resized_masks.append(masks)

        # Concatenate all resized masks
        if resized_masks:
            target_masks = torch.cat(resized_masks, dim=0)
        else:
            # Return zero loss if no valid masks
            return {"loss_mask": torch.as_tensor(0.0, device=src_masks.device)}

        # Compute dice loss
        src_masks = src_masks.flatten(1)  # [num_matched_queries, 28*28]
        target_masks = target_masks.flatten(1)  # [num_matched_queries, 28*28]

        # Convert masks to float32 for numerical stability
        src_masks = src_masks.float()
        target_masks = target_masks.float()

        # Dice loss with numerical stability
        numerator = 2 * (src_masks * target_masks).sum(1)
        denominator = src_masks.sum(1) + target_masks.sum(1)

        # Clamp denominator to avoid division by very small numbers
        denominator = torch.clamp(denominator, min=1e-6)

        # Compute dice loss with stability
        loss_dice = 1 - (numerator + 1) / (denominator + 1)

        # Check for inf/nan values and handle them
        if torch.isnan(loss_dice).any() or torch.isinf(loss_dice).any():
            print("Warning: inf/nan in dice loss, replacing with zero")
            loss_dice = torch.where(
                torch.isnan(loss_dice) | torch.isinf(loss_dice),
                torch.zeros_like(loss_dice),
                loss_dice,
            )

        # Clamp the loss to reasonable range
        loss_dice = torch.clamp(loss_dice, min=0.0, max=10.0)

        losses = {"loss_mask": loss_dice.sum() / max(num_boxes, 1)}

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            # Add pred_masks key if missing for compatibility with mask loss
            if "masks" in self.losses and "pred_masks" not in enc_outputs:
                enc_outputs["pred_masks"] = torch.zeros(
                    (outputs["pred_masks"].shape[0], enc_outputs["pred_boxes"].shape[1], 28, 28),
                    device=outputs["pred_masks"].device,
                )
            indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs["log"] = False
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def sigmoid_varifocal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    focal_weight = (
        targets * (targets > 0.0).float()
        + (1 - alpha) * (prob - targets).abs().pow(gamma) * (targets <= 0.0).float()
    )
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * focal_weight

    return loss.mean(1).sum() / num_boxes


def position_supervised_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * (torch.abs(targets - prob) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        out_mask = outputs.get("pred_masks", None)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), self.num_select, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for i, (s, l, b) in enumerate(zip(scores, labels, boxes)):
            result = {"scores": s, "labels": l, "boxes": b}

            # Include mask predictions if available
            if out_mask is not None:
                # Get the top-k masks
                masks_per_image = out_mask[i]
                masks = torch.gather(
                    masks_per_image, 0, topk_boxes[i].unsqueeze(-1).unsqueeze(-1).repeat(1, 28, 28)
                )

                # Resize masks to original image size with batching to reduce memory consumption
                batch_size = 32  # Process masks in batches of this size
                num_masks = masks.shape[0]
                resized_masks = []

                for j in range(0, num_masks, batch_size):
                    # Process a batch of masks
                    batch_masks = masks[j : j + batch_size]
                    # Ensure FP16 for interpolation
                    batch_resized = F.interpolate(
                        batch_masks.unsqueeze(1).half(),
                        size=(int(img_h[i].item()), int(img_w[i].item())),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    resized_masks.append(batch_resized)

                # Concatenate batches back together
                if resized_masks:
                    result["masks"] = torch.cat(resized_masks, dim=0)
                else:
                    result["masks"] = torch.zeros(
                        (0, int(img_h[i].item()), int(img_w[i].item())), device=masks.device
                    )

            results.append(result)

        return results


class PostProcessSegm(nn.Module):
    """
    Post-processes the mask predictions from the model into a format suitable for
    evaluation and visualization.
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        Process the mask outputs to create high-resolution binary masks.

        Args:
            outputs: Dict containing the model outputs including 'pred_masks'
            target_sizes: Tensor of shape [batch_size, 2] containing the size of each image

        Returns:
            List of dicts with mask predictions for each image
        """
        out_logits, _out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        assert "pred_masks" in outputs, "Masks not found in model outputs"
        out_masks = outputs["pred_masks"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        # Extract masks for top-k predictions
        batch_size = out_masks.shape[0]
        results = []

        for i in range(batch_size):
            masks = out_masks[i, topk_boxes[i]]  # [num_queries, H/4, W/4]

            # Upsample masks to original image size with batching for memory efficiency
            img_h, img_w = target_sizes[i]
            batch_size = 32  # Process masks in batches of this size
            num_masks = masks.shape[0]
            resized_masks = []

            for j in range(0, num_masks, batch_size):
                # Process a batch of masks
                batch_masks = masks[j : j + batch_size]
                # Ensure FP16 for interpolation
                batch_resized = F.interpolate(
                    batch_masks.unsqueeze(1).half(),
                    size=(int(img_h.item()), int(img_w.item())),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                # Apply threshold immediately to save memory
                batch_resized = batch_resized > self.threshold
                resized_masks.append(batch_resized)

            # Concatenate batches back together
            if resized_masks:
                masks = torch.cat(resized_masks, dim=0)
            else:
                masks = torch.zeros(
                    (0, int(img_h.item()), int(img_w.item())), dtype=torch.bool, device=masks.device
                )

            results.append(
                {
                    "masks": masks,
                    "scores": scores[i],
                    "labels": labels[i],
                }
            )

        return results


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(config):
    """
    Build the RF-DETR model using a Pydantic ModelConfig object.
    
    Args:
        config: A Pydantic ModelConfig instance containing all model parameters.
               For backward compatibility, can also accept a dict or an object with attributes.
    
    Returns:
        A configured LWDETR model instance.
    """
    # Import here to avoid circular imports
    from rfdetr.model_config import ModelConfig
    
    # Handle different types of input for backward compatibility
    if not isinstance(config, ModelConfig):
        # If a dict is passed, convert to ModelConfig
        if isinstance(config, dict):
            config = ModelConfig(**config)
        else:
            # If an object with attributes is passed, convert to ModelConfig
            # by first converting to dict
            config_dict = {
                k: getattr(config, k) for k in dir(config) 
                if not k.startswith('_') and not callable(getattr(config, k))
            }
            config = ModelConfig(**config_dict)
    
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    num_classes = config.num_classes + 1
    device = torch.device(config.device)
    
    # Determine target shape
    target_shape = config.shape if config.shape else (config.resolution, config.resolution)
    
    backbone = build_backbone(
        encoder=config.encoder,
        vit_encoder_num_layers=config.vit_encoder_num_layers,
        pretrained_encoder=config.pretrained_encoder,
        window_block_indexes=config.window_block_indexes,
        drop_path=config.drop_path,
        out_channels=config.hidden_dim,
        out_feature_indexes=config.out_feature_indexes,
        projector_scale=config.projector_scale,
        use_cls_token=config.use_cls_token,
        hidden_dim=config.hidden_dim,
        position_embedding=config.position_embedding,
        freeze_encoder=config.freeze_encoder,
        layer_norm=config.layer_norm,
        target_shape=target_shape,
        rms_norm=config.rms_norm,
        backbone_lora=config.backbone_lora,
        force_no_pretrain=config.force_no_pretrain,
        gradient_checkpointing=config.gradient_checkpointing,
        load_dinov2_weights=config.pretrain_weights is None,
    )
    
    # Special return cases
    if hasattr(config, "encoder_only") and config.encoder_only:
        return backbone[0].encoder, None, None
    if hasattr(config, "backbone_only") and config.backbone_only:
        return backbone, None, None

    # Set num_feature_levels based on projector_scale length
    feature_levels = len(config.projector_scale)
    
    # Create a dict version of the config for transformer (will be updated later)
    transformer_config = config.dict_for_model_build()
    transformer_config["num_feature_levels"] = feature_levels
    
    # Set default transformer parameters if they don't exist
    if "decoder_norm" not in transformer_config:
        transformer_config["decoder_norm"] = "LN"
    if "dec_n_points" not in transformer_config:
        transformer_config["dec_n_points"] = 4
    
    # Convert dict to an object with attribute access
    class AttributeDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttributeDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    
    transformer_config_obj = AttributeDict(transformer_config)
    transformer = build_transformer(transformer_config_obj)

    model = LWDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=config.num_queries,
        aux_loss=config.aux_loss,
        group_detr=config.group_detr,
        two_stage=config.two_stage,
        lite_refpoint_refine=config.lite_refpoint_refine,
        bbox_reparam=config.bbox_reparam,
    )
    return model


def build_criterion_and_postprocessors(config):
    """
    Build the criterion and postprocessors for model training and inference.
    
    Args:
        config: A Pydantic ModelConfig instance containing all model parameters.
               For backward compatibility, can also accept a dict or an object with attributes.
    
    Returns:
        Tuple of (criterion, postprocessors)
    """
    # Import here to avoid circular imports
    from rfdetr.model_config import ModelConfig
    
    # Handle different types of input for backward compatibility
    if not isinstance(config, ModelConfig):
        # If a dict is passed, convert to ModelConfig
        if isinstance(config, dict):
            config = ModelConfig(**config)
        else:
            # If an object with attributes is passed, convert to ModelConfig
            # by first converting to dict
            config_dict = {
                k: getattr(config, k) for k in dir(config) 
                if not k.startswith('_') and not callable(getattr(config, k))
            }
            config = ModelConfig(**config_dict)
    
    device = torch.device(config.device)
    matcher = build_matcher(config)
    
    # Set up weight dictionary
    weight_dict = {
        "loss_ce": config.cls_loss_coef, 
        "loss_bbox": config.bbox_loss_coef,
        "loss_giou": config.giou_loss_coef,
        "loss_mask": 1.0  # Default weight for mask loss
    }
    
    # Add auxiliary loss weights if enabled
    if config.aux_loss:
        aux_weight_dict = {}
        for i in range(config.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        if config.two_stage:
            aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # Define losses to compute
    losses = ["labels", "boxes", "cardinality", "masks"]

    # Create criterion
    criterion = SetCriterion(
        config.num_classes + 1,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=config.focal_alpha,
        losses=losses,
        group_detr=config.group_detr,
        sum_group_losses=config.sum_group_losses,
        use_varifocal_loss=config.use_varifocal_loss,
        use_position_supervised_loss=config.use_position_supervised_loss,
        ia_bce_loss=config.ia_bce_loss,
    )
    criterion.to(device)
    
    # Set up postprocessors
    postprocessors = {"bbox": PostProcess(num_select=config.num_select)}
    
    # Add segmentation postprocessor
    postprocessors.update({"segm": PostProcessSegm()})

    return criterion, postprocessors
