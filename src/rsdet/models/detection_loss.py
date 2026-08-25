"""Hungarian matching and training losses for BHC-DETR.

The paper uses focal classification, rotated IoU and L1 costs.  XH-202625
annotations are horizontal ``xyxy`` boxes without an angle, therefore this
task-compatible implementation uses generalized IoU for horizontal boxes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rsdet.models.bhcl import BalancedHierarchicalContrastiveLoss
from rsdet.models.hierarchy import XH_HIERARCHY
from rsdet.models.uhr_small_object import (
    build_gain_map_targets,
    distribution_focal_gain_loss,
    local_peak_margin_loss,
)


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        (
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
            x2 - x1,
            y2 - y1,
        ),
        dim=-1,
    )


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(
        min=0
    )


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-7), union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    iou, union = box_iou(boxes1, boxes2)
    top_left = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    return iou - (enclosing - union) / enclosing.clamp(min=1e-7)


def sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    alpha: float,
    gamma: float,
) -> Tensor:
    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    target_probability = probability * targets + (1.0 - probability) * (1.0 - targets)
    loss = cross_entropy * (1.0 - target_probability).pow(gamma)
    if alpha >= 0.0:
        alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_factor * loss
    return loss


@dataclass(frozen=True)
class BHCDetrLossConfig:
    match_class_cost: float = 2.0
    match_bbox_cost: float = 5.0
    match_giou_cost: float = 2.0
    class_loss_weight: float = 2.0
    bbox_loss_weight: float = 5.0
    giou_loss_weight: float = 2.0
    bhcl_weight: float = 0.6
    auxiliary_loss_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    temperature: float = 0.1
    prototype_momentum: float = 0.1
    gain_map_weight: float = 1.0
    gain_lpm_weight: float = 1.0
    gain_lpm_margin: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            self.match_class_cost,
            self.match_bbox_cost,
            self.match_giou_cost,
            self.class_loss_weight,
            self.bbox_loss_weight,
            self.giou_loss_weight,
            self.bhcl_weight,
            self.gain_map_weight,
            self.gain_lpm_weight,
        )
        if any(value < 0.0 for value in positive) or not any(positive[:3]):
            raise ValueError("matching/loss weights must be non-negative")
        if self.auxiliary_loss_weight < 0.0:
            raise ValueError("auxiliary_loss_weight must be non-negative")
        if not 0.0 <= self.focal_alpha <= 1.0 or self.focal_gamma < 0.0:
            raise ValueError("invalid focal loss parameters")
        if self.temperature <= 0.0 or not 0.0 < self.prototype_momentum <= 1.0:
            raise ValueError("temperature must be positive and prototype_momentum in (0,1]")
        if self.gain_lpm_margin < 0.0:
            raise ValueError("gain_lpm_margin must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "BHCDetrLossConfig":
        raw = dict(value or {})
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown BHCDetr loss fields: {unknown}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HungarianMatcher(nn.Module):
    """One-to-one matching from paper Eq. (1)."""

    def __init__(self, config: BHCDetrLossConfig) -> None:
        super().__init__()
        self.class_cost = config.match_class_cost
        self.bbox_cost = config.match_bbox_cost
        self.giou_cost = config.match_giou_cost
        self.alpha = config.focal_alpha
        self.gamma = config.focal_gamma

    @torch.no_grad()
    def _cost_matrix(
        self,
        outputs: Mapping[str, Tensor],
        target: Mapping[str, Tensor],
        batch_index: int,
    ) -> Tensor:
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        target_labels = target["labels"]
        target_boxes = target["boxes"]
        eps = 1e-8
        probability = logits[batch_index].sigmoid()
        negative_cost = (
            (1.0 - self.alpha)
            * probability.pow(self.gamma)
            * -torch.log((1.0 - probability).clamp(min=eps))
        )
        positive_cost = (
            self.alpha
            * (1.0 - probability).pow(self.gamma)
            * -torch.log(probability.clamp(min=eps))
        )
        classification = positive_cost[:, target_labels] - negative_cost[:, target_labels]
        bbox = torch.cdist(boxes[batch_index], target_boxes, p=1)
        giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(boxes[batch_index]),
            box_cxcywh_to_xyxy(target_boxes),
        )
        return self.class_cost * classification + self.bbox_cost * bbox + self.giou_cost * giou

    @torch.no_grad()
    def match_layers(
        self,
        outputs: Sequence[Mapping[str, Tensor]],
        targets: Sequence[Mapping[str, Tensor]],
    ) -> list[list[tuple[Tensor, Tensor]]]:
        """Match several decoder layers with a single GPU-to-CPU synchronization.

        SciPy still solves every image/layer cost matrix independently with the
        exact linear-sum assignment algorithm.  Concatenating the matrices only
        batches their device transfer; it does not mix images or decoder layers.
        """

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as error:
            raise ImportError("BHC-DETR Hungarian matching requires scipy") from error

        if not outputs:
            return []

        # Each pending item identifies one independent rectangular assignment
        # matrix.  Keeping the matrices on the accelerator until all layers have
        # queued their work avoids one blocking ``.cpu()`` per image and layer.
        pending: list[tuple[int, int, int, int, Tensor]] = []
        results: list[list[tuple[Tensor, Tensor] | None]] = [[None] * len(targets) for _ in outputs]
        common_device = outputs[0]["pred_logits"].device
        for layer_index, layer_outputs in enumerate(outputs):
            logits = layer_outputs["pred_logits"]
            boxes = layer_outputs["pred_boxes"]
            if logits.ndim != 3 or boxes.shape[:2] != logits.shape[:2]:
                raise ValueError("invalid BHC-DETR output shapes")
            if logits.device != common_device or boxes.device != common_device:
                raise ValueError("all BHC-DETR decoder outputs must use one device")
            for batch_index, target in enumerate(targets):
                target_count = int(target["labels"].numel())
                if target_count == 0:
                    empty = torch.empty(0, dtype=torch.int64, device=common_device)
                    results[layer_index][batch_index] = (empty, empty.clone())
                    continue
                cost = self._cost_matrix(layer_outputs, target, batch_index)
                pending.append((layer_index, batch_index, cost.shape[0], target_count, cost))

        if pending:
            # Exactly one synchronization and D2H copy for all non-empty cost
            # matrices.  Their flattened order is layer-major then batch-major,
            # identical to the former nested matcher calls.
            flat_costs = torch.cat([item[-1].reshape(-1) for item in pending]).cpu().numpy()
            offset = 0
            solved: list[tuple[int, int, Tensor, Tensor]] = []
            for layer_index, batch_index, query_count, target_count, _ in pending:
                element_count = query_count * target_count
                cost_matrix = flat_costs[offset : offset + element_count].reshape(
                    query_count,
                    target_count,
                )
                offset += element_count
                source_indices, target_indices = linear_sum_assignment(cost_matrix)
                solved.append(
                    (
                        layer_index,
                        batch_index,
                        torch.from_numpy(source_indices),
                        torch.from_numpy(target_indices),
                    )
                )

            # Assignments are tiny, but copying each NumPy result separately to
            # CUDA still launches two transfers per matrix.  Pack every result
            # into one [2,N] tensor and slice it on the destination device.
            packed_assignments = torch.cat(
                [torch.stack((source, target)) for _, _, source, target in solved],
                dim=1,
            ).to(device=common_device, dtype=torch.int64)
            offset = 0
            for layer_index, batch_index, source, _ in solved:
                assignment_count = source.numel()
                results[layer_index][batch_index] = (
                    packed_assignments[0, offset : offset + assignment_count],
                    packed_assignments[1, offset : offset + assignment_count],
                )
                offset += assignment_count

        if any(item is None for layer in results for item in layer):
            raise RuntimeError("Hungarian matching did not produce all assignments")
        return [[item for item in layer if item is not None] for layer in results]

    @torch.no_grad()
    def forward(
        self,
        outputs: Mapping[str, Tensor],
        targets: Sequence[Mapping[str, Tensor]],
    ) -> list[tuple[Tensor, Tensor]]:
        return self.match_layers((outputs,), targets)[0]


class BHCDetrCriterion(nn.Module):
    """Detection losses plus decoder-layer BHCL from paper Eq. (2)."""

    def __init__(
        self,
        *,
        num_classes: int,
        projection_dim: int,
        decoder_layers: int,
        config: BHCDetrLossConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if num_classes != 25:
            raise ValueError("XH hierarchy requires exactly 25 fine classes")
        if decoder_layers <= 0:
            raise ValueError("decoder_layers must be positive")
        self.num_classes = num_classes
        self.config = (
            config
            if isinstance(config, BHCDetrLossConfig)
            else BHCDetrLossConfig.from_mapping(config)
        )
        self.matcher = HungarianMatcher(self.config)
        self.bhcl_layers = nn.ModuleList(
            BalancedHierarchicalContrastiveLoss(
                projection_dim=projection_dim,
                hierarchy=XH_HIERARCHY,
                temperature=self.config.temperature,
                epsilon=self.config.prototype_momentum,
            )
            for _ in range(decoder_layers)
        )

    @staticmethod
    def _matched_permutation(
        indices: Sequence[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        batch = torch.cat(
            [
                torch.full_like(source, batch_index)
                for batch_index, (source, _) in enumerate(indices)
            ]
        )
        source = torch.cat([source for source, _ in indices])
        return batch, source

    @staticmethod
    def _number_of_boxes(targets: Sequence[Mapping[str, Tensor]], device: torch.device) -> Tensor:
        count = torch.tensor(
            [sum(int(target["labels"].numel()) for target in targets)],
            dtype=torch.float32,
            device=device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(count)
            count /= torch.distributed.get_world_size()
        return count.clamp(min=1.0)

    def _detection_losses(
        self,
        output: Mapping[str, Tensor],
        targets: Sequence[Mapping[str, Tensor]],
        indices: Sequence[tuple[Tensor, Tensor]],
        number_of_boxes: Tensor,
    ) -> dict[str, Tensor]:
        logits = output["pred_logits"]
        target_classes = torch.zeros_like(logits)
        for batch_index, (source_indices, target_indices) in enumerate(indices):
            if source_indices.numel():
                labels = targets[batch_index]["labels"][target_indices]
                target_classes[batch_index, source_indices, labels] = 1.0
        classification = (
            sigmoid_focal_loss(
                logits,
                target_classes,
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
            ).sum()
            / number_of_boxes
        )

        if not any(source.numel() for source, _ in indices):
            zero = output["pred_boxes"].sum() * 0.0
            return {"loss_class": classification, "loss_bbox": zero, "loss_giou": zero}
        batch_indices, source_indices = self._matched_permutation(indices)
        predicted_boxes = output["pred_boxes"][batch_indices, source_indices]
        target_boxes = torch.cat(
            [
                target["boxes"][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ]
        )
        bbox = F.l1_loss(predicted_boxes, target_boxes, reduction="none").sum() / number_of_boxes
        giou_matrix = generalized_box_iou(
            box_cxcywh_to_xyxy(predicted_boxes),
            box_cxcywh_to_xyxy(target_boxes),
        )
        giou = (1.0 - torch.diag(giou_matrix)).sum() / number_of_boxes
        return {"loss_class": classification, "loss_bbox": bbox, "loss_giou": giou}

    @staticmethod
    def _matched_projected_queries(
        output: Mapping[str, Tensor],
        targets: Sequence[Mapping[str, Tensor]],
        indices: Sequence[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        if not any(source.numel() for source, _ in indices):
            projected = output["projected_queries"]
            return projected.new_empty((0, projected.shape[-1])), torch.empty(
                0,
                dtype=torch.long,
                device=projected.device,
            )
        batch_indices, source_indices = BHCDetrCriterion._matched_permutation(indices)
        projected = output["projected_queries"][batch_indices, source_indices]
        labels = torch.cat(
            [
                target["labels"][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ]
        )
        return projected, labels

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Tensor]],
    ) -> dict[str, Tensor]:
        layers = [*outputs.get("aux_outputs", []), outputs]
        if len(layers) != len(self.bhcl_layers):
            raise ValueError(
                f"criterion has {len(self.bhcl_layers)} layers but model returned {len(layers)}"
            )
        device = outputs["pred_logits"].device
        number_of_boxes = self._number_of_boxes(targets, device)
        result: dict[str, Tensor] = {}
        total = outputs["pred_logits"].sum() * 0.0
        # Matching is independent across decoder layers.  Queue every cost
        # matrix first so the exact SciPy assignments need only one blocking
        # accelerator-to-host transfer per criterion call.
        layer_indices = self.matcher.match_layers(layers, targets)
        for layer_index, (layer_output, bhcl, indices) in enumerate(
            zip(layers, self.bhcl_layers, layer_indices)
        ):
            detection = self._detection_losses(
                layer_output,
                targets,
                indices,
                number_of_boxes,
            )
            projected, labels = self._matched_projected_queries(layer_output, targets, indices)
            contrastive = bhcl(
                projected,
                labels,
                update_prototypes=self.training,
            )
            suffix = "" if layer_index + 1 == len(layers) else f"_aux{layer_index}"
            for name, value in detection.items():
                result[f"{name}{suffix}"] = value
            result[f"loss_bhcl{suffix}"] = contrastive
            detection_total = (
                self.config.class_loss_weight * detection["loss_class"]
                + self.config.bbox_loss_weight * detection["loss_bbox"]
                + self.config.giou_loss_weight * detection["loss_giou"]
            )
            if layer_index + 1 != len(layers):
                detection_total = self.config.auxiliary_loss_weight * detection_total
            total = total + detection_total
            total = total + self.config.bhcl_weight * contrastive / len(layers)
        gain_logits = outputs.get("gain_logits")
        if gain_logits is not None:
            if not isinstance(gain_logits, Tensor) or gain_logits.ndim != 4:
                raise ValueError("gain_logits must be [B,M+1,H,W]")
            gain_map = outputs.get("gain_map")
            if not isinstance(gain_map, Tensor):
                raise ValueError("gain_map output is required with gain_logits")
            patch_fraction = outputs.get("gain_patch_fraction")
            if not isinstance(patch_fraction, tuple) or len(patch_fraction) != 2:
                raise ValueError("gain_patch_fraction must be a (height,width) tuple")
            valid_mask = outputs.get("gain_valid_mask")
            if valid_mask is not None and not isinstance(valid_mask, Tensor):
                raise ValueError("gain_valid_mask must be a tensor")
            # Keep tiny-box intersection/area arithmetic in float32 even when
            # the detector forward runs under AMP.  Casting logits to float32
            # preserves the gradient path back into the Gain Head.
            stable_gain_logits = gain_logits.float()
            stable_gain_map = gain_map.float()
            gain_targets = build_gain_map_targets(
                targets,
                spatial_size=gain_logits.shape[-2:],
                patch_fraction=patch_fraction,
                bin_limit=gain_logits.shape[1] - 1,
                device=gain_logits.device,
                dtype=torch.float32,
            )
            map_loss = distribution_focal_gain_loss(
                stable_gain_logits,
                gain_targets,
                valid_mask=valid_mask,
            )
            lpm_loss = local_peak_margin_loss(
                stable_gain_map,
                gain_targets,
                margin=self.config.gain_lpm_margin,
                valid_mask=valid_mask,
            )
            result["loss_gain_map"] = map_loss
            result["loss_gain_lpm"] = lpm_loss
            total = total + self.config.gain_map_weight * map_loss
            total = total + self.config.gain_lpm_weight * lpm_loss
        result["loss_total"] = total
        return result


__all__ = [
    "BHCDetrCriterion",
    "BHCDetrLossConfig",
    "HungarianMatcher",
    "box_cxcywh_to_xyxy",
    "box_xyxy_to_cxcywh",
    "generalized_box_iou",
    "sigmoid_focal_loss",
]
