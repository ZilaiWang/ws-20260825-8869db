"""In-model proposal-quality distillation from a stronger detector teacher.

The deployable branch consumes ROIAlign features from the primary YOLO model.
It never changes proposal geometry or fine labels.  Its final layer is exactly
zero initialized, so attaching a fresh branch is provably identical to the
primary detector before distillation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LetterboxGeometry:
    scale: float
    pad_left: float
    pad_top: float
    resized_width: int
    resized_height: int
    target_size: int


def letterbox_geometry(width: int, height: int, target_size: int) -> LetterboxGeometry:
    """Return the centered Ultralytics-style square letterbox geometry."""
    if width <= 0 or height <= 0 or target_size <= 0:
        raise ValueError("image dimensions and target_size must be positive")
    scale = min(target_size / width, target_size / height)
    resized_width = min(target_size, int(round(width * scale)))
    resized_height = min(target_size, int(round(height * scale)))
    # Ultralytics splits the residual padding around the image.  The floating
    # half-padding is the coordinate transform used for boxes; integer image
    # placement uses the matching round(x - 0.1) convention in the trainer.
    pad_left = (target_size - resized_width) / 2.0
    pad_top = (target_size - resized_height) / 2.0
    return LetterboxGeometry(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        resized_width=resized_width,
        resized_height=resized_height,
        target_size=target_size,
    )


def boxes_to_letterbox(
    boxes_xyxy: torch.Tensor,
    geometry: LetterboxGeometry,
) -> torch.Tensor:
    """Transform original-image xyxy boxes to model-input coordinates."""
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape [K, 4]")
    output = boxes_xyxy.clone()
    output[:, (0, 2)] = output[:, (0, 2)] * geometry.scale + geometry.pad_left
    output[:, (1, 3)] = output[:, (1, 3)] * geometry.scale + geometry.pad_top
    output.clamp_(0.0, float(geometry.target_size))
    return output


def proposal_metadata(
    detector_scores: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    *,
    image_size: int,
) -> torch.Tensor:
    """Build bounded proposal metadata without any labels or GT information."""
    if detector_scores.ndim != 1:
        raise ValueError("detector_scores must have shape [K]")
    if boxes_xyxy.shape != (len(detector_scores), 4):
        raise ValueError("boxes and detector scores are not row aligned")
    eps = torch.finfo(detector_scores.dtype).eps
    score = detector_scores.clamp(eps, 1.0 - eps)
    score_logit = torch.logit(score)
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
    width = (x2 - x1).clamp_min(0.0) / float(image_size)
    height = (y2 - y1).clamp_min(0.0) / float(image_size)
    center_x = ((x1 + x2) * 0.5) / float(image_size)
    center_y = ((y1 + y2) * 0.5) / float(image_size)
    return torch.stack((score_logit, center_x, center_y, width, height), dim=1)


class AgreementResidualHead(nn.Module):
    """Small zero-impact residual head over detector ROI evidence."""

    metadata_dim = 5

    def __init__(self, roi_dim: int, *, hidden_dim: int = 128) -> None:
        super().__init__()
        if roi_dim <= 0 or hidden_dim <= 0:
            raise ValueError("roi_dim and hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(roi_dim + self.metadata_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, roi_features: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        if roi_features.ndim != 2 or metadata.ndim != 2:
            raise ValueError("roi_features and metadata must be matrices")
        if len(roi_features) != len(metadata) or metadata.shape[1] != self.metadata_dim:
            raise ValueError("ROI features and metadata are not aligned")
        return self.network(torch.cat((roi_features, metadata), dim=1)).squeeze(1)


def apply_logit_residual(
    detector_scores: torch.Tensor,
    residual_logits: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Apply a bounded residual in logit space while preserving score ordering limits."""
    if detector_scores.shape != residual_logits.shape:
        raise ValueError("detector_scores and residual_logits must have identical shape")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    eps = torch.finfo(detector_scores.dtype).eps
    baseline = torch.logit(detector_scores.clamp(eps, 1.0 - eps))
    return torch.sigmoid(baseline + alpha * torch.tanh(residual_logits))


def pairwise_support_ranking_loss(
    residual_logits: torch.Tensor,
    support_targets: torch.Tensor,
    *,
    positive_floor: float = 0.20,
    margin: float = 0.20,
) -> torch.Tensor:
    """Rank teacher-supported proposals above unsupported proposals per image."""
    if residual_logits.shape != support_targets.shape or residual_logits.ndim != 1:
        raise ValueError("ranking inputs must be aligned vectors")
    positive = residual_logits[support_targets >= positive_floor]
    negative = residual_logits[support_targets <= 0.0]
    if len(positive) == 0 or len(negative) == 0:
        return residual_logits.sum() * 0.0
    # All pairs are safe at the frozen per-image cap (default 32+32).
    return F.relu(margin - positive[:, None] + negative[None, :]).mean()


def normalized_feature_anchor(
    student: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Anchor trainable terminal FPN maps to the frozen primary detector."""
    if len(student) != len(reference) or not student:
        raise ValueError("student/reference feature sequences must align")
    losses = []
    for current, frozen in zip(student, reference, strict=True):
        if current.shape != frozen.shape:
            raise ValueError("student/reference feature shapes differ")
        scale = frozen.detach().float().square().mean().clamp_min(1e-6)
        losses.append((current.float() - frozen.detach().float()).square().mean() / scale)
    return torch.stack(losses).mean()


__all__ = [
    "AgreementResidualHead",
    "LetterboxGeometry",
    "apply_logit_residual",
    "boxes_to_letterbox",
    "letterbox_geometry",
    "normalized_feature_anchor",
    "pairwise_support_ranking_loss",
    "proposal_metadata",
]
