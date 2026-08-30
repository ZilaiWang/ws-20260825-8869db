"""Official-match-aware proposal quality learning for HERA-Guard V4.

This module intentionally does not replace the incumbent detector score.  It
learns a bounded residual from deployable detector/FPN evidence and supervises
four distinct quantities:

* intrinsic matchability: same-fine-class IoU reaches the official threshold;
* protected TP: the proposal is a TP inside the frozen operating prefix;
* active FP: the proposal is an FP inside that same prefix;
* score residual: a bounded correction to the incumbent detector/OER logit.

The split is important: an intrinsically matchable duplicate is not background,
and a low-score TP outside the current prefix is not an active FP.  The labels
must be produced with the repository's prediction-first official matcher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

COARSE_SHIP = 0
COARSE_AIRCRAFT = 1
COARSE_VEHICLE = 2
OFFICIAL_IOU_THRESHOLDS = (0.50, 0.50, 0.35)


def safe_logit_tensor(probability: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Numerically stable logit for probabilities in ``[0, 1]``."""

    return torch.logit(probability.clamp(epsilon, 1.0 - epsilon))


def official_iou_thresholds(
    coarse_ids: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return the class-specific official IoU threshold for every proposal."""

    if coarse_ids.ndim != 1:
        raise ValueError("coarse_ids must be a one-dimensional tensor")
    if coarse_ids.numel() and (
        int(coarse_ids.min()) < COARSE_SHIP or int(coarse_ids.max()) > COARSE_VEHICLE
    ):
        raise ValueError("coarse_ids must use 0=ship, 1=aircraft, 2=vehicle")
    values = torch.as_tensor(
        OFFICIAL_IOU_THRESHOLDS,
        device=coarse_ids.device,
        dtype=dtype or torch.float32,
    )
    return values[coarse_ids.long()]


def soft_official_match_target(
    best_same_fine_iou: torch.Tensor,
    coarse_ids: torch.Tensor,
    *,
    temperature: float = 0.04,
) -> torch.Tensor:
    """Smooth target centered exactly on the competition IoU threshold.

    Vehicle proposals are centered on 0.35; ship and aircraft on 0.50.  The
    continuous target supplies useful gradients to near-miss proposals without
    pretending that all IoUs are equivalent.
    """

    if best_same_fine_iou.ndim != 1 or best_same_fine_iou.shape != coarse_ids.shape:
        raise ValueError("best_same_fine_iou and coarse_ids must be aligned vectors")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and > 0")
    thresholds = official_iou_thresholds(coarse_ids, dtype=best_same_fine_iou.dtype)
    return torch.sigmoid((best_same_fine_iou - thresholds) / float(temperature))


@dataclass(frozen=True)
class OfficialQualityOutputs:
    match_logit: torch.Tensor
    protect_logit: torch.Tensor
    active_fp_logit: torch.Tensor
    residual: torch.Tensor
    final_logit: torch.Tensor


class OfficialMatchQualityHead(nn.Module):
    """Small bounded residual head over detector/FPN proposal evidence."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 192,
        residual_limit: float = 1.75,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim < 16:
            raise ValueError("hidden_dim must be >= 16")
        if not math.isfinite(residual_limit) or residual_limit <= 0.0:
            raise ValueError("residual_limit must be finite and > 0")
        self.input_dim = int(input_dim)
        self.residual_limit = float(residual_limit)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.match_head = nn.Linear(hidden_dim, 1)
        self.protect_head = nn.Linear(hidden_dim, 1)
        self.active_fp_head = nn.Linear(hidden_dim, 1)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Start as an exact identity mapping over the incumbent score.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        detector_score: torch.Tensor,
    ) -> OfficialQualityOutputs:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape [N, {self.input_dim}]")
        if detector_score.ndim != 1 or detector_score.shape[0] != features.shape[0]:
            raise ValueError("detector_score must be an aligned vector")
        hidden = self.trunk(features)
        match_logit = self.match_head(hidden).squeeze(1)
        protect_logit = self.protect_head(hidden).squeeze(1)
        active_fp_logit = self.active_fp_head(hidden).squeeze(1)
        evidence = torch.stack(
            (
                torch.tanh(match_logit),
                torch.tanh(protect_logit),
                torch.tanh(active_fp_logit),
            ),
            dim=1,
        )
        residual_raw = self.residual_head(torch.cat((hidden, evidence), dim=1)).squeeze(1)
        residual = self.residual_limit * torch.tanh(residual_raw)
        final_logit = safe_logit_tensor(detector_score) + residual
        return OfficialQualityOutputs(
            match_logit=match_logit,
            protect_logit=protect_logit,
            active_fp_logit=active_fp_logit,
            residual=residual,
            final_logit=final_logit,
        )


@dataclass(frozen=True)
class QualityLossWeights:
    match: float = 1.0
    protect: float = 1.0
    active_fp: float = 1.0
    final_quality: float = 1.0
    rank: float = 0.35
    residual_l2: float = 0.02


def active_pair_rank_loss(
    final_logit: torch.Tensor,
    detector_score: torch.Tensor,
    protected_tp: torch.Tensor,
    active_fp: torch.Tensor,
    coarse_ids: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    score_band: float = 1.25,
    max_pairs: int = 4096,
    relax_group_if_empty: bool = False,
) -> torch.Tensor:
    """Rank high-value TPs above active FPs in the same coarse/domain group.

    This deliberately differs from an object-group-only duplicate loss.  The
    competition bottleneck is a low-score TP competing with a high-score
    background FP near the operating frontier.  Pairs are restricted to the
    same source/domain group and coarse category.  Relaxation is opt-in so a
    sparse batch cannot silently change the registered experiment contract.
    """

    vectors = (final_logit, detector_score, protected_tp, active_fp, coarse_ids, group_ids)
    if any(value.ndim != 1 for value in vectors):
        raise ValueError("all active-pair inputs must be vectors")
    if len({int(value.shape[0]) for value in vectors}) != 1:
        raise ValueError("active-pair inputs must have identical lengths")
    if max_pairs == 0:
        return final_logit.sum() * 0.0

    positive = torch.nonzero(protected_tp.bool(), as_tuple=False).flatten()
    negative = torch.nonzero(active_fp.bool(), as_tuple=False).flatten()
    if positive.numel() == 0 or negative.numel() == 0:
        return final_logit.sum() * 0.0

    pos_grid = positive[:, None].expand(-1, negative.numel())
    neg_grid = negative[None, :].expand(positive.numel(), -1)
    same_coarse = coarse_ids[pos_grid] == coarse_ids[neg_grid]
    same_group = group_ids[pos_grid] == group_ids[neg_grid]
    detector_logit = safe_logit_tensor(detector_score)
    near_frontier = (
        detector_logit[pos_grid] - detector_logit[neg_grid]
    ).abs() <= float(score_band)

    mask = same_coarse & same_group & near_frontier
    if not bool(mask.any()) and relax_group_if_empty:
        mask = same_coarse & near_frontier
    if not bool(mask.any()):
        return final_logit.sum() * 0.0

    pos_index = pos_grid[mask]
    neg_index = neg_grid[mask]
    # Hardest pairs first: FP currently outranks TP by the largest amount.
    difficulty = final_logit[neg_index] - final_logit[pos_index]
    order = torch.argsort(difficulty, descending=True)
    if max_pairs > 0:
        order = order[:max_pairs]
    return F.softplus(
        -(final_logit[pos_index[order]] - final_logit[neg_index[order]])
    ).mean()


def official_quality_loss(
    outputs: OfficialQualityOutputs,
    *,
    detector_score: torch.Tensor,
    soft_match_target: torch.Tensor,
    protected_tp: torch.Tensor,
    active_fp: torch.Tensor,
    active_mask: torch.Tensor,
    coarse_ids: torch.Tensor,
    group_ids: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    weights: QualityLossWeights = QualityLossWeights(),
    rank_enabled: bool = True,
    rank_relax_group_if_empty: bool = False,
    rank_max_pairs: int = 4096,
    final_positive_weight: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Return per-sample and scalar losses for nested/group-robust training."""

    n = outputs.final_logit.shape[0]
    aligned = (
        detector_score,
        soft_match_target,
        protected_tp,
        active_fp,
        active_mask,
        coarse_ids,
        group_ids,
    )
    if any(value.ndim != 1 or value.shape[0] != n for value in aligned):
        raise ValueError("quality labels must be aligned vectors")
    match_loss = F.binary_cross_entropy_with_logits(
        outputs.match_logit,
        soft_match_target.float(),
        reduction="none",
    )
    protect_loss = F.binary_cross_entropy_with_logits(
        outputs.protect_logit,
        protected_tp.float(),
        reduction="none",
    )
    active_fp_loss = F.binary_cross_entropy_with_logits(
        outputs.active_fp_logit,
        active_fp.float(),
        reduction="none",
    )
    active = active_mask.float()
    # Inside the frozen workpoint, official one-winner TP/FP status is the
    # deployment target.  Outside it, intrinsic same-fine matchability keeps
    # low-score recoverable TPs positive without calling active duplicates TP.
    final_target = torch.where(active_mask.bool(), protected_tp.float(), soft_match_target)
    final_quality_loss = F.binary_cross_entropy_with_logits(
        outputs.final_logit,
        final_target,
        pos_weight=torch.as_tensor(
            final_positive_weight,
            device=outputs.final_logit.device,
            dtype=outputs.final_logit.dtype,
        ),
        reduction="none",
    )
    per_sample = (
        weights.match * match_loss
        + weights.protect * protect_loss * active
        + weights.active_fp * active_fp_loss * active
        + weights.final_quality * final_quality_loss
        + weights.residual_l2 * outputs.residual.square()
    )
    if sample_weight is not None:
        if sample_weight.ndim != 1 or sample_weight.shape[0] != n:
            raise ValueError("sample_weight must be an aligned vector")
        per_sample = per_sample * sample_weight

    rank = outputs.final_logit.sum() * 0.0
    if rank_enabled:
        rank = active_pair_rank_loss(
            outputs.final_logit,
            detector_score,
            protected_tp,
            active_fp,
            coarse_ids,
            group_ids,
            max_pairs=rank_max_pairs,
            relax_group_if_empty=rank_relax_group_if_empty,
        )
    total = per_sample.mean() + weights.rank * rank
    return {
        "total": total,
        "per_sample": per_sample,
        "match": match_loss.mean(),
        "protect": (protect_loss * active).sum() / active.sum().clamp_min(1.0),
        "active_fp": (active_fp_loss * active).sum() / active.sum().clamp_min(1.0),
        "final_quality": final_quality_loss.mean(),
        "rank": rank,
        "residual_l2": outputs.residual.square().mean(),
    }


class NormalizedQualityRuntime(nn.Module):
    """TorchScript-friendly normalization and score runtime."""

    def __init__(
        self,
        head: OfficialMatchQualityHead,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(feature_mean, dtype=torch.float32)
        std = torch.as_tensor(feature_std, dtype=torch.float32)
        if mean.ndim != 1 or std.shape != mean.shape or int(mean.numel()) != head.input_dim:
            raise ValueError("normalization vectors must match the quality-head input")
        if torch.any(std <= 0):
            raise ValueError("feature_std must be strictly positive")
        self.head = head
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)

    def forward(self, features: torch.Tensor, detector_score: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        outputs = self.head(normalized, detector_score)
        return torch.stack(
            (
                torch.sigmoid(outputs.final_logit),
                torch.sigmoid(outputs.match_logit),
                torch.sigmoid(outputs.protect_logit),
                torch.sigmoid(outputs.active_fp_logit),
                outputs.residual,
            ),
            dim=1,
        )


__all__ = [
    "COARSE_AIRCRAFT",
    "COARSE_SHIP",
    "COARSE_VEHICLE",
    "OFFICIAL_IOU_THRESHOLDS",
    "NormalizedQualityRuntime",
    "OfficialMatchQualityHead",
    "OfficialQualityOutputs",
    "QualityLossWeights",
    "active_pair_rank_loss",
    "official_iou_thresholds",
    "official_quality_loss",
    "safe_logit_tensor",
    "soft_official_match_target",
]
