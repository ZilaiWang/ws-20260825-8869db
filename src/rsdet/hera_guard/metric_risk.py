"""Metric-aligned residual risk model used by HERA-Guard V3.

The model is deliberately small and score-anchored.  It may alter the logit
of the incumbent OER score by at most ``residual_limit``; consequently every
stage has to demonstrate that its extra supervision improves the official
frontier instead of winning by replacing the established detector ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rsdet.hera_guard.metric_aligned import (
    CANONICAL,
    CROSS_COARSE,
    CROSS_FINE,
    DUPLICATE,
    MetricAlignedRole,
)

FEATURE_COLUMNS = (
    "anchor_logit",
    "detector_logit",
    "model_y5",
    "model_m3",
    "coarse_ship",
    "coarse_aircraft",
    "coarse_vehicle",
    "log_short_edge",
    "log_area",
    "log_aspect",
    "foreground_logit",
    "coarse_foreground_logit",
    "crop_class_probability",
    "crop_conditional_class_probability",
    "crop_top1_probability",
    "crop_margin",
    "crop_entropy_normalized",
    "detector_crop_agree",
    "support_y5_rot_max_iou",
    "support_y5_800_max_iou",
    "support_m3_id_max_iou",
    "support_coph_max_iou",
    "source_support_count",
    "source_support_score_sum",
    "heterogeneous_support",
)


def safe_logit(value: float, epsilon: float = 1e-6) -> float:
    clipped = min(max(float(value), epsilon), 1.0 - epsilon)
    return math.log(clipped / (1.0 - clipped))


def candidate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Stable identity shared by evidence and incumbent-score ledgers."""

    return (
        int(row["image_id"]),
        int(row["category_id"]),
        tuple(round(float(value), 6) for value in row["bbox"]),
        int(row.get("source_fold", -1)),
        str(row.get("source_model", "")),
    )


def align_anchor_scores(
    evidence: Sequence[Mapping[str, Any]], anchor: Sequence[Mapping[str, Any]]
) -> list[float]:
    """Align an incumbent score ledger without assuming identical row order."""

    if len(evidence) != len(anchor):
        raise ValueError("evidence and anchor ledgers must have identical lengths")
    by_key: dict[tuple[Any, ...], list[float]] = {}
    for row in anchor:
        by_key.setdefault(candidate_key(row), []).append(float(row["score"]))
    output: list[float] = []
    for row in evidence:
        values = by_key.get(candidate_key(row))
        if not values:
            raise ValueError(f"anchor ledger lacks candidate {candidate_key(row)}")
        output.append(values.pop(0))
    if any(values for values in by_key.values()):
        raise ValueError("anchor ledger contains unmatched candidates")
    return output


def build_metric_features(
    records: Sequence[Mapping[str, Any]],
    *,
    anchor_scores: Sequence[float],
    category_mapping: Mapping[int, str],
) -> list[list[float]]:
    """Build deployable numeric evidence; no GT-derived feature is included."""

    if len(records) != len(anchor_scores):
        raise ValueError("records and anchor_scores are misaligned")
    rows: list[list[float]] = []
    for item, anchor_score in zip(records, anchor_scores, strict=True):
        x, y, width, height = (float(value) for value in item["bbox"])
        del x, y
        if width <= 0.0 or height <= 0.0:
            raise ValueError("proposal width/height must be positive")
        coarse = str(category_mapping[int(item["category_id"])])
        model = str(item.get("source_model", "")).upper()
        foreground = float(item.get("foreground_probability", 0.5))
        coarse_foreground = float(item.get("coarse_foreground_probability", 0.5))
        values = [
            safe_logit(anchor_score),
            safe_logit(float(item.get("detector_score", item["score"]))),
            float(model == "Y5"),
            float(model == "M3"),
            float(coarse == "ship"),
            float(coarse == "aircraft"),
            float(coarse == "vehicle"),
            math.log1p(min(width, height)),
            math.log1p(width * height),
            math.log(max(width / height, height / width)),
            safe_logit(foreground),
            safe_logit(coarse_foreground),
            float(item.get("crop_class_probability", 0.0)),
            float(item.get("crop_conditional_class_probability", 0.0)),
            float(item.get("crop_top1", 0.0)),
            float(item.get("crop_margin", 0.0)),
            float(item.get("crop_entropy", 0.0)) / math.log(25.0),
            float(item.get("detector_crop_agree", 0.0)),
            float(item.get("support_y5_rot_max_iou", 0.0)),
            float(item.get("support_y5_800_max_iou", 0.0)),
            float(item.get("support_m3_id_max_iou", 0.0)),
            float(item.get("support_coph_max_iou", 0.0)),
            float(item.get("source_support_count", 0.0)),
            float(item.get("source_support_score_sum", 0.0)),
            float(item.get("heterogeneous_support", 0.0)),
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("metric feature row contains NaN/Inf")
        rows.append(values)
    return rows


def deterministic_rank_pairs(
    roles: Sequence[MetricAlignedRole],
    *,
    include_roles: frozenset[str],
    max_pairs: int,
) -> list[tuple[int, int]]:
    """Return canonical-positive/negative pairs from the same object group."""

    positives: dict[str, list[int]] = {}
    negatives: dict[str, list[int]] = {}
    for index, row in enumerate(roles):
        if not row.object_group_id:
            continue
        if row.role == CANONICAL:
            positives.setdefault(row.object_group_id, []).append(index)
        elif row.role in include_roles:
            negatives.setdefault(row.object_group_id, []).append(index)
    pairs = [
        (positive, negative)
        for group in sorted(set(positives) & set(negatives))
        for positive in sorted(positives[group])
        for negative in sorted(negatives[group])
    ]
    if max_pairs <= 0 or len(pairs) <= max_pairs:
        return pairs
    # Evenly spaced deterministic subsampling retains the full score range.
    return [pairs[(index * len(pairs)) // max_pairs] for index in range(max_pairs)]


@dataclass(frozen=True)
class MetricRiskLossWeights:
    rank: float = 0.35
    soft_fdr: float = 2.0
    soft_recall: float = 0.10
    one_winner: float = 0.50


try:  # Preserve importability in CPU audit environments without PyTorch.
    from torch import nn as _nn
except ImportError:  # pragma: no cover

    class _ModuleBase:  # type: ignore[no-redef]
        pass
else:
    _ModuleBase = _nn.Module


class AnchoredResidualRisk(_ModuleBase):
    """A bounded residual over the first (incumbent-logit) feature."""

    def __init__(
        self,
        *,
        feature_mean: Any,
        feature_std: Any,
        hidden_dim: int = 64,
        residual_limit: float = 2.5,
        dropout: float = 0.10,
    ) -> None:
        import torch
        from torch import nn

        super().__init__()
        mean = torch.as_tensor(feature_mean, dtype=torch.float32)
        std = torch.as_tensor(feature_std, dtype=torch.float32)
        if mean.ndim != 1 or std.shape != mean.shape or torch.any(std <= 0):
            raise ValueError("feature mean/std must be aligned positive vectors")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)
        self.residual_limit = float(residual_limit)
        self.residual = nn.Sequential(
            nn.Linear(int(mean.numel()), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: Any) -> Any:
        import torch

        normalized = (features - self.feature_mean) / self.feature_std
        correction = self.residual_limit * torch.tanh(self.residual(normalized).squeeze(1))
        return features[:, 0] + correction


def metric_risk_loss(
    logits: Any,
    targets: Any,
    *,
    stage: str,
    rank_pairs: Any | None,
    one_winner_pairs: Any | None,
    soft_threshold: float,
    target_fdr: float = 0.15,
    temperature: float = 0.35,
    positive_weight: float = 8.0,
    weights: MetricRiskLossWeights = MetricRiskLossWeights(),
) -> dict[str, Any]:
    """BCE → RankNet → soft-FDR → one-winner cumulative objective."""

    import torch
    from torch.nn import functional

    stages = ("bce", "rank", "soft_fdr", "one_winner")
    if stage not in stages:
        raise ValueError(f"unknown metric risk stage: {stage}")
    if logits.ndim != 1 or logits.shape != targets.shape:
        raise ValueError("logits and targets must be aligned vectors")
    bce = functional.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        pos_weight=torch.as_tensor(positive_weight, device=logits.device),
    )
    zero = logits.sum() * 0.0
    rank = zero
    if stages.index(stage) >= 1 and rank_pairs is not None and rank_pairs.numel():
        rank = functional.softplus(
            -(logits[rank_pairs[:, 0]] - logits[rank_pairs[:, 1]])
        ).mean()
    soft_fdr = zero
    soft_recall = zero
    if stages.index(stage) >= 2:
        selection = torch.sigmoid((logits - float(soft_threshold)) / temperature)
        soft_tp = (selection * targets).sum()
        soft_fp = (selection * (1.0 - targets)).sum()
        soft_fdr_value = soft_fp / (soft_tp + soft_fp + 1e-6)
        soft_fdr = torch.relu(soft_fdr_value - target_fdr).square()
        soft_recall = soft_tp / (targets.sum() + 1e-6)
    one_winner = zero
    if (
        stages.index(stage) >= 3
        and one_winner_pairs is not None
        and one_winner_pairs.numel()
    ):
        one_winner = functional.softplus(
            -(logits[one_winner_pairs[:, 0]] - logits[one_winner_pairs[:, 1]])
        ).mean()
    total = (
        bce
        + weights.rank * rank
        + weights.soft_fdr * soft_fdr
        - weights.soft_recall * soft_recall
        + weights.one_winner * one_winner
    )
    return {
        "total": total,
        "bce": bce,
        "rank": rank,
        "soft_fdr": soft_fdr,
        "soft_recall": soft_recall,
        "one_winner": one_winner,
    }


def broad_rank_roles() -> frozenset[str]:
    return frozenset({DUPLICATE, CROSS_FINE, CROSS_COARSE})


def one_winner_roles() -> frozenset[str]:
    return frozenset({DUPLICATE, CROSS_FINE})


__all__ = [
    "AnchoredResidualRisk",
    "FEATURE_COLUMNS",
    "MetricRiskLossWeights",
    "align_anchor_scores",
    "broad_rank_roles",
    "build_metric_features",
    "candidate_key",
    "deterministic_rank_pairs",
    "metric_risk_loss",
    "one_winner_roles",
    "safe_logit",
]
