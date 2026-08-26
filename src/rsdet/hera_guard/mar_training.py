"""Feature construction and conservative training utilities for HERA MAR."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from rsdet.hera_guard.resolver import MonotoneAsymmetricResolver

MAR_FEATURE_NAMES = (
    "pav_foreground_logit",
    "pav_quality_logit",
    "pav_protect_logit",
    "pav_detector_fine_logit",
    "pav_detector_fine_margin",
    "pav_negative_fine_entropy",
)


@dataclass(frozen=True)
class MARFitResult:
    validation_scores: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    constrained_weights: np.ndarray
    rho: float
    bias: float
    final_loss: float


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def build_mar_features(
    *,
    rows: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    logits: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Build oriented evidence; larger always means safer current prediction."""

    candidate_ids = np.asarray(logits["candidate_id"], dtype=np.int64)
    if len(rows) != len(predictions) or len(candidate_ids) != len(rows):
        raise ValueError("MAR rows/predictions/logits length mismatch")
    if not np.array_equal(candidate_ids, np.arange(len(rows))):
        raise ValueError("MAR requires a sorted contiguous candidate ledger")
    fine_logits = np.asarray(logits["fine_logits"], dtype=np.float64)
    shifted = fine_logits - fine_logits.max(axis=1, keepdims=True)
    fine = np.exp(shifted)
    fine /= fine.sum(axis=1, keepdims=True)
    detector_category = np.asarray(
        [int(row["detector_category_id"]) for row in rows], dtype=np.int64
    )
    detector_probability = fine[np.arange(len(rows)), detector_category]
    other = fine.copy()
    other[np.arange(len(rows)), detector_category] = -1.0
    detector_margin = detector_probability - other.max(axis=1)
    entropy = -(fine * np.log(np.clip(fine, 1e-12, 1.0))).sum(axis=1) / math.log(fine.shape[1])
    features = np.column_stack(
        [
            np.asarray(logits["foreground_logit"], dtype=np.float64),
            np.asarray(logits["quality_logit"], dtype=np.float64),
            np.asarray(logits["protect_logit"], dtype=np.float64),
            _logit(detector_probability),
            detector_margin,
            -entropy,
        ]
    )
    if features.shape != (len(rows), len(MAR_FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError("MAR evidence is invalid")
    return features.astype(np.float32)


def fit_monotone_mar(
    *,
    train_base_score: np.ndarray,
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_role: Sequence[str],
    train_coarse: Sequence[int],
    validation_base_score: np.ndarray,
    validation_features: np.ndarray,
    epochs: int = 160,
    learning_rate: float = 0.03,
    rho_max: float = 1.0,
    seed: int = 202625,
) -> MARFitResult:
    """Fit one tiny monotone resolver without validation-based model selection."""

    import torch
    from torch.nn import functional as functional

    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("invalid MAR optimization settings")
    train_x = np.asarray(train_features, dtype=np.float32)
    validation_x = np.asarray(validation_features, dtype=np.float32)
    mean = train_x.mean(axis=0, dtype=np.float64)
    std = train_x.std(axis=0, dtype=np.float64)
    std[std < 1e-6] = 1.0
    train_x = ((train_x - mean) / std).astype(np.float32)
    validation_x = ((validation_x - mean) / std).astype(np.float32)

    torch.manual_seed(seed)
    model = MonotoneAsymmetricResolver(train_x.shape[1], rho_max=rho_max)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    base = torch.as_tensor(train_base_score, dtype=torch.float32)
    evidence = torch.from_numpy(train_x)
    target = torch.as_tensor(train_target, dtype=torch.float32)
    roles = np.asarray(list(train_role), dtype=object)
    coarse = np.asarray(list(train_coarse), dtype=np.int64)
    if len(coarse) != len(roles) or not set(np.unique(coarse)).issubset({0, 1, 2}):
        raise ValueError("MAR coarse labels are invalid")
    weights = np.full(len(roles), 0.10, dtype=np.float32)
    weights[target.numpy() > 0.5] = 1.0
    weights[roles == "protected_tp"] = 4.0
    weights[roles == "active_fp"] = 4.0
    sample_weight = torch.from_numpy(weights)
    protected = torch.from_numpy(roles == "protected_tp")
    active = torch.from_numpy(roles == "active_fp")
    coarse_tensor = torch.from_numpy(coarse)
    if not protected.any() or not active.any():
        raise ValueError("MAR training split lacks protected TP or active FP")
    base_logit = torch.logit(base.clamp(1e-6, 1.0 - 1e-6))

    final_loss = math.nan
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        resolved = model(base, evidence)
        resolved_logit = torch.logit(resolved.clamp(1e-6, 1.0 - 1e-6))
        pointwise = functional.binary_cross_entropy(resolved, target, weight=sample_weight)
        # Compare active FP only with protected TP from the same coarse class.
        ranking_terms = []
        for coarse_id in (0, 1, 2):
            positive_logit = resolved_logit[protected & (coarse_tensor == coarse_id)]
            negative_logit = resolved_logit[active & (coarse_tensor == coarse_id)]
            if not len(positive_logit) or not len(negative_logit):
                continue
            pair_count = max(len(positive_logit), len(negative_logit))
            positive_pair = positive_logit[torch.arange(pair_count) % len(positive_logit)]
            negative_pair = negative_logit[torch.arange(pair_count) % len(negative_logit)]
            ranking_terms.append(functional.softplus(-(positive_pair - negative_pair)).mean())
        if not ranking_terms:
            raise ValueError("MAR has no within-coarse ranking pairs")
        ranking = torch.stack(ranking_terms).mean()
        preserve = functional.relu(base_logit[protected] - resolved_logit[protected]).mean()
        residual = ((resolved_logit - base_logit) ** 2).mean()
        loss = pointwise + 0.50 * ranking + 0.50 * preserve + 0.01 * residual
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    with torch.inference_mode():
        validation_scores = model(
            torch.as_tensor(validation_base_score, dtype=torch.float32),
            torch.from_numpy(validation_x),
        ).numpy()
        constrained = model.constrained_parameters()
    return MARFitResult(
        validation_scores=validation_scores.astype(np.float64),
        train_mean=mean,
        train_std=std,
        constrained_weights=constrained["weights"].numpy(),
        rho=float(constrained["rho"].item()),
        bias=float(constrained["bias"].item()),
        final_loss=final_loss,
    )


__all__ = [
    "MAR_FEATURE_NAMES",
    "MARFitResult",
    "build_mar_features",
    "fit_monotone_mar",
]
