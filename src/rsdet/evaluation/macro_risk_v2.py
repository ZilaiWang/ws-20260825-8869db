"""MacroRisk V2: hierarchical fine thresholds with group-robust admission.

The optimizer is deliberately split into two auditable operations:

1. choose a raw operating point for each fine class, then shrink it in logit
   space toward its coarse anchor with an evidence-dependent movement cap;
2. evaluate the frozen thresholds by resampling *groups*, never individual
   images, and admit only on pessimistic recall/FDR quantiles.

This module contains no deployment code.  The same frozen threshold mapping is
consumed by :mod:`rsdet.postprocess.thresholds` in offline and Docker paths.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from rsdet.evaluation.hierarchical_thresholds import ThresholdCurves, select_threshold
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.protocol import EvaluationProtocol
from rsdet.postprocess.thresholds import normalize_fine_thresholds


@dataclass(frozen=True)
class RobustAdmission:
    iterations: int
    recall_p10: float
    recall_p50: float
    fdr_p50: float
    fdr_p90: float
    recall_pass_probability: float
    fdr_pass_probability: float
    joint_pass_probability: float
    admitted: bool


def _logit(value: float) -> float:
    clipped = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def _expit(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def logit_movement_cap(evidence: int, group_count: int) -> float:
    """Return the pre-registered maximum movement from the coarse anchor."""

    if evidence < 0 or group_count < 0:
        raise ValueError("evidence and group_count must be non-negative")
    if evidence < 30 or group_count < 3:
        return 0.20
    if evidence < 100 or group_count < 6:
        return 0.40
    return 0.80


def shrink_with_cap(
    raw_threshold: float,
    anchor_threshold: float,
    *,
    evidence: int,
    group_count: int,
    prior_strength: float,
    minimum_evidence: int,
) -> tuple[float, dict[str, float]]:
    """Shrink and clamp a fine threshold in logit space."""

    if prior_strength < 0.0 or minimum_evidence < 0:
        raise ValueError("prior controls must be non-negative")
    raw_logit = _logit(raw_threshold)
    anchor_logit = _logit(anchor_threshold)
    weight = (
        0.0
        if evidence < minimum_evidence
        else evidence / (evidence + prior_strength)
        if evidence + prior_strength > 0.0
        else 1.0
    )
    proposed = anchor_logit + weight * (raw_logit - anchor_logit)
    cap = logit_movement_cap(evidence, group_count)
    selected_logit = min(max(proposed, anchor_logit - cap), anchor_logit + cap)
    selected = _expit(selected_logit)
    return selected, {
        "fine_weight": weight,
        "logit_cap": cap,
        "raw_logit_delta": raw_logit - anchor_logit,
        "selected_logit_delta": selected_logit - anchor_logit,
    }


def fit_macro_risk_v2(
    curves: ThresholdCurves,
    *,
    protocol: EvaluationProtocol,
    group_counts_by_fine: Mapping[int, int],
    target_fdr_by_coarse: Mapping[str, float],
    prior_strength: float = 50.0,
    minimum_evidence: int = 10,
) -> dict[str, Any]:
    """Fit frozen 25-class thresholds from training-only curves.

    Raw fine points optimize recall under the class-specific FDR budget.  A
    coarse anchor is fitted with the same budget and every fine movement is
    shrunk and capped according to GT and independent-group evidence.
    """

    missing_coarse = set(protocol.class_names) - set(target_fdr_by_coarse)
    if missing_coarse:
        raise ValueError(f"missing coarse FDR targets: {sorted(missing_coarse)}")
    coarse_points = {
        name: select_threshold(curves.coarse_curves[name], target_fdr_by_coarse[name])
        for name in protocol.class_names
    }
    thresholds: dict[int, float] = {}
    audit: dict[int, dict[str, Any]] = {}
    for category_id in sorted(curves.fine_curves):
        coarse = protocol.category_mapping[category_id]
        raw_point = select_threshold(
            curves.fine_curves[category_id], target_fdr_by_coarse[coarse]
        )
        anchor = float(coarse_points[coarse]["threshold"])
        evidence = int(curves.fine_gt_counts[category_id])
        groups = int(group_counts_by_fine.get(category_id, 0))
        selected, shrink = shrink_with_cap(
            float(raw_point["threshold"]),
            anchor,
            evidence=evidence,
            group_count=groups,
            prior_strength=prior_strength,
            minimum_evidence=minimum_evidence,
        )
        thresholds[category_id] = selected
        audit[category_id] = {
            "coarse_class": coarse,
            "gt_evidence": evidence,
            "independent_group_count": groups,
            "raw_threshold": float(raw_point["threshold"]),
            "coarse_anchor": anchor,
            "selected_threshold": selected,
            "raw_train_recall": float(raw_point["overall_recall"]),
            "raw_train_fdr": float(raw_point["overall_fdr"]),
            **shrink,
        }
    normalize_fine_thresholds(thresholds, require_complete=True)
    return {
        "version": "macro_risk_v2",
        "metric_protocol": protocol.metric_protocol,
        "coarse_anchors": {
            name: float(point["threshold"]) for name, point in coarse_points.items()
        },
        "fine_thresholds": thresholds,
        "fine_audit": audit,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires non-empty values")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def group_bootstrap_admission(
    gt: Mapping[int, list[dict[str, Any]]],
    predictions: Mapping[int, list[dict[str, Any]]],
    *,
    group_by_image: Mapping[int, str],
    protocol: EvaluationProtocol,
    iterations: int = 1000,
    seed: int = 20260901,
    recall_p10_min: float | None = None,
    fdr_p90_max: float | None = None,
) -> RobustAdmission:
    """Evaluate frozen predictions with a group-level non-parametric bootstrap."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    image_ids = set(gt) | set(predictions)
    missing = image_ids - set(group_by_image)
    if missing:
        raise ValueError(f"group mapping missing image ids: {sorted(missing)[:10]}")
    images_by_group: dict[str, list[int]] = defaultdict(list)
    for image_id in sorted(image_ids):
        images_by_group[str(group_by_image[image_id])].append(image_id)
    groups = sorted(images_by_group)
    if len(groups) < 2:
        raise ValueError("group bootstrap requires at least two groups")

    rng = random.Random(seed)
    recalls: list[float] = []
    fdrs: list[float] = []
    for _ in range(iterations):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        boot_gt: dict[int, list[dict[str, Any]]] = {}
        boot_pred: dict[int, list[dict[str, Any]]] = {}
        synthetic_id = 0
        for group in sampled:
            for image_id in images_by_group[group]:
                boot_gt[synthetic_id] = list(gt.get(image_id, []))
                boot_pred[synthetic_id] = list(predictions.get(image_id, []))
                synthetic_id += 1
        ranking = evaluate_ranking_metrics(
            boot_gt,
            boot_pred,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
            require_complete_taxonomy=False,
        )
        coarse = [ranking.per_coarse[name] for name in protocol.class_names]
        recalls.append(fmean(item.macro_recall for item in coarse))
        fdrs.append(fmean(item.macro_fdr for item in coarse))

    recall_floor = protocol.recall_min if recall_p10_min is None else recall_p10_min
    fdr_ceiling = protocol.fdr_max if fdr_p90_max is None else fdr_p90_max
    recall_probability = sum(value >= recall_floor for value in recalls) / iterations
    fdr_probability = sum(value <= fdr_ceiling for value in fdrs) / iterations
    joint_probability = sum(
        recall >= recall_floor and fdr <= fdr_ceiling
        for recall, fdr in zip(recalls, fdrs, strict=True)
    ) / iterations
    recall_p10 = _quantile(recalls, 0.10)
    fdr_p90 = _quantile(fdrs, 0.90)
    return RobustAdmission(
        iterations=iterations,
        recall_p10=recall_p10,
        recall_p50=_quantile(recalls, 0.50),
        fdr_p50=_quantile(fdrs, 0.50),
        fdr_p90=fdr_p90,
        recall_pass_probability=recall_probability,
        fdr_pass_probability=fdr_probability,
        joint_pass_probability=joint_probability,
        admitted=recall_p10 >= recall_floor and fdr_p90 <= fdr_ceiling,
    )


__all__ = [
    "RobustAdmission",
    "fit_macro_risk_v2",
    "group_bootstrap_admission",
    "logit_movement_cap",
    "shrink_with_cap",
]
