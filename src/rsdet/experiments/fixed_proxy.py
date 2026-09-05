"""Small, fixed-score trend decision. No threshold fitting or hidden gates."""

from __future__ import annotations

import math
from typing import Any

from rsdet.evaluation.absolute_score import fdr_points, recall_points


def quality_contribution(platform: dict[str, Any]) -> float:
    if platform.get("metric_protocol") != "platform_observed_20260831":
        raise ValueError("fixed proxy requires the platform-observed metric protocol")
    return sum(
        recall_points(platform["per_coarse"][name]["macro_recall"])
        + fdr_points(platform["per_coarse"][name]["macro_fdr"])
        for name in ("ship", "aircraft", "vehicle")
    ) / 7.0


def review_quality_delta(
    baseline: dict[str, Any], candidate: dict[str, Any], *, stage: str, minimum: float
) -> dict[str, Any]:
    if stage not in {"hard", "sentinel"} or not math.isfinite(minimum):
        raise ValueError("invalid fixed proxy review stage/minimum")
    bq, cq = quality_contribution(baseline), quality_contribution(candidate)
    delta = cq - bq
    worthwhile = delta > minimum
    return {
        "stage": stage,
        "baseline_quality_contribution": bq,
        "candidate_quality_contribution": cq,
        "delta_quality": delta,
        "minimum_exclusive": minimum,
        "direction_pass": worthwhile,
        "next_action": (
            "evaluate_frozen_sentinel" if worthwhile and stage == "hard" else
            "review_deployment_cost_and_risk" if worthwhile else "stop_fixed_recipe"
        ),
        "formal_admission": False,
        "quality_excludes_latency": True,
        "not_an_official_score_prediction": True,
    }
