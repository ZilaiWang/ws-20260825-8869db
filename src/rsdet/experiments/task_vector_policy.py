"""Pure utilities for conservative task-vector admission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rsdet.evaluation.absolute_score import platform_confirmed_score


def select_conservative_alpha(scores: Mapping[float, float]) -> float:
    """Select the highest score and break ties toward the smallest alpha."""

    if not scores:
        raise ValueError("scores must not be empty")
    return min(scores, key=lambda alpha: (-float(scores[alpha]), float(alpha)))


def score_from_fine_counts(
    counts: Mapping[int, Mapping[str, int]],
    category_mapping: Mapping[int, str],
    *,
    latency_seconds: float = 0.0,
) -> dict[str, Any]:
    """Compute the platform-confirmed score from a complete fine-count ledger."""

    per_coarse: dict[str, list[tuple[float, float]]] = {
        "ship": [],
        "aircraft": [],
        "vehicle": [],
    }
    for fine_id, coarse in sorted(category_mapping.items()):
        row = counts.get(int(fine_id), {})
        tp = int(row.get("tp", 0))
        fp = int(row.get("fp", 0))
        fn = int(row.get("fn", 0))
        if min(tp, fp, fn) < 0:
            raise ValueError("fine counts must be non-negative")
        recall = tp / (tp + fn) if tp + fn else 1.0
        fdr = fp / (tp + fp) if tp + fp else 0.0
        per_coarse[str(coarse)].append((recall, fdr))
    rows: dict[str, dict[str, float]] = {}
    for coarse, values in per_coarse.items():
        if not values:
            raise ValueError(f"taxonomy has no fine classes for {coarse}")
        rows[coarse] = {
            "recall": sum(value[0] for value in values) / len(values),
            "fdr": sum(value[1] for value in values) / len(values),
        }
    return platform_confirmed_score(rows, latency_seconds)


def stress_incremental_vehicle_fp(
    baseline: Mapping[int, Mapping[str, int]],
    candidate: Mapping[int, Mapping[str, int]],
    *,
    vehicle_class_id: int = 24,
    multiplier: float = 6.0,
) -> dict[int, dict[str, int]]:
    """Multiply only positive incremental Vehicle FP; retain any reduction once."""

    if multiplier < 1.0:
        raise ValueError("multiplier must be >= 1")
    output = {
        int(fine_id): {name: int(row.get(name, 0)) for name in ("tp", "fp", "fn")}
        for fine_id, row in candidate.items()
    }
    base_fp = int(baseline.get(vehicle_class_id, {}).get("fp", 0))
    candidate_fp = int(candidate.get(vehicle_class_id, {}).get("fp", 0))
    incremental = candidate_fp - base_fp
    stressed = (
        base_fp + int(round(multiplier * incremental))
        if incremental > 0
        else candidate_fp
    )
    output.setdefault(vehicle_class_id, {"tp": 0, "fp": 0, "fn": 0})["fp"] = stressed
    return output


def percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolated percentile without a NumPy dependency."""

    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


__all__ = [
    "percentile",
    "score_from_fine_counts",
    "select_conservative_alpha",
    "stress_incremental_vehicle_fp",
]
