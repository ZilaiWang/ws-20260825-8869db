"""Absolute preliminary-round score published on 2026-08-31.

This scorer is deliberately separate from :mod:`official_ranking`, which
implements the superseded V1.6 relative-ranking protocol.  The organiser's
new formula maps Recall, false-detection rate (FDR), and latency to an
absolute score.  The public notice did not state how the three coarse-class
rows were combined, so the historical interpretations remain available.
The 2026-08-31 formal platform response subsequently resolved the ambiguity;
use :func:`platform_confirmed_score` for current submission decisions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

COARSE_CLASSES: tuple[str, ...] = ("ship", "aircraft", "vehicle")


def _rate(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def recall_points(recall: float) -> float:
    """Return the published Recall sub-score in ``[0, 100]``."""
    recall = _rate("recall", recall)
    if recall <= 0.85:
        return recall / 0.85 * 60.0
    return 60.0 + (recall - 0.85) / 0.15 * 40.0


def fdr_points(fdr: float) -> float:
    """Return the published false-detection-rate sub-score."""
    fdr = _rate("fdr", fdr)
    if fdr <= 0.2:
        return 100.0 - fdr / 0.2 * 40.0
    return 60.0 - (fdr - 0.2) / 0.8 * 60.0


def latency_points(latency_seconds: float) -> float:
    """Return the published inference-time sub-score.

    The notice is discontinuous at 20 seconds: ``t == 20`` scores 80 while
    ``t > 20`` scores 0.  We preserve that literal contract and make it
    visible in tests rather than smoothing it by assumption.
    """
    latency_seconds = float(latency_seconds)
    if not math.isfinite(latency_seconds) or latency_seconds < 0.0:
        raise ValueError("latency_seconds must be finite and non-negative")
    return 100.0 - latency_seconds if latency_seconds <= 20.0 else 0.0


def competition_score(recall: float, fdr: float, latency_seconds: float) -> dict[str, float]:
    """Compute all three sub-scores and the published weighted total."""
    sr = recall_points(recall)
    sf = fdr_points(fdr)
    st = latency_points(latency_seconds)
    total = 3.0 / 7.0 * sr + 3.0 / 7.0 * sf + 1.0 / 7.0 * st
    return {
        "recall": float(recall),
        "fdr": float(fdr),
        "latency_seconds": float(latency_seconds),
        "recall_points": sr,
        "fdr_points": sf,
        "latency_points": st,
        "total_score": total,
    }


def platform_confirmed_score(
    per_coarse: Mapping[str, Mapping[str, Any]],
    latency_seconds: float,
    *,
    recall_gate: float = 0.85,
    fdr_gate: float = 0.20,
    latency_gate_seconds: float = 20.0,
) -> dict[str, Any]:
    """Reproduce the aggregation observed on the formal evaluation platform.

    The 2026-08-31 formal response resolved an ambiguity left by the public
    formula.  The platform scores the six coarse-class Recall/FDR values plus
    latency as seven equally weighted sub-scores.  Its hard gates use the
    arithmetic mean of the three displayed coarse-class rates.  Raw pooled
    TP/FP/FN are returned by the API, but are diagnostic and do not drive the
    displayed score or the gate flags.

    Gate comparisons are inclusive here so exact-boundary behaviour remains
    conservative and explicit.  Deployment admission should use a margin
    rather than relying on equality at a published boundary.
    """
    rows = _coarse_rows(per_coarse)
    latency_seconds = float(latency_seconds)
    recall_gate = _rate("recall_gate", recall_gate)
    fdr_gate = _rate("fdr_gate", fdr_gate)
    if not math.isfinite(latency_gate_seconds) or latency_gate_seconds < 0.0:
        raise ValueError("latency_gate_seconds must be finite and non-negative")

    per_coarse_points: dict[str, dict[str, float]] = {}
    seven_subscores: list[float] = []
    for coarse in COARSE_CLASSES:
        row = rows[coarse]
        recall_score = recall_points(row["recall"])
        fdr_score = fdr_points(row["fdr"])
        per_coarse_points[coarse] = {
            "recall": row["recall"],
            "fdr": row["fdr"],
            "recall_points": recall_score,
            "fdr_points": fdr_score,
        }
        seven_subscores.extend((recall_score, fdr_score))
    time_score = latency_points(latency_seconds)
    seven_subscores.append(time_score)

    gate_recall = sum(row["recall"] for row in rows.values()) / 3.0
    gate_fdr = sum(row["fdr"] for row in rows.values()) / 3.0
    output: dict[str, Any] = {
        "protocol": "platform_confirmed_absolute_score_v2026_08_31",
        "per_coarse": per_coarse_points,
        "latency_seconds": latency_seconds,
        "latency_points": time_score,
        "seven_subscores": seven_subscores,
        "total_score": sum(seven_subscores) / 7.0,
        "hard_gates": {
            "macro_coarse_recall": gate_recall,
            "macro_coarse_fdr": gate_fdr,
            "latency_seconds": latency_seconds,
            "recall_threshold": recall_gate,
            "fdr_threshold": fdr_gate,
            "latency_threshold_seconds": latency_gate_seconds,
            "recall_pass": gate_recall >= recall_gate,
            "fdr_pass": gate_fdr <= fdr_gate,
            "latency_pass": latency_seconds <= latency_gate_seconds,
        },
        "pooled_counts_are_diagnostic_only": True,
    }
    if all(all(count in row for count in ("tp", "fp", "fn")) for row in rows.values()):
        tp = int(sum(row["tp"] for row in rows.values()))
        fp = int(sum(row["fp"] for row in rows.values()))
        fn = int(sum(row["fn"] for row in rows.values()))
        output["pooled_counts"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "fdr": fp / (tp + fp) if tp + fp else 0.0,
        }
    return output


def _coarse_rows(per_coarse: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    missing = set(COARSE_CLASSES) - set(per_coarse)
    if missing:
        raise ValueError(f"per_coarse is missing: {sorted(missing)}")
    rows: dict[str, dict[str, float]] = {}
    for coarse in COARSE_CLASSES:
        raw = per_coarse[coarse]
        rows[coarse] = {
            "recall": _rate(f"{coarse}.recall", float(raw["recall"])),
            "fdr": _rate(f"{coarse}.fdr", float(raw["fdr"])),
        }
        for count in ("tp", "fp", "fn"):
            if count in raw:
                value = int(raw[count])
                if value < 0 or float(raw[count]) != value:
                    raise ValueError(f"{coarse}.{count} must be a non-negative integer")
                rows[coarse][count] = float(value)
    return rows


def score_coarse_interpretations(
    per_coarse: Mapping[str, Mapping[str, Any]], latency_seconds: float
) -> dict[str, Any]:
    """Score every aggregation interpretation supported by available data.

    ``macro_raw_then_score`` averages the three raw Recall/FDR values before
    applying the formula. ``mean_per_coarse_score`` scores each coarse class
    first and then averages the three totals. ``pooled_counts`` is emitted
    only when every row contains TP/FP/FN.
    """
    rows = _coarse_rows(per_coarse)
    macro_recall = sum(row["recall"] for row in rows.values()) / 3.0
    macro_fdr = sum(row["fdr"] for row in rows.values()) / 3.0
    macro = competition_score(macro_recall, macro_fdr, latency_seconds)
    individual = {
        coarse: competition_score(row["recall"], row["fdr"], latency_seconds)
        for coarse, row in rows.items()
    }
    mean_per_coarse = sum(item["total_score"] for item in individual.values()) / 3.0
    output: dict[str, Any] = {
        "status": "complete",
        "formula_version": "absolute_preliminary_2026-08-31",
        "aggregation_is_not_stated_in_public_formula": True,
        "macro_raw_then_score": macro,
        "mean_per_coarse_score": {
            "total_score": mean_per_coarse,
            "per_coarse": individual,
        },
        "platform_confirmed": platform_confirmed_score(rows, latency_seconds),
    }
    if all(all(count in row for count in ("tp", "fp", "fn")) for row in rows.values()):
        tp = sum(row["tp"] for row in rows.values())
        fp = sum(row["fp"] for row in rows.values())
        fn = sum(row["fn"] for row in rows.values())
        recall = tp / (tp + fn) if tp + fn else 0.0
        fdr = fp / (tp + fp) if tp + fp else 0.0
        output["pooled_counts"] = {
            **competition_score(recall, fdr, latency_seconds),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
    return output
