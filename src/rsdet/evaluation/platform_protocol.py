"""Observed formal-platform aggregation contract frozen on 2026-08-31."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any

from rsdet.evaluation.absolute_score import platform_confirmed_score
from rsdet.evaluation.official_metric import RankingMetrics

PLATFORM_OBSERVED_PROTOCOL = "platform_observed_20260831"
LEGACY_POOLED_PROTOCOL = "legacy_pooled"
DOCUMENTED_V1_6_PROTOCOL = "v1_6_documented"
SUPPORTED_METRIC_PROTOCOLS = frozenset(
    {PLATFORM_OBSERVED_PROTOCOL, LEGACY_POOLED_PROTOCOL, DOCUMENTED_V1_6_PROTOCOL}
)
COARSE_ORDER = ("ship", "aircraft", "vehicle")


@dataclass(frozen=True)
class PlatformObservedMetrics:
    """Three-coarse macro gate and seven-subscore platform result."""

    coarse_recall: dict[str, float]
    coarse_fdr: dict[str, float]
    gate_recall: float
    gate_fdr: float
    recall_pass: bool
    fdr_pass: bool
    latency_pass: bool | None
    absolute_score: float | None
    score_payload: dict[str, Any] | None


def build_platform_observed_metrics(
    ranking: RankingMetrics,
    *,
    recall_min: float = 0.85,
    fdr_max: float = 0.20,
    latency_seconds: float | None = None,
    latency_max_seconds: float | None = 20.0,
) -> PlatformObservedMetrics:
    """Build the only active formal gate from a complete ranking result."""

    missing = set(COARSE_ORDER) - set(ranking.per_coarse)
    if missing:
        raise ValueError(f"platform protocol missing coarse classes: {sorted(missing)}")
    recalls = {
        name: float(ranking.per_coarse[name].macro_recall) for name in COARSE_ORDER
    }
    fdrs = {name: float(ranking.per_coarse[name].macro_fdr) for name in COARSE_ORDER}
    gate_recall = fmean(recalls.values())
    gate_fdr = fmean(fdrs.values())
    latency_pass = (
        None
        if latency_seconds is None or latency_max_seconds is None
        else float(latency_seconds) <= float(latency_max_seconds)
    )
    score_payload = None
    absolute_score = None
    if latency_seconds is not None:
        score_payload = platform_confirmed_score(
            {
                name: {"recall": recalls[name], "fdr": fdrs[name]}
                for name in COARSE_ORDER
            },
            float(latency_seconds),
            recall_gate=float(recall_min),
            fdr_gate=float(fdr_max),
            latency_gate_seconds=(
                20.0 if latency_max_seconds is None else float(latency_max_seconds)
            ),
        )
        absolute_score = float(score_payload["total_score"])
    return PlatformObservedMetrics(
        coarse_recall=recalls,
        coarse_fdr=fdrs,
        gate_recall=gate_recall,
        gate_fdr=gate_fdr,
        recall_pass=gate_recall >= float(recall_min),
        fdr_pass=gate_fdr <= float(fdr_max),
        latency_pass=latency_pass,
        absolute_score=absolute_score,
        score_payload=score_payload,
    )


def platform_metrics_payload(metrics: PlatformObservedMetrics) -> dict[str, Any]:
    """Return a stable JSON-ready payload shared by reports and admissions."""

    return {
        "metric_protocol": PLATFORM_OBSERVED_PROTOCOL,
        "per_coarse": {
            name: {
                "macro_recall": metrics.coarse_recall[name],
                "macro_fdr": metrics.coarse_fdr[name],
            }
            for name in COARSE_ORDER
        },
        "gate_recall": metrics.gate_recall,
        "gate_fdr": metrics.gate_fdr,
        "recall_pass": metrics.recall_pass,
        "fdr_pass": metrics.fdr_pass,
        "latency_pass": metrics.latency_pass,
        "absolute_score": metrics.absolute_score,
        "score_payload": metrics.score_payload,
        "ranking_overall_is_not_platform_gate": True,
        "pooled_counts_are_diagnostic_only": True,
    }


__all__ = [
    "COARSE_ORDER",
    "DOCUMENTED_V1_6_PROTOCOL",
    "LEGACY_POOLED_PROTOCOL",
    "PLATFORM_OBSERVED_PROTOCOL",
    "PlatformObservedMetrics",
    "SUPPORTED_METRIC_PROTOCOLS",
    "build_platform_observed_metrics",
    "platform_metrics_payload",
]
