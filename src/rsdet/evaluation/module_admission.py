"""Independent-module admission and final-recipe composition contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rsdet.evaluation.platform_protocol import PLATFORM_OBSERVED_PROTOCOL


@dataclass(frozen=True)
class ModuleAdmission:
    name: str
    metric_protocol: str
    independently_evaluated: bool
    admitted: bool
    gate_recall_delta: float
    gate_fdr_delta: float
    score_delta: float
    max_coarse_recall_drop: float
    latency_delta_seconds: float
    background_fp100mp_delta: float | None = None
    sentinel_b_passed: bool | None = None


def validate_module_admission(
    module: ModuleAdmission,
    *,
    minimum_recall_delta: float = 0.005,
    maximum_fdr_delta: float = 0.0,
    maximum_coarse_recall_drop: float = 0.005,
    maximum_latency_delta_seconds: float = 2.0,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if module.metric_protocol != PLATFORM_OBSERVED_PROTOCOL:
        reasons.append("wrong_metric_protocol")
    if not module.independently_evaluated:
        reasons.append("not_independently_evaluated")
    if not module.admitted:
        reasons.append("module_self_rejected")
    if module.gate_recall_delta < minimum_recall_delta:
        reasons.append("recall_gain_below_gate")
    if module.gate_fdr_delta > maximum_fdr_delta:
        reasons.append("fdr_regression")
    if module.score_delta <= 0.0:
        reasons.append("score_not_improved")
    if module.max_coarse_recall_drop > maximum_coarse_recall_drop:
        reasons.append("coarse_recall_drop")
    if module.latency_delta_seconds > maximum_latency_delta_seconds:
        reasons.append("latency_regression")
    if (
        module.background_fp100mp_delta is not None
        and module.background_fp100mp_delta > 0.0
    ):
        reasons.append("background_regression")
    if module.sentinel_b_passed is False:
        reasons.append("sentinel_b_failed")
    return not reasons, reasons


def compose_final_recipe(
    baseline_recipe: Mapping[str, Any],
    modules: Sequence[ModuleAdmission],
) -> dict[str, Any]:
    """Compose names only; module implementation/config remains immutable."""

    accepted: list[str] = []
    rejected: dict[str, list[str]] = {}
    for module in modules:
        passed, reasons = validate_module_admission(module)
        if passed:
            accepted.append(module.name)
        else:
            rejected[module.name] = reasons
    if "ship_fine_tail" in accepted and "ship_objectness_quality" in accepted:
        raise ValueError("exclusive Ship interventions cannot both enter final recipe")
    payload: dict[str, Any] = {
        "version": "macroshift_final_recipe_v1",
        "metric_protocol": PLATFORM_OBSERVED_PROTOCOL,
        "baseline_recipe": dict(baseline_recipe),
        "accepted_modules": sorted(accepted),
        "rejected_modules": rejected,
        "unique_full_training_admission": bool(accepted),
        "no_univariate_module_is_combined_without_independent_admission": True,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["recipe_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


__all__ = [
    "ModuleAdmission",
    "compose_final_recipe",
    "validate_module_admission",
]
