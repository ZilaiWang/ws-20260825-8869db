"""Evidence gate for choosing exactly one Ship training intervention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ShipTrainingDecision:
    fp_background_share: float
    fp_classification_share: float
    fn_miss_share: float
    fn_classification_share: float
    selected_direction: str
    admitted: bool
    reason: str


def decide_ship_training_direction(
    fp_counts: Mapping[str, int],
    fn_counts: Mapping[str, int],
    *,
    minimum_cases: int = 100,
) -> ShipTrainingDecision:
    """Choose ``fine_tail`` or ``objectness_quality`` from reviewed errors.

    Duplicate and localization errors are excluded from the direction vote:
    neither proposed training intervention directly targets them.
    """

    fp_bg = int(fp_counts.get("FP_BG", 0))
    fp_cls = int(fp_counts.get("FP_CLS", 0))
    fn_miss = int(fn_counts.get("FN_MISS", 0))
    fn_cls = int(fn_counts.get("FN_CLS", 0))
    fp_total = fp_bg + fp_cls
    fn_total = fn_miss + fn_cls
    evidence = fp_total + fn_total
    bg_share = fp_bg / fp_total if fp_total else 0.0
    fp_cls_share = fp_cls / fp_total if fp_total else 0.0
    miss_share = fn_miss / fn_total if fn_total else 0.0
    fn_cls_share = fn_cls / fn_total if fn_total else 0.0
    if evidence < minimum_cases:
        direction = "none"
        admitted = False
        reason = f"insufficient reviewed evidence: {evidence} < {minimum_cases}"
    elif bg_share >= 0.50 or miss_share >= 0.50:
        direction = "objectness_quality"
        admitted = True
        reason = "background false positives and/or missed objects dominate"
    elif fp_cls_share >= 0.50 and fn_cls_share >= 0.50:
        direction = "fine_tail"
        admitted = True
        reason = "same-object fine-class confusions dominate both FP and FN"
    else:
        direction = "none"
        admitted = False
        reason = "mixed error composition; neither exclusive intervention is justified"
    return ShipTrainingDecision(
        fp_background_share=bg_share,
        fp_classification_share=fp_cls_share,
        fn_miss_share=miss_share,
        fn_classification_share=fn_cls_share,
        selected_direction=direction,
        admitted=admitted,
        reason=reason,
    )


__all__ = ["ShipTrainingDecision", "decide_ship_training_direction"]
