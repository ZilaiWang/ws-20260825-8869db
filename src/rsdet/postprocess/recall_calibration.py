"""Recall-preserving fine threshold fitting from Normal-only score curves.

This is not the historical MacroRisk V2 recipe. It reuses its support shrinkage
but anchors at the incumbent threshold, only permits lower thresholds, and
optimizes the actual coarse macro score rather than pooled class counts.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from rsdet.evaluation.absolute_score import fdr_points, recall_points
from rsdet.evaluation.macro_risk_v2 import shrink_with_cap
from rsdet.postprocess.thresholds import effective_threshold, normalize_fine_thresholds

TARGETS = (0, 1, 2, 3, 24)


def operating_metrics(point: dict) -> tuple[float, float]:
    return float(point["overall_recall"]), float(point["overall_fdr"])


def select_recall_thresholds(
    curves: Mapping[int, list[dict]], *, incumbent: float,
    support: Mapping[int, tuple[int, int]], fdr_budget: float = .15,
) -> dict:
    if not 0 < incumbent < 1 or not 0 <= fdr_budget <= 1:
        raise ValueError("invalid incumbent or FDR budget")
    if set(curves) != set(TARGETS) or set(support) != set(TARGETS):
        raise ValueError("all five weak fine classes are required")
    for label, points in curves.items():
        if not points or len({p["threshold"] for p in points}) != len(points):
            raise ValueError("empty or duplicate curve thresholds")
        for point in points:
            if not all(math.isfinite(float(point[k])) and 0 <= point[k] <= 1
                       for k in ("threshold", "overall_recall", "overall_fdr")):
                raise ValueError("invalid curve values")
        if sum(p["threshold"] == incumbent for p in points) != 1:
            raise ValueError("curve must include the exact incumbent")
        if min(support[label]) < 0:
            raise ValueError("negative support")
    current = {c: next(p for p in curves[c] if p["threshold"] == incumbent) for c in TARGETS}
    thresholds, audit = {c: incumbent for c in range(25)}, {}

    def coarse_metrics(labels, override=None):
        rows = [override[1] if override is not None and c == override[0] else current[c] for c in labels]
        r = sum(operating_metrics(p)[0] for p in rows) / len(rows)
        f = sum(operating_metrics(p)[1] for p in rows) / len(rows)
        return r, f, (recall_points(r)+fdr_points(f))/7

    ceilings = {"ship": max(fdr_budget, coarse_metrics(TARGETS[:4])[1]),
                "vehicle": max(fdr_budget, coarse_metrics((24,))[1])}
    for label in TARGETS:  # Frozen deterministic coordinate order, one pass only.
        labels = TARGETS[:4] if label < 4 else (24,)
        ceiling = ceilings["ship" if label < 4 else "vehicle"]
        before = coarse_metrics(labels)
        count, groups = support[label]
        eligible = [p for p in curves[label] if p["threshold"] <= incumbent
                    and coarse_metrics(labels, (label, p))[1] <= ceiling + 1e-12
                    and coarse_metrics(labels, (label, p))[0] >= before[0] - 1e-12]
        # The incumbent may be infeasible only after a caller supplied inconsistent
        # curves. Do not silently invent a fallback threshold.
        if not eligible:
            raise ValueError("no incumbent-preserving feasible point")
        raw = max(eligible, key=lambda p: (coarse_metrics(labels, (label, p))[2],
                  coarse_metrics(labels, (label, p))[0], p["threshold"]))
        shrunk, shrink = shrink_with_cap(raw["threshold"], incumbent, evidence=count,
            group_count=groups, prior_strength=50, minimum_evidence=10)
        rounded = min((p for p in curves[label] if shrunk <= p["threshold"] <= incumbent),
                      key=lambda p: p["threshold"])
        after = coarse_metrics(labels, (label, rounded))
        accepted = count >= 10 and groups >= 2 and after[2] > before[2] + 1e-12 and after[1] <= ceiling + 1e-12
        if accepted:
            thresholds[label] = rounded["threshold"]
            current[label] = rounded
        audit[str(label)] = {"gt_support": count, "groups": groups, "raw_threshold": raw["threshold"],
            "shrunk_threshold": shrunk, "rounded_threshold": rounded["threshold"],
            "selected_threshold": thresholds[label], "accepted_on_normal_fit": accepted,
            "before_coarse_r_f_quality": before, "proposed_coarse_r_f_quality": after,
            "coarse_fdr_ceiling": ceiling, **shrink}
    normalize_fine_thresholds(thresholds, require_complete=True)
    return {"thresholds": thresholds, "fine_audit": audit, "aircraft_unchanged": True,
            "incumbent_records_preserved": True, "no_hard_sentinel_fit": True}


def apply_recall_thresholds(predictions, image_folds, mappings):
    output = {}
    for image_id, fold in image_folds.items():
        mapping = normalize_fine_thresholds(mappings[fold], require_complete=True)
        output[image_id] = [r for r in predictions.get(image_id, []) if r["score"] >=
            effective_threshold(r["category_id"], global_threshold=1., fine_thresholds=mapping)]
    return output
