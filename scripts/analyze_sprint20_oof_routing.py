#!/usr/bin/env python3
"""Cross-fit per-coarse thresholds for P40 OTO/OTM head ownership.

Thresholds for each held-out outer fold are selected only from the other two
OOF folds using the current fine-macro platform metric.  The historical P40
folds are source-disjoint but shorter than the mature full model, so this is
directional evidence rather than a hidden-score estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import build_threshold_curve
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config
from sprint20.evaluation import (
    cache_predictions,
    evaluate,
    group_counts,
    gt_from_coco,
    paired_bootstrap,
    route_dicts,
)

COARSE = ("ship", "aircraft", "vehicle")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _subset(values: dict[int, list[Any]], ids: set[int]) -> dict[int, list[Any]]:
    return {image_id: list(values[image_id]) for image_id in sorted(ids)}


def _select_coarse(curve: list[dict[str, Any]], coarse: str, target_fdr: float) -> dict[str, Any]:
    fdr_key = f"{coarse}_macro_fdr"
    recall_key = f"{coarse}_macro_recall"
    feasible = [row for row in curve if float(row[fdr_key]) <= target_fdr]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                float(row[recall_key]),
                -float(row[fdr_key]),
                float(row["threshold"]),
            ),
        )
        rule = "max_macro_recall_under_macro_fdr"
    else:
        selected = min(
            curve,
            key=lambda row: (
                float(row[fdr_key]),
                -float(row[recall_key]),
                -float(row["threshold"]),
            ),
        )
        rule = "fallback_min_macro_fdr"
    return {
        "threshold": float(selected["threshold"]),
        "selection_macro_recall": float(selected[recall_key]),
        "selection_macro_fdr": float(selected[fdr_key]),
        "rule": rule,
    }


def _filter_by_coarse(
    pred: dict[int, list[dict[str, Any]]],
    ids: set[int],
    thresholds: dict[str, float],
    mapping: dict[int, str],
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            row
            for row in pred[image_id]
            if float(row["score"]) >= float(thresholds[mapping[int(row["category_id"])]])
        ]
        for image_id in sorted(ids)
    }


def _filter_fixed_threshold(
    pred: dict[int, list[dict[str, Any]]], ids: set[int], threshold: float
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [row for row in pred[image_id] if float(row["score"]) >= threshold]
        for image_id in sorted(ids)
    }


def _route_fixed_primary_ship23(
    oto: dict[int, list[dict[str, Any]]],
    otm: dict[int, list[dict[str, Any]]],
    ids: set[int],
    *,
    primary_threshold: float,
    alternative_threshold: float,
) -> dict[int, list[dict[str, Any]]]:
    """Keep the deployed OTO policy fixed; vary only OTM ownership for QHS/MS."""
    return {
        image_id: [
            row
            for row in oto[image_id]
            if int(row["category_id"]) not in (2, 3) and float(row["score"]) >= primary_threshold
        ]
        + [
            row
            for row in otm[image_id]
            if int(row["category_id"]) in (2, 3) and float(row["score"]) >= alternative_threshold
        ]
        for image_id in sorted(ids)
    }


def _alternative_threshold_curve(
    gt: dict[int, list[dict[str, Any]]],
    oto: dict[int, list[dict[str, Any]]],
    otm: dict[int, list[dict[str, Any]]],
    ids: set[int],
    thresholds: list[float],
    protocol: Any,
    primary_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use the official prefix scorer while only thresholding OTM QHS/MS.

    Retained OTO rows use score 1.0 in this *selection-only* stream.  OTM owns
    different fine classes (2/3), so this does not change within-class matching;
    it prevents the prefix threshold from silently retuning the incumbent OTO
    policy while it sweeps the alternative head.
    """
    stream = {}
    for image_id in sorted(ids):
        fixed = [
            {**row, "score": 1.0}
            for row in oto[image_id]
            if int(row["category_id"]) not in (2, 3) and float(row["score"]) >= primary_threshold
        ]
        alternative = [row for row in otm[image_id] if int(row["category_id"]) in (2, 3)]
        stream[image_id] = fixed + alternative
    return build_threshold_curve(
        _subset(gt, ids),
        stream,
        thresholds=thresholds,
        protocol=protocol,
    )


def _score_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    return float(candidate["score"]["total_score"]) - float(baseline["score"]["total_score"])


def _macro_fdr(metrics: dict[str, Any], labels: tuple[int, ...]) -> float:
    def fdr(row: dict[str, Any]) -> float:
        tp, fp = int(row["tp"]), int(row["fp"])
        return fp / (tp + fp) if tp + fp else 0.0

    return sum(fdr(metrics["per_fine"][str(label)]) for label in labels) / len(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--oto-cache", type=Path, required=True)
    parser.add_argument("--otm-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--fdr-levels", type=float, nargs="+", default=(0.10, 0.12, 0.15, 0.20))
    parser.add_argument("--diagnostic-latency", type=float, default=3.551833)
    parser.add_argument("--primary-threshold", type=float, default=0.536)
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not 0.0 < args.step <= 1.0:
        raise ValueError("step must be in (0,1]")
    coco, oto_cache, otm_cache = _read(args.coco), _read(args.oto_cache), _read(args.otm_cache)
    if oto_cache.get("role") != "outer_oof_short" or otm_cache.get("role") != "outer_oof_short":
        raise ValueError("This analysis requires the paired outer_oof_short caches")
    if oto_cache.get("head") != "oto" or otm_cache.get("head") != "otm":
        raise ValueError("Expected native OTO and native OTM caches")
    if oto_cache["coco_sha256"] != _sha256(args.coco) or otm_cache["coco_sha256"] != _sha256(
        args.coco
    ):
        raise ValueError("COCO provenance mismatch")
    for key in ("base_commit", "weight_sha256", "pipeline", "inference_model"):
        if oto_cache.get(key) != otm_cache.get(key):
            raise ValueError(f"Head-only comparison changed {key}")

    gt = gt_from_coco(coco)
    image_fold = {int(row["id"]): int(row["fold"]) for row in coco["images"]}
    folds = {
        fold: {image_id for image_id, value in image_fold.items() if value == fold}
        for fold in (0, 1, 2)
    }
    if set().union(*folds.values()) != set(gt) or any(not ids for ids in folds.values()):
        raise ValueError("COCO fold coverage is incomplete")
    floor = float(oto_cache["pipeline"]["score_threshold"])
    if not floor <= args.primary_threshold <= 1.0:
        raise ValueError("primary-threshold must be at or above the candidate floor")
    oto = cache_predictions(oto_cache, floor)
    otm = cache_predictions(otm_cache, floor)
    methods = {
        "native_oto": oto,
        "native_otm": otm,
        "oto_aircraft_otm_ship_fsc": route_dicts(oto, otm, [0, 1, 2, 3, 24]),
        "oto_aircraft_fsc_otm_ship": route_dicts(oto, otm, [0, 1, 2, 3]),
        "oto_ship01_aircraft_fsc_otm_ship23": route_dicts(oto, otm, [2, 3]),
        "oto_ship_aircraft_otm_fsc": route_dicts(oto, otm, [24]),
    }
    image_to_group = None
    if args.groups is not None:
        group_payload = _read(args.groups)
        image_to_group = {}
        for sample in group_payload["samples"]:
            image_id = int(sample["image_id"])
            group = str(sample.get(args.group_key, "")).strip()
            if not group:
                raise ValueError(f"Missing {args.group_key} for image {image_id}")
            if image_id in image_to_group and image_to_group[image_id] != group:
                raise ValueError(f"Conflicting groups for image {image_id}")
            image_to_group[image_id] = group
        if set(gt) - set(image_to_group):
            raise ValueError("Group manifest does not cover every OOF image")
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    thresholds = build_threshold_grid(floor, 1.0, args.step)
    all_ids = set(gt)
    selection_curves: dict[str, dict[int, tuple[list[dict[str, Any]], dict[str, Any]]]] = {}
    all_oof_curves: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for method, pred in methods.items():
        all_oof_curves[method] = build_threshold_curve(
            gt,
            pred,
            thresholds=thresholds,
            protocol=protocol,
        )
        selection_curves[method] = {}
        for held_out in (0, 1, 2):
            selection_ids = all_ids - folds[held_out]
            selection_curves[method][held_out] = build_threshold_curve(
                _subset(gt, selection_ids),
                _subset(pred, selection_ids),
                thresholds=thresholds,
                protocol=protocol,
            )
    result: dict[str, Any] = {}
    for target in args.fdr_levels:
        target_key = f"{target:.3f}"
        result[target_key] = {}
        stitched_by_method = {}
        for method, pred in methods.items():
            stitched: dict[int, list[dict[str, Any]]] = {}
            held_rows = []
            for held_out in (0, 1, 2):
                selection_ids = all_ids - folds[held_out]
                curve, trace_parity = selection_curves[method][held_out]
                selected = {
                    coarse: _select_coarse(curve, coarse, float(target)) for coarse in COARSE
                }
                held_pred = _filter_by_coarse(
                    pred,
                    folds[held_out],
                    {coarse: float(row["threshold"]) for coarse, row in selected.items()},
                    protocol.category_mapping,
                )
                stitched.update(held_pred)
                held_metrics = evaluate(
                    _subset(gt, folds[held_out]), held_pred, args.diagnostic_latency
                )
                held_rows.append(
                    {
                        "fold": held_out,
                        "selection_images": len(selection_ids),
                        "evaluation_images": len(folds[held_out]),
                        "thresholds": selected,
                        "trace_parity": trace_parity,
                        "evaluation": held_metrics,
                    }
                )
            result[target_key][method] = {
                "folds": held_rows,
                "stitched": evaluate(gt, stitched, args.diagnostic_latency),
                "all_oof_policy_fit_after_crossfit_evaluation": {
                    coarse: _select_coarse(all_oof_curves[method][0], coarse, float(target))
                    for coarse in COARSE
                },
                "all_oof_trace_parity": all_oof_curves[method][1],
            }
            stitched_by_method[method] = stitched

        baseline = result[target_key]["native_oto"]
        for method, item in result[target_key].items():
            item["delta_score_vs_native_oto"] = _score_delta(item["stitched"], baseline["stitched"])
            item["fold_score_deltas_vs_native_oto"] = [
                _score_delta(row["evaluation"], base_row["evaluation"])
                for row, base_row in zip(item["folds"], baseline["folds"], strict=True)
            ]
            if image_to_group is not None:
                base_groups, base_counts = group_counts(
                    gt, stitched_by_method["native_oto"], image_to_group
                )
                candidate_groups, candidate_counts = group_counts(
                    gt, stitched_by_method[method], image_to_group
                )
                if base_groups != candidate_groups:
                    raise RuntimeError("Paired source groups changed")
                item["source_group_bootstrap_vs_native_oto"] = paired_bootstrap(
                    base_counts,
                    candidate_counts,
                    args.diagnostic_latency,
                    args.diagnostic_latency,
                    repetitions=args.bootstrap,
                )

    # Mirror the candidate contract exactly: keep production P40 OTO at its
    # fixed threshold and select only the OTM QHS/MS threshold on other folds.
    fixed_primary = _filter_fixed_threshold(oto, all_ids, args.primary_threshold)
    alternative_curves = {
        held_out: _alternative_threshold_curve(
            gt,
            oto,
            otm,
            all_ids - folds[held_out],
            thresholds,
            protocol,
            args.primary_threshold,
        )
        for held_out in (0, 1, 2)
    }
    all_alternative_curve = _alternative_threshold_curve(
        gt, oto, otm, all_ids, thresholds, protocol, args.primary_threshold
    )
    deployment_like = {}
    for target in args.fdr_levels:
        target_key = f"{target:.3f}"
        stitched: dict[int, list[dict[str, Any]]] = {}
        held_rows = []
        for held_out in (0, 1, 2):
            curve, trace_parity = alternative_curves[held_out]
            selected = _select_coarse(curve, "ship", float(target))
            held_pred = _route_fixed_primary_ship23(
                oto,
                otm,
                folds[held_out],
                primary_threshold=args.primary_threshold,
                alternative_threshold=float(selected["threshold"]),
            )
            stitched.update(held_pred)
            candidate_metrics = evaluate(
                _subset(gt, folds[held_out]), held_pred, args.diagnostic_latency
            )
            baseline_metrics = evaluate(
                _subset(gt, folds[held_out]),
                _subset(fixed_primary, folds[held_out]),
                args.diagnostic_latency,
            )
            held_rows.append(
                {
                    "fold": held_out,
                    "selected_otm_ship23_threshold": selected,
                    "selection_trace_parity": trace_parity,
                    "candidate": candidate_metrics,
                    "fixed_primary_baseline": baseline_metrics,
                    "delta_score": _score_delta(candidate_metrics, baseline_metrics),
                }
            )
        candidate_metrics = evaluate(gt, stitched, args.diagnostic_latency)
        baseline_metrics = evaluate(gt, fixed_primary, args.diagnostic_latency)
        item = {
            "fixed_primary_threshold": args.primary_threshold,
            "folds": held_rows,
            "stitched": candidate_metrics,
            "fixed_primary_baseline": baseline_metrics,
            "delta_score_vs_fixed_primary": _score_delta(candidate_metrics, baseline_metrics),
            "fold_score_deltas_vs_fixed_primary": [row["delta_score"] for row in held_rows],
            "all_oof_otm_ship23_threshold_fit_after_crossfit_evaluation": _select_coarse(
                all_alternative_curve[0], "ship", float(target)
            ),
            "all_oof_trace_parity": all_alternative_curve[1],
        }
        if image_to_group is not None:
            base_groups, base_counts = group_counts(gt, fixed_primary, image_to_group)
            candidate_groups, candidate_counts = group_counts(gt, stitched, image_to_group)
            if base_groups != candidate_groups:
                raise RuntimeError("Paired source groups changed")
            item["source_group_bootstrap_vs_fixed_primary"] = paired_bootstrap(
                base_counts,
                candidate_counts,
                args.diagnostic_latency,
                args.diagnostic_latency,
                repetitions=args.bootstrap,
            )
        deployment_like[target_key] = item

    # A fairer deployment decision uses the incumbent's selection-fold Ship
    # risk as the budget.  It does not require the candidate to meet an
    # arbitrary FDR that the fixed incumbent itself misses, and it never uses
    # the held-out fold to choose the OTM threshold.
    same_risk_stitched: dict[int, list[dict[str, Any]]] = {}
    same_risk_folds = []
    for held_out in (0, 1, 2):
        selection_ids = all_ids - folds[held_out]
        selection_baseline = evaluate(
            _subset(gt, selection_ids),
            _subset(fixed_primary, selection_ids),
            args.diagnostic_latency,
        )
        ship_fdr_budget = _macro_fdr(selection_baseline, (0, 1, 2, 3))
        curve, trace_parity = alternative_curves[held_out]
        selected = _select_coarse(curve, "ship", ship_fdr_budget)
        held_pred = _route_fixed_primary_ship23(
            oto,
            otm,
            folds[held_out],
            primary_threshold=args.primary_threshold,
            alternative_threshold=float(selected["threshold"]),
        )
        same_risk_stitched.update(held_pred)
        candidate_metrics = evaluate(
            _subset(gt, folds[held_out]), held_pred, args.diagnostic_latency
        )
        baseline_metrics = evaluate(
            _subset(gt, folds[held_out]),
            _subset(fixed_primary, folds[held_out]),
            args.diagnostic_latency,
        )
        same_risk_folds.append(
            {
                "fold": held_out,
                "selection_ship_macro_fdr_budget": ship_fdr_budget,
                "selected_otm_ship23_threshold": selected,
                "selection_trace_parity": trace_parity,
                "candidate": candidate_metrics,
                "fixed_primary_baseline": baseline_metrics,
                "delta_score": _score_delta(candidate_metrics, baseline_metrics),
            }
        )
    all_baseline_metrics = evaluate(gt, fixed_primary, args.diagnostic_latency)
    all_ship_fdr_budget = _macro_fdr(all_baseline_metrics, (0, 1, 2, 3))
    all_selected = _select_coarse(all_alternative_curve[0], "ship", all_ship_fdr_budget)
    same_risk_metrics = evaluate(gt, same_risk_stitched, args.diagnostic_latency)
    same_risk = {
        "selection_rule": "max_ship_macro_recall_at_or_below_fixed_primary_ship_macro_fdr",
        "fixed_primary_threshold": args.primary_threshold,
        "folds": same_risk_folds,
        "stitched": same_risk_metrics,
        "fixed_primary_baseline": all_baseline_metrics,
        "delta_score_vs_fixed_primary": _score_delta(same_risk_metrics, all_baseline_metrics),
        "fold_score_deltas_vs_fixed_primary": [row["delta_score"] for row in same_risk_folds],
        "all_oof_ship_macro_fdr_budget": all_ship_fdr_budget,
        "all_oof_otm_ship23_threshold_fit_after_crossfit_evaluation": all_selected,
        "all_oof_trace_parity": all_alternative_curve[1],
    }
    if image_to_group is not None:
        base_groups, base_counts = group_counts(gt, fixed_primary, image_to_group)
        candidate_groups, candidate_counts = group_counts(gt, same_risk_stitched, image_to_group)
        if base_groups != candidate_groups:
            raise RuntimeError("Paired source groups changed")
        same_risk["source_group_bootstrap_vs_fixed_primary"] = paired_bootstrap(
            base_counts,
            candidate_counts,
            args.diagnostic_latency,
            args.diagnostic_latency,
            repetitions=args.bootstrap,
        )

    payload = {
        "status": "complete_directional_oof",
        "protocol": "sprint20_short_p40_threefold_crossfit_macro_thresholds_v1",
        "metric": "platform_observed_20260831 fine-macro per coarse",
        "evidence_role": "outer_oof_short",
        "formal_admission": False,
        "warning": (
            "These folds are source-disjoint but use S1024/40e -> P40/40e, whereas the "
            "deployed full model uses S1024/160e -> P40/40e. Deltas are directional and "
            "the supplied common latency is diagnostic, not a shared-head endpoint timing."
        ),
        "comparison": {
            "images": len(gt),
            "ground_truth_objects": sum(len(rows) for rows in gt.values()),
            "candidate_floor": floor,
            "threshold_step": args.step,
            "diagnostic_latency_seconds": args.diagnostic_latency,
            "methods": list(methods),
        },
        "input_sha256": {
            "coco": _sha256(args.coco),
            "oto_cache": _sha256(args.oto_cache),
            "otm_cache": _sha256(args.otm_cache),
            "groups": None if args.groups is None else _sha256(args.groups),
        },
        "frontiers": result,
        "deployment_like_fixed_primary_ship23": deployment_like,
        "deployment_like_same_risk_fixed_primary_ship23": same_risk,
        "decision_boundary": (
            "Do not deploy from this file alone. Require consistent fold deltas, mature-full "
            "mechanism agreement, native-continuous sanity, exact shared parity, and latency."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                target: {
                    method: {
                        "delta": row["delta_score_vs_native_oto"],
                        "fold_deltas": row["fold_score_deltas_vs_native_oto"],
                    }
                    for method, row in methods_at_target.items()
                }
                for target, methods_at_target in result.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
