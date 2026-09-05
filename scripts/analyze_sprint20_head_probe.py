#!/usr/bin/env python3
"""Compare mature P40 native OTO/OTM readouts without overstating evidence.

The input caches must differ only in the native YOLO26 head selected before
inference.  This tool reports fixed-policy metrics and same-split diagnostic
frontiers.  A ``full_seen`` result is never converted into deployment
admission, irrespective of its score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import build_threshold_curve
from rsdet.evaluation.absolute_score import platform_confirmed_score
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config
from sprint20.evaluation import cache_predictions, gt_from_coco

COARSE = ("ship", "aircraft", "vehicle")
IMMUTABLE_CACHE_KEYS = (
    "role",
    "base_commit",
    "config_sha256",
    "coco_sha256",
    "weight_sha256",
    "pipeline",
    "inference_model",
    "aircraft_d4_applied",
    "threshold_stage",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timing(cache: dict[str, Any]) -> dict[str, float | int]:
    values = [float(row["wall_seconds_excluding_read"]) for row in cache["images"]]
    warm = values[1:] if len(values) > 1 else values
    return {
        "images": len(values),
        "cold_first_seconds": values[0],
        "warm_median_seconds": statistics.median(warm),
        "warm_mean_seconds": statistics.fmean(warm),
        "total_seconds_excluding_read": sum(values),
    }


def _metric_payload(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    protocol: Any,
    latency: float,
) -> dict[str, Any]:
    ranking = evaluate_ranking_metrics(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    rows = {
        name: {
            "recall": float(ranking.per_coarse[name].macro_recall),
            "fdr": float(ranking.per_coarse[name].macro_fdr),
        }
        for name in COARSE
    }
    score = platform_confirmed_score(rows, latency)
    return {
        "platform_score_with_nonofficial_cache_latency": score,
        "platform_gate_recall": float(score["hard_gates"]["macro_coarse_recall"]),
        "platform_gate_fdr": float(score["hard_gates"]["macro_coarse_fdr"]),
        "per_coarse": {
            name: {
                "macro_recall": float(ranking.per_coarse[name].macro_recall),
                "macro_fdr": float(ranking.per_coarse[name].macro_fdr),
                "pooled_recall": float(ranking.per_coarse[name].pooled_recall),
                "pooled_fdr": float(ranking.per_coarse[name].pooled_fdr),
            }
            for name in COARSE
        },
        "per_fine": {
            str(label): {
                "tp": int(item.tp),
                "fp": int(item.fp),
                "fn": int(item.fn),
                "recall": float(item.recall),
                "fdr": float(item.fdr),
            }
            for label, item in ranking.per_fine.items()
        },
    }


def _compact_point(row: dict[str, Any]) -> dict[str, Any]:
    output = {
        key: row[key]
        for key in (
            "threshold",
            "detections_kept",
            "tp",
            "fp",
            "fn",
            "pooled_recall",
            "pooled_fdr",
            "platform_gate_recall",
            "platform_gate_fdr",
            "platform_quality_score",
        )
    }
    output["per_coarse"] = {
        name: {
            "macro_recall": row[f"{name}_macro_recall"],
            "macro_fdr": row[f"{name}_macro_fdr"],
            "tp": row[f"{name}_tp"],
            "fp": row[f"{name}_fp"],
            "fn": row[f"{name}_fn"],
        }
        for name in COARSE
    }
    return output


def _point_at(curve: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    return min(curve, key=lambda row: abs(float(row["threshold"]) - threshold))


def _best_under_fdr(curve: list[dict[str, Any]], target: float) -> dict[str, Any]:
    feasible = [row for row in curve if float(row["platform_gate_fdr"]) <= target]
    if not feasible:
        return min(
            curve,
            key=lambda row: (
                float(row["platform_gate_fdr"]),
                -float(row["platform_gate_recall"]),
            ),
        )
    return max(
        feasible,
        key=lambda row: (
            float(row["platform_gate_recall"]),
            float(row["platform_quality_score"]),
            -float(row["platform_gate_fdr"]),
        ),
    )


def _best_under_fp(curve: list[dict[str, Any]], fp_budget: int) -> dict[str, Any]:
    feasible = [row for row in curve if int(row["fp"]) <= fp_budget]
    if not feasible:
        raise RuntimeError("Threshold grid has no point under the requested FP budget")
    return max(
        feasible,
        key=lambda row: (
            float(row["platform_gate_recall"]),
            float(row["platform_quality_score"]),
            -int(row["fp"]),
        ),
    )


def _fine_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for label in sorted(baseline, key=int):
        left, right = baseline[label], candidate[label]
        output[label] = {
            "delta_tp": int(right["tp"]) - int(left["tp"]),
            "delta_fp": int(right["fp"]) - int(left["fp"]),
            "delta_fn": int(right["fn"]) - int(left["fn"]),
            "delta_recall": float(right["recall"]) - float(left["recall"]),
            "delta_fdr": float(right["fdr"]) - float(left["fdr"]),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--oto-cache", type=Path, required=True)
    parser.add_argument("--otm-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-threshold", type=float, default=0.536)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--fdr-levels", type=float, nargs="+", default=(0.10, 0.12, 0.15, 0.20))
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not 0.0 < args.step <= 1.0:
        raise ValueError("step must be in (0,1]")

    coco = _read(args.coco)
    oto, otm = _read(args.oto_cache), _read(args.otm_cache)
    if oto.get("head") != "oto" or otm.get("head") != "otm":
        raise ValueError("Expected native OTO and native OTM caches")
    changed = {
        key: {"oto": oto.get(key), "otm": otm.get(key)}
        for key in IMMUTABLE_CACHE_KEYS
        if oto.get(key) != otm.get(key)
    }
    if changed:
        raise ValueError(f"Head-only comparison changed immutable fields: {changed}")
    if oto["coco_sha256"] != _sha256(args.coco):
        raise ValueError("COCO SHA does not match caches")
    oto_pixels = {int(row["image_id"]): row["pixel_sha256"] for row in oto["images"]}
    otm_pixels = {int(row["image_id"]): row["pixel_sha256"] for row in otm["images"]}
    if oto_pixels != otm_pixels or not oto_pixels:
        raise ValueError("OTO/OTM did not use identical nonempty image pixels")

    gt = gt_from_coco(coco)
    floor = float(oto["pipeline"]["score_threshold"])
    if floor != float(otm["pipeline"]["score_threshold"]):
        raise ValueError("Head caches use different candidate floors")
    oto_pred = cache_predictions(oto, floor)
    otm_pred = cache_predictions(otm, floor)
    if set(gt) != set(oto_pred) or set(gt) != set(otm_pred):
        raise ValueError("Caches must cover every positive and negative COCO image")

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    timing = {"oto": _timing(oto), "otm": _timing(otm)}
    fixed = {}
    fixed_predictions = {}
    for name, pred, cache in (("oto", oto_pred, oto), ("otm", otm_pred, otm)):
        fixed_pred = {
            image_id: [row for row in records if float(row["score"]) >= args.fixed_threshold]
            for image_id, records in pred.items()
        }
        fixed_predictions[name] = fixed_pred
        fixed[name] = _metric_payload(
            gt,
            fixed_pred,
            protocol,
            float(timing[name]["warm_median_seconds"]),
        )
        fixed[name]["cache_role"] = cache["role"]

    ownership_specs = {
        "otm_ship_all": [0, 1, 2, 3],
        "otm_ship_common_2_3": [2, 3],
        "otm_fsc": [24],
        "otm_ship_all_and_fsc": [0, 1, 2, 3, 24],
    }
    fixed_ownership = {}
    for name, labels in ownership_specs.items():
        routed = {
            image_id: [
                row
                for row in fixed_predictions["oto"][image_id]
                if int(row["category_id"]) not in labels
            ]
            + [
                row
                for row in fixed_predictions["otm"][image_id]
                if int(row["category_id"]) in labels
            ]
            for image_id in gt
        }
        metrics = _metric_payload(
            gt,
            routed,
            protocol,
            float(timing["oto"]["warm_median_seconds"]),
        )
        metrics["alternative_labels"] = labels
        metrics["delta_score_without_shared_overhead"] = (
            metrics["platform_score_with_nonofficial_cache_latency"]["total_score"]
            - fixed["oto"]["platform_score_with_nonofficial_cache_latency"]["total_score"]
        )
        fixed_ownership[name] = metrics

    thresholds = build_threshold_grid(floor, 1.0, args.step)
    thresholds = sorted(set(thresholds + [float(args.fixed_threshold)]))
    curves, trace_audits = {}, {}
    for name, pred in (("oto", oto_pred), ("otm", otm_pred)):
        curves[name], trace_audits[name] = build_threshold_curve(
            gt,
            pred,
            thresholds=thresholds,
            protocol=protocol,
        )
    fixed_curve = {name: _point_at(curve, args.fixed_threshold) for name, curve in curves.items()}
    oto_fp_budget = int(fixed_curve["oto"]["fp"])
    frontiers = {
        name: {
            "same_global_fp_as_oto_fixed": _compact_point(_best_under_fp(curve, oto_fp_budget)),
            "fdr": {
                f"{target:.3f}": _compact_point(_best_under_fdr(curve, target))
                for target in args.fdr_levels
            },
        }
        for name, curve in curves.items()
    }

    fixed_deltas = {
        "platform_gate_recall": fixed["otm"]["platform_gate_recall"]
        - fixed["oto"]["platform_gate_recall"],
        "platform_gate_fdr": fixed["otm"]["platform_gate_fdr"] - fixed["oto"]["platform_gate_fdr"],
        "nonofficial_cache_latency_seconds": float(timing["otm"]["warm_median_seconds"])
        - float(timing["oto"]["warm_median_seconds"]),
        "fine": _fine_delta(fixed["oto"]["per_fine"], fixed["otm"]["per_fine"]),
    }
    weak = ["0", "1", "2", "3", "24"]
    weak_tp_gain = sum(fixed_deltas["fine"][label]["delta_tp"] for label in weak)
    weak_fp_gain = sum(fixed_deltas["fine"][label]["delta_fp"] for label in weak)
    output = {
        "status": "complete_diagnostic_only",
        "protocol": "sprint20_mature_p40_native_head_probe_v1",
        "evidence_role": oto["role"],
        "formal_admission": False,
        "warning": (
            "Full-seen labels may diagnose head behavior but cannot select a deployment policy "
            "or estimate hidden-set generalization. Cache latency excludes image I/O and is not "
            "official endpoint latency."
        ),
        "comparison_contract": {
            "only_intended_difference": "native YOLO26 OTO versus native OTM readout",
            "immutable_cache_keys_equal": True,
            "identical_image_pixels": True,
            "images": len(gt),
            "ground_truth_objects": sum(len(records) for records in gt.values()),
            "candidate_floor": floor,
            "fixed_threshold": args.fixed_threshold,
            "threshold_grid_step": args.step,
            "weight_sha256": oto["weight_sha256"],
        },
        "input_sha256": {
            "coco": _sha256(args.coco),
            "oto_cache": _sha256(args.oto_cache),
            "otm_cache": _sha256(args.otm_cache),
        },
        "timing_diagnostic": timing,
        "fixed_threshold": fixed,
        "fixed_threshold_class_ownership": fixed_ownership,
        "fixed_threshold_delta_otm_minus_oto": fixed_deltas,
        "weak_class_object_delta_at_fixed_threshold": {
            "labels": [0, 1, 2, 3, 24],
            "delta_tp": weak_tp_gain,
            "delta_fp": weak_fp_gain,
        },
        "same_split_oracle_frontiers": frontiers,
        "trace_parity": trace_audits,
        "decision": {
            "native_otm_direct_admission": False,
            "shared_head_deployment_admission": False,
            "reason": (
                "This is full-seen diagnostic evidence. A useful OTM signal would still require "
                "a policy frozen outside a mature source-isolated evaluation set; otherwise retain "
                "the incumbent native OTO path."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "fixed_threshold_delta": fixed_deltas,
                "weak_class_object_delta": output["weak_class_object_delta_at_fixed_threshold"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
