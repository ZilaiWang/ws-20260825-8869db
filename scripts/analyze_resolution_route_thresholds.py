#!/usr/bin/env python3
"""Calibrate a class-disjoint expert threshold on frozen development folds.

The primary branch keeps its deployed threshold.  Only the expert-owned labels
use the candidate threshold grid, exactly matching ``ResolutionLabelRoute`` in
the Docker runtime.  This script is for development-fold calibration; a held-
out confirmation fold must remain untouched until the threshold is frozen.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsdet.evaluation.absolute_score import fdr_points, recall_points
from rsdet.evaluation.coco import load_coco_ground_truth
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import COARSE_ORDER, build_platform_observed_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config
from scripts.compose_class_disjoint_predictions import compose


@dataclass(frozen=True)
class FoldInput:
    name: str
    gt: Path
    primary: Path
    expert: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"prediction file must contain a COCO list: {path}")
    return payload


def _prediction_map(
    rows: list[dict[str, Any]], gt: dict[int, list[dict[str, Any]]]
) -> dict[int, list[dict[str, Any]]]:
    result = {image_id: [] for image_id in gt}
    for row in rows:
        image_id = int(row["image_id"])
        if image_id not in result:
            raise ValueError(f"prediction image id is outside GT: {image_id}")
        box = [float(value) for value in row["bbox"]]
        if len(box) != 4 or box[2] < 0.0 or box[3] < 0.0:
            raise ValueError(f"invalid COCO bbox: {box}")
        result[image_id].append(
            {
                "bbox_xyxy": [box[0], box[1], box[0] + box[2], box[1] + box[3]],
                "category_id": int(row["category_id"]),
                "score": float(row["score"]),
            }
        )
    return result


def _evaluate(
    gt: dict[int, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    *,
    protocol: Any,
) -> dict[str, Any]:
    pred = _prediction_map(rows, gt)
    pooled = evaluate_predictions(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    ranking = evaluate_ranking_metrics(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    platform = build_platform_observed_metrics(
        ranking,
        recall_min=protocol.recall_min,
        fdr_max=protocol.fdr_max,
        latency_seconds=None,
        latency_max_seconds=protocol.latency_max_seconds,
    )
    per_coarse = {
        name: {
            "macro_recall": platform.coarse_recall[name],
            "macro_fdr": platform.coarse_fdr[name],
            "tp": pooled.per_class[name].tp,
            "fp": pooled.per_class[name].fp,
            "fn": pooled.per_class[name].fn,
        }
        for name in COARSE_ORDER
    }
    quality = sum(
        recall_points(per_coarse[name]["macro_recall"])
        + fdr_points(per_coarse[name]["macro_fdr"])
        for name in COARSE_ORDER
    ) / 7.0
    return {
        "gate_recall": platform.gate_recall,
        "gate_fdr": platform.gate_fdr,
        "quality_contribution_out_of_600_over_7": quality,
        "per_coarse": per_coarse,
    }


def _grid(start: float, stop: float, step: float) -> list[float]:
    if not 0.0 <= start <= stop <= 1.0 or step <= 0.0:
        raise ValueError("invalid threshold grid")
    count = int(round((stop - start) / step))
    values = [round(start + index * step, 10) for index in range(count + 1)]
    if values[-1] < stop - 1e-9:
        values.append(stop)
    return values


def select_point(
    curve: list[dict[str, Any]], *, max_vehicle_recall_drop_pp: float
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return recall-protected and unconstrained quality selections."""

    if not curve:
        raise ValueError("curve must not be empty")
    unconstrained = max(
        curve,
        key=lambda row: (
            float(row["mean_quality_delta"]),
            float(row["worst_vehicle_recall_delta_pp"]),
            float(row["expert_threshold"]),
        ),
    )
    feasible = [
        row
        for row in curve
        if float(row["worst_vehicle_recall_delta_pp"])
        >= -float(max_vehicle_recall_drop_pp)
        and float(row["mean_quality_delta"]) > 0.0
        and bool(row["all_fold_fdr_not_worse"])
    ]
    if not feasible:
        return None, unconstrained
    protected = max(
        feasible,
        key=lambda row: (
            float(row["mean_quality_delta"]),
            float(row["worst_vehicle_recall_delta_pp"]),
            float(row["expert_threshold"]),
        ),
    )
    return protected, unconstrained


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        nargs=4,
        action="append",
        metavar=("NAME", "GT", "PRIMARY", "EXPERT"),
        required=True,
    )
    parser.add_argument("--primary-threshold", type=float, required=True)
    parser.add_argument("--expert-start", type=float, default=0.40)
    parser.add_argument("--expert-stop", type=float, default=0.60)
    parser.add_argument("--expert-step", type=float, default=0.005)
    parser.add_argument("--max-vehicle-recall-drop-pp", type=float, default=2.2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    if not 0.0 <= args.primary_threshold <= 1.0:
        raise ValueError("primary threshold must be in [0, 1]")
    folds = [
        FoldInput(name, Path(gt), Path(primary), Path(expert))
        for name, gt, primary, expert in args.fold
    ]
    if len({fold.name for fold in folds}) != len(folds):
        raise ValueError("fold names must be unique")

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    loaded: dict[str, tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    baselines: dict[str, dict[str, Any]] = {}
    inputs: dict[str, Any] = {}
    for fold in folds:
        gt = load_coco_ground_truth(fold.gt)
        primary = _read_rows(fold.primary)
        expert = _read_rows(fold.expert)
        loaded[fold.name] = (gt, primary, expert)
        baseline_rows = [
            dict(row)
            for row in primary
            if float(row["score"]) >= args.primary_threshold
        ]
        baselines[fold.name] = _evaluate(gt, baseline_rows, protocol=protocol)
        inputs[fold.name] = {
            "gt": str(fold.gt),
            "primary": str(fold.primary),
            "expert": str(fold.expert),
            "sha256": {
                "gt": _sha256(fold.gt),
                "primary": _sha256(fold.primary),
                "expert": _sha256(fold.expert),
            },
        }

    curve: list[dict[str, Any]] = []
    for threshold in _grid(args.expert_start, args.expert_stop, args.expert_step):
        per_fold: dict[str, Any] = {}
        for fold in folds:
            gt, primary, expert = loaded[fold.name]
            rows = compose(
                primary,
                expert,
                primary_labels=frozenset(range(24)),
                expert_labels=frozenset({24}),
                primary_threshold=args.primary_threshold,
                expert_threshold=threshold,
            )
            candidate = _evaluate(gt, rows, protocol=protocol)
            baseline = baselines[fold.name]
            vehicle_delta = 100.0 * (
                candidate["per_coarse"]["vehicle"]["macro_recall"]
                - baseline["per_coarse"]["vehicle"]["macro_recall"]
            )
            quality_delta = (
                candidate["quality_contribution_out_of_600_over_7"]
                - baseline["quality_contribution_out_of_600_over_7"]
            )
            per_fold[fold.name] = {
                "baseline": baseline,
                "candidate": candidate,
                "vehicle_recall_delta_pp": vehicle_delta,
                "vehicle_fdr_delta_pp": 100.0
                * (
                    candidate["per_coarse"]["vehicle"]["macro_fdr"]
                    - baseline["per_coarse"]["vehicle"]["macro_fdr"]
                ),
                "quality_delta": quality_delta,
            }
        curve.append(
            {
                "expert_threshold": threshold,
                "mean_quality_delta": fmean(
                    row["quality_delta"] for row in per_fold.values()
                ),
                "worst_vehicle_recall_delta_pp": min(
                    row["vehicle_recall_delta_pp"] for row in per_fold.values()
                ),
                "mean_vehicle_recall_delta_pp": fmean(
                    row["vehicle_recall_delta_pp"] for row in per_fold.values()
                ),
                "mean_vehicle_fdr_delta_pp": fmean(
                    row["vehicle_fdr_delta_pp"] for row in per_fold.values()
                ),
                "all_fold_fdr_not_worse": all(
                    row["candidate"]["gate_fdr"] <= row["baseline"]["gate_fdr"] + 1e-12
                    for row in per_fold.values()
                ),
                "folds": per_fold,
            }
        )

    protected, unconstrained = select_point(
        curve,
        max_vehicle_recall_drop_pp=args.max_vehicle_recall_drop_pp,
    )
    result = {
        "status": "complete",
        "schema_version": "class_disjoint_resolution_route_threshold_calibration_v1",
        "metric_protocol": "platform_observed_20260831",
        "warning": (
            "Select only on development folds. Freeze the returned threshold before "
            "opening an untouched confirmation fold or formal platform result."
        ),
        "route": {
            "primary_labels": "0-23",
            "expert_labels": "24",
            "primary_threshold": args.primary_threshold,
        },
        "guard": {
            "max_vehicle_recall_drop_pp_per_fold": args.max_vehicle_recall_drop_pp,
            "require_positive_mean_quality_delta": True,
            "require_gate_fdr_not_worse_each_fold": True,
        },
        "selection": {
            "recall_protected": protected,
            "diagnostic_unconstrained": unconstrained,
            "admission": protected is not None,
        },
        "inputs": inputs,
        "curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["selection"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
