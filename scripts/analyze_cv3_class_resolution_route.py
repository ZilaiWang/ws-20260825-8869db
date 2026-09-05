#!/usr/bin/env python3
"""Evaluate a class-disjoint S1024/S1280 route with outer-CV3 thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

COARSE = ("ship", "aircraft", "vehicle")
OFFICIAL_LABELS = frozenset(range(25))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_labels(text: str) -> frozenset[int]:
    values: set[int] = set()
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            first, last = token.split("-", 1)
            start, stop = int(first), int(last)
            if stop < start:
                raise ValueError(f"descending label range: {token}")
            values.update(range(start, stop + 1))
        else:
            values.add(int(token))
    labels = frozenset(values)
    if not labels or labels - OFFICIAL_LABELS:
        raise ValueError(f"invalid or empty official label set: {sorted(labels)}")
    return labels


def _fold_map(frontier: dict[str, Any]) -> dict[int, int]:
    result: dict[int, int] = {}
    raw = frontier.get("fold_image_ids")
    if not isinstance(raw, dict) or set(raw) != {"0", "1", "2"}:
        raise ValueError("frontier must contain fold_image_ids for folds 0, 1 and 2")
    for fold_text, values in raw.items():
        fold = int(fold_text)
        for value in values:
            image_id = int(value)
            if image_id in result:
                raise ValueError(f"image {image_id} appears in multiple folds")
            result[image_id] = fold
    return result


def _thresholds(frontier: dict[str, Any], level: str) -> dict[int, float]:
    try:
        raw = frontier["frontiers"][level]["crossfit_thresholds"]
    except KeyError as error:
        raise ValueError(f"missing crossfit frontier level {level}") from error
    values = {int(fold): float(value) for fold, value in raw.items()}
    if set(values) != {0, 1, 2} or any(not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError("crossfit thresholds must cover folds 0, 1 and 2 in [0, 1]")
    return values


def _filter(
    predictions: dict[int, list[dict[str, Any]]],
    *,
    image_folds: dict[int, int],
    thresholds: dict[int, float],
    labels: Iterable[int],
) -> dict[int, list[dict[str, Any]]]:
    owned = frozenset(int(value) for value in labels)
    return {
        image_id: [
            row
            for row in predictions.get(image_id, [])
            if int(row["category_id"]) in owned
            and float(row["score"]) >= thresholds[image_folds[image_id]]
        ]
        for image_id in image_folds
    }


def _metrics(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    *,
    protocol: Any,
    latency_seconds: float | None,
) -> dict[str, Any]:
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
    platform = platform_metrics_payload(
        build_platform_observed_metrics(
            ranking,
            recall_min=protocol.recall_min,
            fdr_max=protocol.fdr_max,
            latency_seconds=latency_seconds,
            latency_max_seconds=protocol.latency_max_seconds,
        )
    )
    return {
        "pooled": {
            "recall": pooled.recall,
            "fdr": pooled.fdr,
            "tp": pooled.details["tp"],
            "fp": pooled.details["fp"],
            "fn": pooled.details["fn"],
        },
        "fine25_macro": {
            "recall": ranking.overall_recall,
            "fdr": ranking.overall_fdr,
        },
        "platform": platform,
    }


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    c = candidate["platform"]
    b = baseline["platform"]
    result: dict[str, Any] = {
        "gate_recall_pp": 100.0 * (c["gate_recall"] - b["gate_recall"]),
        "gate_fdr_pp": 100.0 * (c["gate_fdr"] - b["gate_fdr"]),
        "per_coarse": {},
    }
    if c["absolute_score"] is not None and b["absolute_score"] is not None:
        result["absolute_score"] = c["absolute_score"] - b["absolute_score"]
    for name in COARSE:
        result["per_coarse"][name] = {
            "recall_pp": 100.0
            * (c["per_coarse"][name]["macro_recall"] - b["per_coarse"][name]["macro_recall"]),
            "fdr_pp": 100.0
            * (c["per_coarse"][name]["macro_fdr"] - b["per_coarse"][name]["macro_fdr"]),
        }
    return result


def analyze(
    *,
    gt: dict[int, list[dict[str, Any]]],
    primary: dict[int, list[dict[str, Any]]],
    highres: dict[int, list[dict[str, Any]]],
    primary_frontier: dict[str, Any],
    highres_frontier: dict[str, Any],
    primary_labels: frozenset[int],
    highres_labels: frozenset[int],
    level: str | None = None,
    primary_level: str | None = None,
    highres_level: str | None = None,
    protocol: Any,
    primary_latency: float | None = None,
    highres_latency: float | None = None,
    target_image_folds: dict[int, int] | None = None,
) -> dict[str, Any]:
    if level is not None:
        if primary_level is not None or highres_level is not None:
            raise ValueError("level cannot be combined with branch-specific levels")
        primary_level = highres_level = level
    if primary_level is None or highres_level is None:
        raise ValueError("both primary_level and highres_level are required")
    if primary_labels & highres_labels:
        raise ValueError("route label ownership must be disjoint")
    if primary_labels | highres_labels != OFFICIAL_LABELS:
        raise ValueError("route label ownership must cover all 25 official labels")
    primary_folds = _fold_map(primary_frontier)
    highres_folds = _fold_map(highres_frontier)
    if primary_folds != highres_folds:
        raise ValueError("primary and highres frontiers use different fold assignments")
    source_is_target = target_image_folds is None
    image_folds = primary_folds if target_image_folds is None else target_image_folds
    if set(image_folds) != set(gt):
        raise ValueError("target fold image ids do not exactly match ground truth")
    if set(image_folds.values()) != {0, 1, 2}:
        raise ValueError("target fold map must contain folds 0, 1 and 2")
    unknown = (set(primary) | set(highres)) - set(gt)
    if unknown:
        raise ValueError(f"prediction ledgers contain unknown image ids: {sorted(unknown)[:5]}")
    p_thresholds = _thresholds(primary_frontier, primary_level)
    h_thresholds = _thresholds(highres_frontier, highres_level)
    primary_all = _filter(
        primary,
        image_folds=image_folds,
        thresholds=p_thresholds,
        labels=OFFICIAL_LABELS,
    )
    highres_all = _filter(
        highres,
        image_folds=image_folds,
        thresholds=h_thresholds,
        labels=OFFICIAL_LABELS,
    )
    routed_primary = _filter(
        primary,
        image_folds=image_folds,
        thresholds=p_thresholds,
        labels=primary_labels,
    )
    routed_highres = _filter(
        highres,
        image_folds=image_folds,
        thresholds=h_thresholds,
        labels=highres_labels,
    )
    routed = {
        image_id: [*routed_primary[image_id], *routed_highres[image_id]] for image_id in sorted(gt)
    }
    route_latency = (
        None
        if primary_latency is None or highres_latency is None
        else primary_latency + highres_latency
    )
    baseline = _metrics(gt, primary_all, protocol=protocol, latency_seconds=primary_latency)
    highres_result = _metrics(gt, highres_all, protocol=protocol, latency_seconds=highres_latency)
    route = _metrics(gt, routed, protocol=protocol, latency_seconds=route_latency)
    return {
        "schema_version": "cv3_class_resolution_route_v1",
        "metric_protocol": "platform_observed_20260831",
        "selection_uses_held_out_labels": False,
        "threshold_source_is_evaluation_target": source_is_target,
        "target_image_count": len(image_folds),
        "fdr_level_is_training_fold_constraint_not_heldout_guarantee": True,
        "fdr_level": (primary_level if primary_level == highres_level else None),
        "fdr_level_by_branch": {
            "primary": primary_level,
            "highres": highres_level,
        },
        "primary_labels": sorted(primary_labels),
        "highres_labels": sorted(highres_labels),
        "primary_threshold_by_fold": p_thresholds,
        "highres_threshold_by_fold": h_thresholds,
        "latency_contract": {
            "primary_seconds": primary_latency,
            "highres_seconds": highres_latency,
            "route_seconds_sequential_sum": route_latency,
        },
        "primary": baseline,
        "highres": highres_result,
        "route": route,
        "route_vs_primary": _delta(route, baseline),
        "route_vs_highres": _delta(route, highres_result),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--primary-pred", type=Path, required=True)
    parser.add_argument("--highres-pred", type=Path, required=True)
    parser.add_argument("--primary-frontier", type=Path, required=True)
    parser.add_argument("--highres-frontier", type=Path, required=True)
    parser.add_argument("--primary-labels", default="0-23")
    parser.add_argument("--highres-labels", default="24")
    parser.add_argument("--fdr-level", default="0.150")
    parser.add_argument("--primary-fdr-level")
    parser.add_argument("--highres-fdr-level")
    parser.add_argument("--primary-latency", type=float)
    parser.add_argument("--highres-latency", type=float)
    parser.add_argument(
        "--transfer-target-folds-from-gt",
        action="store_true",
        help=(
            "apply thresholds selected on the source frontiers to the target GT fold "
            "assignments; required for frozen Hard/Sentinel transfer"
        ),
    )
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.gt,
        args.primary_pred,
        args.highres_pred,
        args.primary_frontier,
        args.highres_frontier,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    primary_frontier = json.loads(args.primary_frontier.read_text(encoding="utf-8"))
    highres_frontier = json.loads(args.highres_frontier.read_text(encoding="utf-8"))
    result = analyze(
        gt=load_coco_ground_truth(args.gt),
        primary=load_coco_predictions(args.primary_pred),
        highres=load_coco_predictions(args.highres_pred),
        primary_frontier=primary_frontier,
        highres_frontier=highres_frontier,
        primary_labels=parse_labels(args.primary_labels),
        highres_labels=parse_labels(args.highres_labels),
        level=(
            args.fdr_level
            if args.primary_fdr_level is None and args.highres_fdr_level is None
            else None
        ),
        primary_level=args.primary_fdr_level,
        highres_level=args.highres_fdr_level,
        protocol=parse_evaluation_protocol(load_config(args.project_config)),
        primary_latency=args.primary_latency,
        highres_latency=args.highres_latency,
        target_image_folds=(
            {int(row["id"]): int(row["fold"]) for row in raw_gt["images"]}
            if args.transfer_target_folds_from_gt
            else None
        ),
    )
    result["input_sha256"] = {
        "gt": _sha256(args.gt),
        "primary_pred": _sha256(args.primary_pred),
        "highres_pred": _sha256(args.highres_pred),
        "primary_frontier": _sha256(args.primary_frontier),
        "highres_frontier": _sha256(args.highres_frontier),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["route_vs_primary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
