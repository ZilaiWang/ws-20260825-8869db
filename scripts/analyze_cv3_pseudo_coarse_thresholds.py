#!/usr/bin/env python3
"""Cross-fit one score threshold per coarse class on pseudo-10K folds.

The official pooled gate is global, while detector score distributions can be
very different for ship, aircraft, and vehicle.  A single threshold therefore
needlessly discards reliable classes or admits too many predictions from a
noisy class.  This script selects three thresholds jointly on the two training
folds and evaluates them on the untouched held-out fold.

Only score-prefix filters are optimized.  The candidate boxes, fine labels,
and within-class order are unchanged, so TP/FP traces computed at the candidate
floor remain valid for every threshold tuple.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import (
    evaluate_predictions,
    evaluate_predictions_with_trace,
    evaluate_ranking_metrics,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scoped(
    mapping: dict[int, list[dict[str, Any]]], image_ids: set[int]
) -> dict[int, list[dict[str, Any]]]:
    return {image_id: list(mapping.get(image_id, [])) for image_id in sorted(image_ids)}


def _trace_rows(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    *,
    class_names: list[str],
    category_mapping: dict[int, str],
    iou_thresholds: dict[str, float],
) -> list[tuple[str, float, bool]]:
    """Return stable ``(coarse, score, is_tp)`` rows at the candidate floor."""

    _, trace = evaluate_predictions_with_trace(
        gt,
        pred,
        class_names=class_names,
        category_mapping=category_mapping,
        iou_thresholds=iou_thresholds,
    )
    rows = [
        (item.class_name, float(item.score), True) for item in trace.matches
    ] + [
        (item.class_name, float(item.score), False)
        for item in trace.unmatched_predictions
    ]
    return rows


def _prefix_counts(
    rows: Iterable[tuple[str, float, bool]],
    thresholds: list[float],
    coarse_names: tuple[str, ...],
) -> dict[str, list[tuple[int, int]]]:
    """Compute TP/FP counts for every coarse class and threshold."""

    grouped: dict[str, list[tuple[float, bool]]] = {name: [] for name in coarse_names}
    for coarse, score, is_tp in rows:
        grouped[coarse].append((score, is_tp))
    output: dict[str, list[tuple[int, int]]] = {}
    for coarse in coarse_names:
        ordered = sorted(grouped[coarse], reverse=True)
        output[coarse] = [
            (
                sum(1 for score, is_tp in ordered if score >= threshold and is_tp),
                sum(1 for score, is_tp in ordered if score >= threshold and not is_tp),
            )
            for threshold in thresholds
        ]
    return output


def select_joint_thresholds(
    counts: dict[str, list[tuple[int, int]]],
    thresholds: list[float],
    *,
    target_fdr: float,
    coarse_names: tuple[str, ...] = ("ship", "aircraft", "vehicle"),
) -> dict[str, Any]:
    """Select the maximum-TP feasible threshold tuple deterministically."""

    if not 0.0 <= target_fdr < 1.0:
        raise ValueError("target_fdr must be in [0, 1)")
    if not thresholds:
        raise ValueError("thresholds must not be empty")
    if set(counts) != set(coarse_names):
        raise ValueError("counts must contain exactly the configured coarse classes")
    if any(len(counts[name]) != len(thresholds) for name in coarse_names):
        raise ValueError("each count curve must align with thresholds")

    best: tuple[tuple[float, ...], tuple[int, ...], int, int] | None = None
    # Three classes and a 200-point grid mean 8M integer additions, which is
    # small and keeps the optimizer fully auditable.
    a, b, c = coarse_names
    for ia, (tp_a, fp_a) in enumerate(counts[a]):
        for ib, (tp_b, fp_b) in enumerate(counts[b]):
            base_tp = tp_a + tp_b
            base_fp = fp_a + fp_b
            for ic, (tp_c, fp_c) in enumerate(counts[c]):
                tp = base_tp + tp_c
                fp = base_fp + fp_c
                selected = tp + fp
                fdr = fp / selected if selected else 0.0
                if fdr > target_fdr + 1e-12:
                    continue
                indices = (ia, ib, ic)
                threshold_tuple = tuple(thresholds[index] for index in indices)
                # Maximize recall/TP; then minimize FP; then prefer the higher
                # (more conservative) lexicographic threshold tuple.
                key = (tp, -fp, threshold_tuple)
                if best is None or key > (best[2], -best[3], best[0]):
                    best = (threshold_tuple, indices, tp, fp)
    if best is None:
        raise RuntimeError("no feasible threshold tuple")
    threshold_tuple, _, tp, fp = best
    return {
        "thresholds": dict(zip(coarse_names, threshold_tuple, strict=True)),
        "train_tp": tp,
        "train_fp": fp,
        "train_fdr": fp / (tp + fp) if tp + fp else 0.0,
    }


def _filter_predictions(
    pred: dict[int, list[dict[str, Any]]],
    image_ids: set[int],
    thresholds: dict[str, float],
    category_mapping: dict[int, str],
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            item
            for item in pred.get(image_id, [])
            if float(item["score"])
            >= float(thresholds[category_mapping[int(item["category_id"])]])
        ]
        for image_id in sorted(image_ids)
    }


def _metric_payload(metrics: Any, ranking: Any) -> dict[str, Any]:
    return {
        "recall": metrics.recall,
        "fdr": metrics.fdr,
        "tp": metrics.details["tp"],
        "fp": metrics.details["fp"],
        "fn": metrics.details["fn"],
        "per_coarse": {
            name: {
                "recall": item.recall,
                "fdr": item.fdr,
                "tp": item.tp,
                "fp": item.fp,
                "fn": item.fn,
            }
            for name, item in metrics.per_class.items()
        },
        "macro_recall": ranking.overall_recall,
        "macro_fdr": ranking.overall_fdr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--threshold-start", type=float, default=0.001)
    parser.add_argument("--threshold-stop", type=float, default=0.996)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument(
        "--fdr-levels", type=float, nargs="+", default=(0.12, 0.15, 0.17, 0.20)
    )
    args = parser.parse_args()

    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_images = {
        fold: {int(item["id"]) for item in raw_gt["images"] if int(item["fold"]) == fold}
        for fold in (0, 1, 2)
    }
    if any(not image_ids for image_ids in fold_images.values()):
        raise ValueError("ground truth must contain non-empty folds 0, 1 and 2")
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    coarse_names = tuple(protocol.class_names)
    if set(coarse_names) != {"ship", "aircraft", "vehicle"}:
        raise ValueError("this audit requires the three official coarse classes")
    gt = load_coco_ground_truth(args.gt)
    pred = load_coco_predictions(args.pred)
    thresholds = build_threshold_grid(
        args.threshold_start, args.threshold_stop, args.threshold_step
    )

    train_counts: dict[int, dict[str, list[tuple[int, int]]]] = {}
    for held_out in (0, 1, 2):
        train_ids = set().union(
            *(ids for fold, ids in fold_images.items() if fold != held_out)
        )
        rows = _trace_rows(
            _scoped(gt, train_ids),
            _scoped(pred, train_ids),
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
        )
        train_counts[held_out] = _prefix_counts(rows, thresholds, coarse_names)

    results: dict[str, Any] = {}
    for target_fdr in args.fdr_levels:
        choices: dict[int, dict[str, Any]] = {}
        selected: dict[int, list[dict[str, Any]]] = {}
        for held_out in (0, 1, 2):
            choice = select_joint_thresholds(
                train_counts[held_out],
                thresholds,
                target_fdr=float(target_fdr),
                coarse_names=coarse_names,
            )
            choices[held_out] = choice
            selected.update(
                _filter_predictions(
                    pred,
                    fold_images[held_out],
                    choice["thresholds"],
                    protocol.category_mapping,
                )
            )
        metrics = evaluate_predictions(
            gt,
            selected,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
        )
        ranking = evaluate_ranking_metrics(
            gt,
            selected,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
            require_complete_taxonomy=True,
        )
        results[f"{target_fdr:.3f}"] = {
            "crossfit_choices": {str(fold): choice for fold, choice in choices.items()},
            "crossfit": _metric_payload(metrics, ranking),
        }

    pooled_rows = _trace_rows(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    pooled_counts = _prefix_counts(pooled_rows, thresholds, coarse_names)
    deployment_choices = {
        f"{level:.3f}": select_joint_thresholds(
            pooled_counts,
            thresholds,
            target_fdr=float(level),
            coarse_names=coarse_names,
        )
        for level in args.fdr_levels
    }
    payload = {
        "status": "complete",
        "protocol": "formal_cv3_crossfit_coarse_thresholds_pseudo10k_v1",
        "warning": (
            "Pseudo-10K is a deployment proxy, not an independent benchmark. "
            "Deployment thresholds use all pseudo folds and are not evaluation estimates."
        ),
        "inputs": {
            "gt": str(args.gt.resolve()),
            "gt_sha256": _sha256(args.gt),
            "pred": str(args.pred.resolve()),
            "pred_sha256": _sha256(args.pred),
        },
        "threshold_grid": {
            "start": args.threshold_start,
            "stop": args.threshold_stop,
            "step": args.threshold_step,
            "count": len(thresholds),
        },
        "fold_image_ids": {str(key): sorted(value) for key, value in fold_images.items()},
        "frontiers": results,
        "deployment_choices": deployment_choices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
