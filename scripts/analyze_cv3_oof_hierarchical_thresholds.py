#!/usr/bin/env python3
"""Cross-fit hierarchical fine thresholds with the exact official matcher."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.hierarchical_thresholds import (
    build_hierarchical_curves,
    filter_by_thresholds,
    fit_hierarchical_thresholds,
)
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config

T = TypeVar("T")


def _scoped(mapping: dict[int, list[T]], image_ids: set[int]) -> dict[int, list[T]]:
    return {image_id: list(mapping.get(image_id, [])) for image_id in sorted(image_ids)}


def _load_predictions_compat(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Load COCO predictions or the audited legacy ``bbox_xyxy`` ledger."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, list) or not document or "bbox_xyxy" not in document[0]:
        return load_coco_predictions(path)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, item in enumerate(document):
        box = [float(value) for value in item["bbox_xyxy"]]
        score = float(item["score"])
        if (
            len(box) != 4
            or not all(math.isfinite(value) for value in box)
            or box[2] < box[0]
            or box[3] < box[1]
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError(f"invalid legacy prediction at index {index}")
        grouped.setdefault(int(item["image_id"]), []).append(
            {
                "bbox_xyxy": box,
                "category_id": int(item["category_id"]),
                "score": score,
            }
        )
    return grouped


def _metric_payload(gt: dict[int, list[dict[str, Any]]], pred: dict[int, list[dict[str, Any]]], protocol: Any, *, complete: bool = True) -> dict[str, Any]:
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
        require_complete_taxonomy=complete,
    )
    return {
        "recall": pooled.recall,
        "fdr": pooled.fdr,
        "tp": pooled.details["tp"],
        "fp": pooled.details["fp"],
        "fn": pooled.details["fn"],
        "official_macro_recall": ranking.overall_recall,
        "official_macro_fdr": ranking.overall_fdr,
        "per_coarse": {
            name: {
                "pooled_recall": pooled.per_class[name].recall,
                "pooled_fdr": pooled.per_class[name].fdr,
                "macro_recall": item.macro_recall,
                "macro_fdr": item.macro_fdr,
            }
            for name, item in ranking.per_coarse.items()
        },
        "per_fine": {
            str(category_id): {
                "coarse_class": item.coarse_class,
                "recall": item.recall,
                "fdr": item.fdr,
                "tp": item.tp,
                "fp": item.fp,
                "fn": item.fn,
            }
            for category_id, item in ranking.per_fine.items()
        },
    }


def _filter_global(predictions: dict[int, list[dict[str, Any]]], threshold: float) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [item for item in items if float(item["score"]) >= threshold]
        for image_id, items in predictions.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--threshold-start", type=float, default=0.001)
    parser.add_argument("--threshold-stop", type=float, default=0.996)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--fdr-levels", type=float, nargs="+", default=(0.10, 0.12, 0.15, 0.20))
    parser.add_argument("--prior-strength", type=float, default=50.0)
    parser.add_argument("--minimum-evidence", type=int, default=10)
    args = parser.parse_args()

    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_images = {
        fold: {int(item["id"]) for item in raw_gt["images"] if int(item["fold"]) == fold}
        for fold in (0, 1, 2)
    }
    if any(not values for values in fold_images.values()):
        raise ValueError("ground truth must provide non-empty folds 0, 1 and 2")
    if len(set().union(*fold_images.values())) != sum(map(len, fold_images.values())):
        raise ValueError("fold image ids overlap")

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.gt)
    pred = _load_predictions_compat(args.pred)
    thresholds = build_threshold_grid(
        args.threshold_start, args.threshold_stop, args.threshold_step
    )
    curves = {}
    for held_out in (0, 1, 2):
        train_ids = set().union(
            *(image_ids for fold, image_ids in fold_images.items() if fold != held_out)
        )
        curves[held_out] = build_hierarchical_curves(
            gt,
            pred,
            image_ids=train_ids,
            thresholds=thresholds,
            protocol=protocol,
        )

    results: dict[str, Any] = {}
    for target_fdr in args.fdr_levels:
        global_oof: dict[int, list[dict[str, Any]]] = {}
        hierarchical_oof: dict[int, list[dict[str, Any]]] = {}
        fold_audit: dict[str, Any] = {}
        for held_out in (0, 1, 2):
            fit = fit_hierarchical_thresholds(
                curves[held_out],
                protocol=protocol,
                target_fdr=float(target_fdr),
                prior_strength=args.prior_strength,
                minimum_evidence=args.minimum_evidence,
            )
            held_pred = _scoped(pred, fold_images[held_out])
            global_oof.update(_filter_global(held_pred, float(fit["global_threshold"])))
            hierarchical_oof.update(
                filter_by_thresholds(held_pred, dict(fit["fine_thresholds"]))
            )
            fold_gt = _scoped(gt, fold_images[held_out])
            fold_audit[str(held_out)] = {
                "train_image_count": sum(
                    len(fold_images[fold]) for fold in (0, 1, 2) if fold != held_out
                ),
                "held_out_image_count": len(fold_images[held_out]),
                "global_threshold": fit["global_threshold"],
                "coarse_thresholds": fit["coarse_thresholds"],
                "fine_thresholds": {
                    str(key): value for key, value in fit["fine_thresholds"].items()
                },
                "fine_audit": {
                    str(key): value for key, value in fit["fine_audit"].items()
                },
                "held_out_global": _metric_payload(
                    fold_gt,
                    _scoped(global_oof, fold_images[held_out]),
                    protocol,
                    complete=False,
                ),
                "held_out_hierarchical": _metric_payload(
                    fold_gt,
                    _scoped(hierarchical_oof, fold_images[held_out]),
                    protocol,
                    complete=False,
                ),
            }
        global_metrics = _metric_payload(gt, global_oof, protocol)
        hierarchical_metrics = _metric_payload(gt, hierarchical_oof, protocol)
        results[f"{target_fdr:.3f}"] = {
            "global_crossfit": global_metrics,
            "hierarchical_crossfit": hierarchical_metrics,
            "delta": {
                "recall": hierarchical_metrics["recall"] - global_metrics["recall"],
                "fdr": hierarchical_metrics["fdr"] - global_metrics["fdr"],
                "official_macro_recall": (
                    hierarchical_metrics["official_macro_recall"]
                    - global_metrics["official_macro_recall"]
                ),
                "official_macro_fdr": (
                    hierarchical_metrics["official_macro_fdr"]
                    - global_metrics["official_macro_fdr"]
                ),
            },
            "folds": fold_audit,
        }

    key = f"{min(args.fdr_levels, key=lambda value: abs(value - 0.15)):.3f}"
    gate = results[key]
    candidate = gate["hierarchical_crossfit"]
    baseline = gate["global_crossfit"]
    coarse_drop = max(
        baseline["per_coarse"][name]["macro_recall"]
        - candidate["per_coarse"][name]["macro_recall"]
        for name in protocol.class_names
    )
    positive = (
        candidate["fdr"] <= float(key) + 0.002
        and coarse_drop <= 0.005
        and (
            candidate["recall"] - baseline["recall"] >= 0.003
            or candidate["official_macro_recall"]
            - baseline["official_macro_recall"]
            >= 0.003
        )
    )
    payload = {
        "status": "complete",
        "protocol": "fixed_benchmark_hierarchical_threshold_crossfit_v1",
        "benchmark_name": args.benchmark_name,
        "selection_policy": "two_folds_fit_one_fold_evaluate_no_heldout_labels",
        "threshold_grid": {
            "start": args.threshold_start,
            "stop": args.threshold_stop,
            "step": args.threshold_step,
        },
        "hierarchical_prior": {
            "prior_strength": args.prior_strength,
            "minimum_evidence": args.minimum_evidence,
            "space": "logit",
            "anchor": "coarse_class_threshold",
        },
        "fold_image_counts": {str(key): len(value) for key, value in fold_images.items()},
        "candidate_floor": _metric_payload(gt, pred, protocol),
        "frontiers": results,
        "admission": {
            "selected_fdr_level": float(key),
            "max_coarse_macro_recall_drop": coarse_drop,
            "passed_this_benchmark": positive,
            "note": "Formal admission requires the same direction on both fixed benchmarks.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["admission"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
