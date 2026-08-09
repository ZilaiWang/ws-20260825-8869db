#!/usr/bin/env python3
"""Search per-coarse-class score thresholds against the official Recall/FDR gate."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from rsdet.evaluation.coco_metric import (
    evaluate_and_write,
    load_coco_ground_truth,
    load_coco_predictions,
)
from rsdet.evaluation.official_metric import evaluate_predictions
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="tune_thresholds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="搜索三大类置信度阈值")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument(
        "--project-config", type=Path, default=Path("configs/project.yaml")
    )
    parser.add_argument("--min-threshold", type=float, default=0.03)
    parser.add_argument("--max-threshold", type=float, default=0.50)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument(
        "--matching-policy",
        choices=("fine", "coarse"),
        default="fine",
        help="coarse 仅用于估计定位/大类正确时的细类重识别理论上限",
    )
    parser.add_argument("--output", type=Path, required=True, help="搜索摘要 JSON")
    parser.add_argument("--best-pred", type=Path, default=None)
    parser.add_argument("--best-metrics", type=Path, default=None)
    return parser.parse_args(argv)


def _threshold_grid(minimum: float, maximum: float, step: float) -> list[float]:
    if not 0.0 <= minimum <= maximum <= 1.0 or step <= 0.0:
        raise ValueError("阈值范围必须满足 0 <= min <= max <= 1 且 step > 0")
    count = int(round((maximum - minimum) / step))
    values = [round(minimum + index * step, 6) for index in range(count + 1)]
    if values[-1] < maximum:
        values.append(round(maximum, 6))
    return values


def _coarse_curves(
    gt: dict[int, list[dict[str, Any]]],
    predictions: dict[int, list[dict[str, Any]]],
    *,
    category_mapping: Mapping[int, str],
    iou_thresholds: Mapping[str, float],
    thresholds: list[float],
    matching_policy: str = "fine",
) -> dict[str, list[dict[str, Any]]]:
    curves: dict[str, list[dict[str, Any]]] = {}
    for coarse_name in ("ship", "aircraft", "vehicle"):
        category_ids = {
            category_id
            for category_id, mapped_name in category_mapping.items()
            if mapped_name == coarse_name
        }
        class_mapping = {category_id: coarse_name for category_id in category_ids}
        class_gt = {
            image_id: [
                item for item in items if int(item["category_id"]) in category_ids
            ]
            for image_id, items in gt.items()
        }
        class_predictions = {
            image_id: [
                item
                for item in items
                if int(item["category_id"]) in category_ids
            ]
            for image_id, items in predictions.items()
        }
        if matching_policy == "coarse":
            representative = min(category_ids)
            class_mapping = {representative: coarse_name}
            class_gt = {
                image_id: [dict(item, category_id=representative) for item in items]
                for image_id, items in class_gt.items()
            }
            class_predictions = {
                image_id: [dict(item, category_id=representative) for item in items]
                for image_id, items in class_predictions.items()
            }
        points: list[dict[str, Any]] = []
        for threshold in thresholds:
            filtered = {
                image_id: [
                    item
                    for item in items
                    if float(item["score"]) >= threshold
                ]
                for image_id, items in class_predictions.items()
            }
            result = evaluate_predictions(
                class_gt,
                filtered,
                class_names=[coarse_name],
                category_mapping=class_mapping,
                iou_thresholds={coarse_name: float(iou_thresholds[coarse_name])},
            )
            metrics = result.per_class[coarse_name]
            points.append(
                {
                    "threshold": threshold,
                    "tp": metrics.tp,
                    "fp": metrics.fp,
                    "fn": metrics.fn,
                    "recall": metrics.recall,
                    "fdr": metrics.fdr,
                }
            )
        curves[coarse_name] = points
    return curves


def _combined_candidate(points: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    names = ("ship", "aircraft", "vehicle")
    tp = sum(int(point["tp"]) for point in points)
    fp = sum(int(point["fp"]) for point in points)
    fn = sum(int(point["fn"]) for point in points)
    return {
        "thresholds": {
            name: float(point["threshold"]) for name, point in zip(names, points)
        },
        "overall_recall": tp / (tp + fn) if tp + fn else 1.0,
        "overall_fdr": fp / (tp + fp) if tp + fp else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_class": {name: point for name, point in zip(names, points)},
    }


def _filter_coco_predictions(
    source: Path,
    target: Path,
    *,
    thresholds: Mapping[str, float],
    category_mapping: Mapping[int, str],
) -> None:
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise ValueError("阈值搜索当前要求预测文件为 COCO detection 列表")
    filtered = [
        item
        for item in document
        if float(item["score"])
        >= float(thresholds[category_mapping[int(item["category_id"])]])
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        project = load_config(args.project_config)
        task = project["task"]
        official = project["official_evaluation"]
        category_mapping = {
            int(category_id): str(coarse)
            for category_id, coarse in task["dataset_category_mapping"].items()
        }
        iou_thresholds = {
            str(name): float(value)
            for name, value in official["iou_thresholds"].items()
        }
        threshold_grid = _threshold_grid(
            args.min_threshold, args.max_threshold, args.step
        )
        curves = _coarse_curves(
            load_coco_ground_truth(args.gt),
            load_coco_predictions(args.pred),
            category_mapping=category_mapping,
            iou_thresholds=iou_thresholds,
            thresholds=threshold_grid,
            matching_policy=args.matching_policy,
        )
        fdr_max = float(official["fdr_max"])
        candidates = (
            _combined_candidate(points)
            for points in itertools.product(
                curves["ship"], curves["aircraft"], curves["vehicle"]
            )
        )
        feasible = [item for item in candidates if item["overall_fdr"] <= fdr_max]
        if not feasible:
            raise RuntimeError("搜索范围内没有满足 FDR 门槛的阈值组合")
        feasible.sort(
            key=lambda item: (
                -item["overall_recall"],
                item["overall_fdr"],
                sum(item["thresholds"].values()),
            )
        )
        best = feasible[0]
        summary = {
            "source_prediction": str(args.pred),
            "search": {
                "min_threshold": args.min_threshold,
                "max_threshold": args.max_threshold,
                "step": args.step,
                "fdr_max": fdr_max,
                "combinations": len(threshold_grid) ** 3,
                "matching_policy": args.matching_policy,
            },
            "best": best,
            "top_candidates": feasible[:20],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if args.best_pred is not None:
            if args.matching_policy != "fine":
                raise ValueError("coarse 上限分析不能导出正式 best prediction")
            _filter_coco_predictions(
                args.pred,
                args.best_pred,
                thresholds=best["thresholds"],
                category_mapping=category_mapping,
            )
            if args.best_metrics is not None:
                evaluate_and_write(
                    args.gt,
                    args.best_pred,
                    args.best_metrics,
                    args.project_config,
                )
        logger.info(
            "最佳阈值 %s: Recall=%.4f, FDR=%.4f",
            best["thresholds"],
            best["overall_recall"],
            best["overall_fdr"],
        )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        logger.error("阈值搜索失败: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
