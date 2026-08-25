#!/usr/bin/env python3
"""Calibrate 25 fine-class thresholds under the global official FDR constraint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from rsdet.data.xh_dataset import FINE_NAMES, coarse_name
from rsdet.evaluation.coco_metric import (
    evaluate_and_write,
    load_coco_ground_truth,
    load_coco_predictions,
)
from rsdet.evaluation.official_metric import evaluate_predictions
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="tune_fine_thresholds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="25 类自适应阈值校准")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--min-threshold", type=float, default=0.001)
    parser.add_argument("--max-threshold", type=float, default=0.50)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-pred", type=Path, required=True)
    parser.add_argument("--best-metrics", type=Path, required=True)
    return parser.parse_args(argv)


def _threshold_grid(minimum: float, maximum: float, step: float) -> list[float]:
    if not 0.0 <= minimum <= maximum <= 1.0 or step <= 0.0:
        raise ValueError("阈值范围必须满足 0 <= min <= max <= 1 且 step > 0")
    count = int(math.floor((maximum - minimum) / step + 1e-9))
    values = [round(minimum + index * step, 6) for index in range(count + 1)]
    if values[-1] < maximum:
        values.append(round(maximum, 6))
    return values


def _class_curve(
    category_id: int,
    gt: dict[int, list[dict[str, Any]]],
    predictions: dict[int, list[dict[str, Any]]],
    thresholds: list[float],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    group = coarse_name(category_id)
    class_gt = {
        image_id: [item for item in items if int(item["category_id"]) == category_id]
        for image_id, items in gt.items()
        if any(int(item["category_id"]) == category_id for item in items)
    }
    class_predictions = {
        image_id: [item for item in items if int(item["category_id"]) == category_id]
        for image_id, items in predictions.items()
        if any(int(item["category_id"]) == category_id for item in items)
    }
    unique: dict[tuple[int, int], dict[str, Any]] = {}
    for threshold in thresholds:
        filtered = {
            image_id: [item for item in items if float(item["score"]) >= threshold]
            for image_id, items in class_predictions.items()
        }
        result = evaluate_predictions(
            class_gt,
            filtered,
            class_names=[group],
            category_mapping={category_id: group},
            iou_thresholds={group: iou_threshold},
        )
        metrics = result.per_class[group]
        point = {
            "threshold": threshold,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "recall": metrics.recall,
            "fdr": metrics.fdr,
        }
        key = (metrics.tp, metrics.fp)
        previous = unique.get(key)
        if previous is None or threshold > float(previous["threshold"]):
            unique[key] = point
    return list(unique.values())


def _optimize(
    curves: list[list[dict[str, Any]]],
    *,
    total_gt: int,
    fdr_max: float,
) -> tuple[list[dict[str, Any]], int, int]:
    max_fp = math.floor(total_gt * fdr_max / max(1.0 - fdr_max, 1e-12))
    states: dict[int, tuple[int, list[dict[str, Any]]]] = {0: (0, [])}
    for curve in curves:
        updated: dict[int, tuple[int, list[dict[str, Any]]]] = {}
        for accumulated_fp, (accumulated_tp, path) in states.items():
            for point in curve:
                fp = accumulated_fp + int(point["fp"])
                if fp > max_fp:
                    continue
                tp = accumulated_tp + int(point["tp"])
                incumbent = updated.get(fp)
                if incumbent is None or tp > incumbent[0]:
                    updated[fp] = (tp, [*path, point])
        states = updated
    feasible = [
        (tp, fp, path)
        for fp, (tp, path) in states.items()
        if fp / (tp + fp) <= fdr_max
        if tp + fp > 0
    ]
    if not feasible:
        raise RuntimeError("搜索范围内没有满足全局 FDR 门槛的细类阈值组合")
    tp, fp, path = max(feasible, key=lambda item: (item[0], -item[1]))
    return path, tp, fp


def _write_filtered_predictions(
    source_path: Path,
    output_path: Path,
    thresholds: list[float],
) -> None:
    document = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise ValueError("预测文件必须是 COCO detection 列表")
    filtered = [
        item for item in document if float(item["score"]) >= thresholds[int(item["category_id"])]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        project = load_config(args.project_config)
        official = project["official_evaluation"]
        iou_thresholds = {
            str(name): float(value) for name, value in official["iou_thresholds"].items()
        }
        fdr_max = float(official["fdr_max"])
        gt = load_coco_ground_truth(args.gt)
        predictions = load_coco_predictions(args.pred)
        grid = _threshold_grid(args.min_threshold, args.max_threshold, args.step)
        curves = [
            _class_curve(
                category_id,
                gt,
                predictions,
                grid,
                iou_thresholds[coarse_name(category_id)],
            )
            for category_id in range(len(FINE_NAMES))
        ]
        total_gt = sum(len(items) for items in gt.values())
        selected, tp, fp = _optimize(curves, total_gt=total_gt, fdr_max=fdr_max)
        fn = total_gt - tp
        thresholds = [float(point["threshold"]) for point in selected]
        summary = {
            "method": "fine_class_adaptive_threshold_calibration",
            "source_prediction": str(args.pred),
            "overall_recall": tp / total_gt,
            "overall_fdr": fp / (tp + fp),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "fdr_max": fdr_max,
            "fine_score_thresholds": {
                name: threshold for name, threshold in zip(FINE_NAMES, thresholds)
            },
            "per_class": {name: point for name, point in zip(FINE_NAMES, selected)},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_filtered_predictions(args.pred, args.best_pred, thresholds)
        exact = evaluate_and_write(args.gt, args.best_pred, args.best_metrics, args.project_config)
        logger.info(
            "25 类校准后 Recall=%.4f, FDR=%.4f: %s",
            exact["overall_recall"],
            exact["overall_fdr"],
            args.output,
        )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        logger.error("细类阈值校准失败: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
