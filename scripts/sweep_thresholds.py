#!/usr/bin/env python3
"""扫描全局置信度阈值并导出三个固定工作点。"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.protocol import EvaluationProtocol, parse_evaluation_protocol
from rsdet.postprocess.calibration import (
    OperatingPointSelection,
    ThresholdSweepPoint,
    build_threshold_grid,
    select_operating_points,
    sweep_global_thresholds,
)
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="sweep_thresholds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="按官方评测规则扫描全局 score 阈值")
    parser.add_argument("--gt", type=Path, required=True, help="COCO ground truth JSON")
    parser.add_argument("--pred", type=Path, required=True, help="COCO detection results JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="扫描产物目录")
    parser.add_argument(
        "--project-config",
        type=Path,
        default=Path("configs/project.yaml"),
        help="冻结评估协议所在的项目配置",
    )
    parser.add_argument("--threshold-start", type=float, default=0.0)
    parser.add_argument("--threshold-stop", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--internal-recall-min", type=float, default=0.88)
    parser.add_argument("--internal-fdr-max", type=float, default=0.17)
    parser.add_argument(
        "--allow-partial-taxonomy",
        action="store_true",
        help="仅用于折内/子集诊断；允许 GT 未覆盖全部 25 细类",
    )
    parser.add_argument(
        "--threshold-stage",
        "--prediction-stage",
        dest="threshold_stage",
        choices=("pre_fusion", "post_fusion"),
        default="post_fusion",
        help="阈值应用阶段；正式结果默认在跨 tile 融合后扫描",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_versions(protocol: EvaluationProtocol) -> dict[str, str]:
    return {
        "contract_version": protocol.contract_version,
        "eval_version": protocol.eval_version,
        "ranking_version": protocol.ranking_version,
    }


def _metrics_payload(point: ThresholdSweepPoint) -> dict[str, Any]:
    return {
        "threshold": point.threshold,
        "detections_kept": point.detections_kept,
        "overall_recall": point.metrics.recall,
        "overall_fdr": point.metrics.fdr,
        "overall_macro_recall": point.ranking_metrics.overall_recall,
        "overall_macro_fdr": point.ranking_metrics.overall_fdr,
        "details": point.metrics.details,
        "per_class": {
            name: {
                "recall": metrics.recall,
                "fdr": metrics.fdr,
                "tp": metrics.tp,
                "fp": metrics.fp,
                "fn": metrics.fn,
            }
            for name, metrics in point.metrics.per_class.items()
        },
        "official_ranking": {
            "details": point.ranking_metrics.details,
            "per_coarse": {
                name: {
                    "macro_recall": item.macro_recall,
                    "macro_fdr": item.macro_fdr,
                    "fine_count": item.fine_count,
                    "fine_ids": item.fine_ids,
                }
                for name, item in point.ranking_metrics.per_coarse.items()
            },
        },
    }


def _selection_payload(selection: OperatingPointSelection) -> dict[str, Any]:
    return {
        "policy": selection.policy,
        "passed": selection.passed,
        **_metrics_payload(selection.point),
    }


def _write_sweep_csv(
    path: Path,
    points: list[ThresholdSweepPoint],
    *,
    protocol: EvaluationProtocol,
    threshold_stage: str,
) -> None:
    class_fields = [
        f"{class_name}_{suffix}"
        for class_name in protocol.class_names
        for suffix in ("recall", "fdr", "tp", "fp", "fn")
    ]
    fieldnames = [
        "contract_version",
        "eval_version",
        "ranking_version",
        "threshold_stage",
        "threshold",
        "detections_kept",
        "overall_recall",
        "overall_fdr",
        "overall_macro_recall",
        "overall_macro_fdr",
        "tp",
        "fp",
        "fn",
        *class_fields,
        "official_gate_passed",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            row: dict[str, Any] = {
                "contract_version": protocol.contract_version,
                "eval_version": protocol.eval_version,
                "ranking_version": protocol.ranking_version,
                "threshold_stage": threshold_stage,
                "threshold": f"{point.threshold:.12g}",
                "detections_kept": point.detections_kept,
                "overall_recall": point.metrics.recall,
                "overall_fdr": point.metrics.fdr,
                "overall_macro_recall": point.ranking_metrics.overall_recall,
                "overall_macro_fdr": point.ranking_metrics.overall_fdr,
                "tp": point.metrics.details["tp"],
                "fp": point.metrics.details["fp"],
                "fn": point.metrics.details["fn"],
                "official_gate_passed": (
                    point.metrics.recall >= protocol.recall_min
                    and point.metrics.fdr <= protocol.fdr_max
                ),
            }
            for class_name, metrics in point.metrics.per_class.items():
                row.update(
                    {
                        f"{class_name}_recall": metrics.recall,
                        f"{class_name}_fdr": metrics.fdr,
                        f"{class_name}_tp": metrics.tp,
                        f"{class_name}_fp": metrics.fp,
                        f"{class_name}_fn": metrics.fn,
                    }
                )
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    """运行阈值扫描。"""
    args = parse_args(argv)
    for path, label in (
        (args.gt, "GT"),
        (args.pred, "预测"),
        (args.project_config, "项目配置"),
    ):
        if not path.exists():
            logger.error("%s文件不存在: %s", label, path)
            return 1

    try:
        protocol = parse_evaluation_protocol(load_config(args.project_config))
        gt_boxes = load_coco_ground_truth(args.gt)
        if sum(len(items) for items in gt_boxes.values()) == 0:
            raise ValueError("GT 不包含任何标注，不能选择正式阈值")
        pred_boxes = load_coco_predictions(args.pred)
        thresholds = build_threshold_grid(
            args.threshold_start,
            args.threshold_stop,
            args.threshold_step,
        )
        points = sweep_global_thresholds(
            gt_boxes,
            pred_boxes,
            thresholds,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
            require_complete_taxonomy=not args.allow_partial_taxonomy,
        )
        selections = select_operating_points(
            points,
            official_recall_min=protocol.recall_min,
            official_fdr_max=protocol.fdr_max,
            internal_recall_min=args.internal_recall_min,
            internal_fdr_max=args.internal_fdr_max,
        )
        source = {
            "gt": str(args.gt),
            "gt_sha256": _sha256(args.gt),
            "pred": str(args.pred),
            "pred_sha256": _sha256(args.pred),
            "threshold_stage": args.threshold_stage,
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("阈值扫描输入无效: %s", error)
        return 1

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / "threshold_sweep.csv"
        selected_path = args.output_dir / "selected_thresholds.yaml"
        metrics_path = args.output_dir / "metrics_at_selected_thresholds.json"

        _write_sweep_csv(
            csv_path,
            points,
            protocol=protocol,
            threshold_stage=args.threshold_stage,
        )
        selected_data = {
            "protocol_versions": _protocol_versions(protocol),
            "source": source,
            "targets": {
                "official": {
                    "recall_min": protocol.recall_min,
                    "fdr_max": protocol.fdr_max,
                },
                "internal": {
                    "recall_min": args.internal_recall_min,
                    "fdr_max": args.internal_fdr_max,
                },
            },
            "workpoints": {
                name: _selection_payload(selection) for name, selection in selections.items()
            },
        }
        with selected_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(selected_data, file, allow_unicode=True, sort_keys=False)

        metrics_data = {
            "protocol_versions": _protocol_versions(protocol),
            "source": source,
            "workpoints": {
                name: _selection_payload(selection) for name, selection in selections.items()
            },
        }
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics_data, file, indent=2, ensure_ascii=False)
    except OSError as error:
        logger.error("阈值扫描结果写入失败: %s", error)
        return 1

    for name, selection in selections.items():
        point = selection.point
        logger.info(
            "%s: threshold=%.4f Recall=%.4f FDR=%.4f passed=%s",
            name,
            point.threshold,
            point.metrics.recall,
            point.metrics.fdr,
            selection.passed,
        )
    logger.info("阈值扫描结果已保存: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
