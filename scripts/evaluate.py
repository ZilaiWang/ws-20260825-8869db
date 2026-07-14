#!/usr/bin/env python3
"""比赛 Recall/FDR 评估入口。"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rsdet.evaluation.official_metric import OverallMetrics, evaluate_predictions
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="按比赛规则计算 Recall 和 FDR")
    parser.add_argument("--gt", type=Path, required=True, help="COCO ground truth JSON")
    parser.add_argument("--pred", type=Path, required=True, help="COCO detection results JSON")
    parser.add_argument(
        "--project-config",
        type=Path,
        default=Path("configs/project.yaml"),
        help="包含 25 细类到三大类映射的项目配置",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="覆盖评估大类，默认读取 project.yaml",
    )
    parser.add_argument("--output", type=Path, default=None, help="评估结果 JSON")
    return parser.parse_args(argv)


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    """把 COCO xywh 转成 xyxy。"""
    if len(box) != 4 or box[2] < 0 or box[3] < 0:
        raise ValueError(f"非法 COCO bbox: {box}")
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def _load_json(path: Path) -> Any:
    """读取 JSON 文件。"""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_ground_truth(path: Path) -> dict[int, list[dict[str, Any]]]:
    """读取标准 COCO ground truth。"""
    data = _load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("annotations"), list):
        raise ValueError("GT 必须是包含 annotations 列表的 COCO JSON 对象")
    return _group_annotations(data["annotations"], require_score=False)


def _load_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    """读取标准 COCO detection list，也兼容含 annotations 的对象。"""
    data = _load_json(path)
    if isinstance(data, list):
        annotations = data
    elif isinstance(data, dict) and isinstance(data.get("annotations"), list):
        annotations = data["annotations"]
    else:
        raise ValueError("预测文件必须是 COCO detection 列表或包含 annotations 的对象")
    return _group_annotations(annotations, require_score=True)


def _group_annotations(
    annotations: list[dict[str, Any]],
    *,
    require_score: bool,
) -> dict[int, list[dict[str, Any]]]:
    """把 COCO 记录按 image_id 分组。"""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        record: dict[str, Any] = {
            "bbox_xyxy": _xywh_to_xyxy([float(value) for value in annotation["bbox"]]),
            "category_id": int(annotation["category_id"]),
        }
        if require_score:
            if "score" not in annotation:
                raise ValueError("预测记录缺少 score")
            record["score"] = float(annotation["score"])
        grouped.setdefault(image_id, []).append(record)
    return grouped


def _format_metrics(result: OverallMetrics) -> list[str]:
    """生成简洁的指标输出。"""
    recall_status = "PASS" if result.recall >= 0.85 else "FAIL"
    fdr_status = "PASS" if result.fdr <= 0.20 else "FAIL"
    lines = [
        f"Overall Recall: {result.recall:.4f} [{recall_status}]",
        f"Overall FDR:    {result.fdr:.4f} [{fdr_status}]",
    ]
    for name, metrics in result.per_class.items():
        lines.append(
            f"{name}: Recall={metrics.recall:.4f}, FDR={metrics.fdr:.4f}, "
            f"TP={metrics.tp}, FP={metrics.fp}, FN={metrics.fn}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    """运行评估。"""
    args = parse_args(argv)
    for path, label in ((args.gt, "GT"), (args.pred, "预测"), (args.project_config, "项目配置")):
        if not path.exists():
            logger.error("%s文件不存在: %s", label, path)
            return 1

    try:
        project_config = load_config(args.project_config)
        task_config = project_config["task"]
        class_names = args.class_names or task_config["class_names"]
        category_mapping = {
            int(category_id): class_name
            for category_id, class_name in task_config["dataset_category_mapping"].items()
        }
        gt_boxes = _load_ground_truth(args.gt)
        pred_boxes = _load_predictions(args.pred)
        result = evaluate_predictions(
            gt_boxes,
            pred_boxes,
            class_names=class_names,
            category_mapping=category_mapping,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("评估输入无效: %s", error)
        return 1

    for line in _format_metrics(result):
        logger.info(line)

    if args.output:
        output_data = {
            "overall_recall": result.recall,
            "overall_fdr": result.fdr,
            "details": result.details,
            "per_class": {
                name: {
                    "recall": metrics.recall,
                    "fdr": metrics.fdr,
                    "tp": metrics.tp,
                    "fp": metrics.fp,
                    "fn": metrics.fn,
                }
                for name, metrics in result.per_class.items()
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=2, ensure_ascii=False)
        logger.info("评估结果已保存: %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
