#!/usr/bin/env python3
"""比赛 Recall/FDR 评估入口。"""

import argparse
import json
import sys
from pathlib import Path

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import (
    OverallMetrics,
    RankingMetrics,
    evaluate_predictions,
    evaluate_ranking_metrics,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
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


def _format_metrics(
    result: OverallMetrics,
    ranking: RankingMetrics,
    *,
    recall_min: float,
    fdr_max: float,
) -> list[str]:
    """生成简洁的指标输出（pooled 门槛 + 官方排名口径）。"""
    recall_status = "PASS" if result.recall >= recall_min else "FAIL"
    fdr_status = "PASS" if result.fdr <= fdr_max else "FAIL"
    lines = [
        f"Overall Recall: {result.recall:.4f} [{recall_status}]",
        f"Overall FDR:    {result.fdr:.4f} [{fdr_status}]",
    ]
    for name, metrics in result.per_class.items():
        lines.append(
            f"{name}: Recall={metrics.recall:.4f}, FDR={metrics.fdr:.4f}, "
            f"TP={metrics.tp}, FP={metrics.fp}, FN={metrics.fn}"
        )
    lines.append(
        f"官方排名口径（细类平均）: Overall Recall={ranking.overall_recall:.4f}, "
        f"Overall FDR={ranking.overall_fdr:.4f}"
    )
    for name, coarse in ranking.per_coarse.items():
        lines.append(
            f"{name} macro: Recall={coarse.macro_recall:.4f}, FDR={coarse.macro_fdr:.4f} "
            f"(pooled Recall={coarse.pooled_recall:.4f}, FDR={coarse.pooled_fdr:.4f}, "
            f"{coarse.fine_count} 细类)"
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
        protocol = parse_evaluation_protocol(
            project_config,
            class_names_override=args.class_names,
        )
        gt_boxes = load_coco_ground_truth(args.gt)
        pred_boxes = load_coco_predictions(args.pred)
        result = evaluate_predictions(
            gt_boxes,
            pred_boxes,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
        )
        ranking = evaluate_ranking_metrics(
            gt_boxes,
            pred_boxes,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("评估输入无效: %s", error)
        return 1

    for line in _format_metrics(
        result,
        ranking,
        recall_min=protocol.recall_min,
        fdr_max=protocol.fdr_max,
    ):
        logger.info(line)

    if args.output:
        output_data = {
            "protocol_versions": {
                "contract_version": protocol.contract_version,
                "eval_version": protocol.eval_version,
            },
            "overall_recall": result.recall,
            "overall_fdr": result.fdr,
            "detection_gate": {
                "recall_min": protocol.recall_min,
                "fdr_max": protocol.fdr_max,
                "passed": (
                    result.recall >= protocol.recall_min
                    and result.fdr <= protocol.fdr_max
                ),
            },
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
            "official_ranking": {
                "overall_recall": ranking.overall_recall,
                "overall_fdr": ranking.overall_fdr,
                "per_coarse": {
                    name: {
                        "macro_recall": coarse.macro_recall,
                        "macro_fdr": coarse.macro_fdr,
                        "pooled_recall": coarse.pooled_recall,
                        "pooled_fdr": coarse.pooled_fdr,
                        "fine_count": coarse.fine_count,
                        "fine_ids": coarse.fine_ids,
                    }
                    for name, coarse in ranking.per_coarse.items()
                },
                "per_fine": {
                    str(category_id): {
                        "coarse_class": fine.coarse_class,
                        "recall": fine.recall,
                        "fdr": fine.fdr,
                        "tp": fine.tp,
                        "fp": fine.fp,
                        "fn": fine.fn,
                    }
                    for category_id, fine in sorted(ranking.per_fine.items())
                },
                "details": ranking.details,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=2, ensure_ascii=False)
        logger.info("评估结果已保存: %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
