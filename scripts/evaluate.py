#!/usr/bin/env python3
"""评估入口脚本。

用法:
    python scripts/evaluate.py --gt gt.json --pred pred.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from rsdet.evaluation.official_metric import evaluate_predictions, OverallMetrics
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="evaluate")


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="官方指标评估",
    )
    parser.add_argument("--gt", type=Path, required=True, help="GT JSON 文件路径")
    parser.add_argument("--pred", type=Path, required=True, help="预测 JSON 文件路径")
    parser.add_argument("--class-names", type=str, nargs="*", default=None,
                        help="类别名列表，如 ship aircraft vehicle")
    parser.add_argument("--output", type=Path, default=None, help="评估结果输出路径")
    return parser.parse_args(argv)


def _load_coco_like(path: Path):
    """加载类 COCO JSON 并转换为评估所需格式。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_boxes = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        bbox_xywh = ann["bbox"]
        bbox_xyxy = [
            bbox_xywh[0],
            bbox_xywh[1],
            bbox_xywh[0] + bbox_xywh[2],
            bbox_xywh[1] + bbox_xywh[3],
        ]
        if img_id not in gt_boxes:
            gt_boxes[img_id] = []
        gt_boxes[img_id].append({
            "bbox_xyxy": bbox_xyxy,
            "category_id": ann["category_id"],
        })

    pred_boxes = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        bbox_xywh = ann["bbox"]
        bbox_xyxy = [
            bbox_xywh[0],
            bbox_xywh[1],
            bbox_xywh[0] + bbox_xywh[2],
            bbox_xywh[1] + bbox_xywh[3],
        ]
        if img_id not in pred_boxes:
            pred_boxes[img_id] = []
        pred_boxes[img_id].append({
            "bbox_xyxy": bbox_xyxy,
            "score": ann.get("score", 1.0),
            "category_id": ann["category_id"],
        })

    return gt_boxes, pred_boxes


def main(argv: list | None = None) -> int:
    args = parse_args(argv)

    gt_path = Path(args.gt)
    pred_path = Path(args.pred)

    if not gt_path.exists():
        logger.error(f"GT 文件不存在: {gt_path}")
        return 1
    if not pred_path.exists():
        logger.error(f"预测文件不存在: {pred_path}")
        return 1

    gt_boxes, pred_boxes = _load_coco_like(gt_path)
    # 预测文件是真正的预测，GT不变
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    # 重建预测框（与 _load_coco_like 区分GT和pred）
    pred_boxes_only = {}
    for ann in pred_data.get("annotations", []):
        img_id = ann["image_id"]
        bbox_xywh = ann["bbox"]
        bbox_xyxy = [
            bbox_xywh[0],
            bbox_xywh[1],
            bbox_xywh[0] + bbox_xywh[2],
            bbox_xywh[1] + bbox_xywh[3],
        ]
        if img_id not in pred_boxes_only:
            pred_boxes_only[img_id] = []
        pred_boxes_only[img_id].append({
            "bbox_xyxy": bbox_xyxy,
            "score": ann.get("score", 1.0),
            "category_id": ann["category_id"],
        })

    class_names = args.class_names or ["ship", "aircraft", "vehicle"]
    result: OverallMetrics = evaluate_predictions(gt_boxes, pred_boxes_only, class_names)

    print(f"\n{'='*50}")
    print(f"Overall Recall: {result.recall:.4f}  {'✅' if result.recall >= 0.85 else '❌'}")
    print(f"Overall FDR:    {result.fdr:.4f}  {'✅' if result.fdr <= 0.20 else '❌'}")
    print(f"{'='*50}")
    for cname, m in result.per_class.items():
        print(f"  {cname}: Recall={m.recall:.4f}, FDR={m.fdr:.4f}, TP={m.tp}, FP={m.fp}, FN={m.fn}")
    print(f"{'='*50}\n")

    if args.output:
        output_data = {
            "overall_recall": result.recall,
            "overall_fdr": result.fdr,
            "per_class": {
                name: {"recall": m.recall, "fdr": m.fdr, "tp": m.tp, "fp": m.fp, "fn": m.fn}
                for name, m in result.per_class.items()
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"评估结果已保存: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
