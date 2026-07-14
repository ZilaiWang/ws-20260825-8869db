"""官方评估指标实现。

基于比赛规则：
- 总体 Recall >= 0.85, FDR <= 0.20
- ship/aircraft IoU = 0.50, vehicle IoU = 0.35
- 预测按 score 降序，greedy matching
- 每个 GT 只匹配一次，重复框计为 FP
- FDR = FP / (FP + TP)
- 当前默认采用 class-aware matching（类别预测错误的处理见 OPEN_QUESTIONS.md）
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

IOU_THRESHOLDS = {
    "ship": 0.50,
    "aircraft": 0.50,
    "vehicle": 0.35,
}


@dataclass
class PerClassMetrics:
    """单类别评估结果。"""
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        if denominator == 0:
            return 1.0  # 无 GT，召回率定义为 1
        return self.tp / denominator

    @property
    def fdr(self) -> float:
        denominator = self.fp + self.tp
        if denominator == 0:
            return 0.0  # 无预测，FDR 为 0（策略：保守处理，详见 OPEN_QUESTIONS.md）
        return self.fp / denominator


@dataclass
class OverallMetrics:
    """总体评估结果。"""
    recall: float = 0.0
    fdr: float = 0.0
    per_class: Dict[str, PerClassMetrics] = field(default_factory=dict)
    details: Dict[str, any] = field(default_factory=dict)


def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """计算两个 xyxy bbox 的 IoU。

    Args:
        box_a: [x1, y1, x2, y2]。
        box_b: [x1, y1, x2, y2]。

    Returns:
        IoU 值 [0, 1]。
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def evaluate_predictions(
    gt_boxes: Dict[int, List[dict]],
    pred_boxes: Dict[int, List[dict]],
    class_names: Optional[List[str]] = None,
) -> OverallMetrics:
    """按官方规则评估预测。

    class-aware matching：预测类别必须正确才能匹配（见 OPEN_QUESTIONS.md）。

    Args:
        gt_boxes: {image_id: [{bbox_xyxy, category_id}, ...]}。
        pred_boxes: {image_id: [{bbox_xyxy, score, category_id}, ...]}。
        class_names: 类别名称列表，用于按类别统计。

    Returns:
        OverallMetrics 包含总体和各类别指标。
    """
    all_tp, all_fp, all_fn = 0, 0, 0
    per_class = {}

    # 确定类别集合
    class_ids = set()
    for img_gts in gt_boxes.values():
        for g in img_gts:
            class_ids.add(g["category_id"])
    for img_preds in pred_boxes.values():
        for p in img_preds:
            class_ids.add(p["category_id"])

    # 按类别评估
    for cid in sorted(class_ids):
        tp_c, fp_c, fn_c = _evaluate_per_class(gt_boxes, pred_boxes, target_cid=cid)
        cname = class_names[cid] if class_names and cid < len(class_names) else str(cid)
        per_class[cname] = PerClassMetrics(tp=tp_c, fp=fp_c, fn=fn_c)
        all_tp += tp_c
        all_fp += fp_c
        all_fn += fn_c

    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 1.0
    overall_fdr = all_fp / (all_fp + all_tp) if (all_fp + all_tp) > 0 else 0.0

    return OverallMetrics(
        recall=overall_recall,
        fdr=overall_fdr,
        per_class=per_class,
        details={"total_gt": all_tp + all_fn, "total_pred": all_tp + all_fp},
    )


def _evaluate_per_class(
    gt_boxes: Dict[int, List[dict]],
    pred_boxes: Dict[int, List[dict]],
    target_cid: int,
) -> Tuple[int, int, int]:
    """对单个类别，在所有图像上执行 class-aware greedy matching。

    评估假设：类别错误的预测计为 FP（class-aware matching）。
    此假设写入函数文档和实验报告。
    """
    tp, fp, fn = 0, 0, 0

    iou_thresh = IOU_THRESHOLDS.get(
        _cid_to_name(target_cid), 0.50
    )

    for image_id in set(list(gt_boxes.keys()) + list(pred_boxes.keys())):
        # 收集该类别的 GT
        gts = [
            (i, g["bbox_xyxy"])
            for i, g in enumerate(gt_boxes.get(image_id, []))
            if g.get("category_id") == target_cid
        ]

        # 收集该类别的预测（按 score 降序）
        preds = [
            (p["score"], p["bbox_xyxy"], p.get("category_id"))
            for p in pred_boxes.get(image_id, [])
            if p.get("category_id") == target_cid
        ]
        preds.sort(key=lambda x: x[0], reverse=True)

        gt_matched = [False] * len(gts)

        for score, pbox, pcid in preds:
            matched = False
            best_iou = 0.0
            best_gt_idx = -1

            for j, (gt_idx, gtbox) in enumerate(gts):
                if gt_matched[j]:
                    continue
                iou = _compute_iou(pbox, gtbox)
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                tp += 1
                matched = True
            else:
                fp += 1

        fn += sum(1 for m in gt_matched if not m)

    return tp, fp, fn


def _cid_to_name(cid: int) -> str:
    """category_id 到类别名的映射（临时），待数据集审计后更新。"""
    mapping = {0: "ship", 1: "aircraft", 2: "vehicle"}
    return mapping.get(cid, "unknown")
