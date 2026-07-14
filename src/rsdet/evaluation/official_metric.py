"""比赛 Recall/FDR 评估。

规则：预测按分数降序贪心匹配；每个 GT 只匹配一次；重复框计为 FP。
评估按 ship、aircraft、vehicle 三大类进行。数据集的 25 个细类必须先通过
``category_mapping`` 归并，不能直接把细类 ID 当成三大类 ID。
"""

import math
from dataclasses import dataclass, field
from typing import Any

IOU_THRESHOLDS = {
    "ship": 0.50,
    "aircraft": 0.50,
    "vehicle": 0.35,
}


@dataclass
class PerClassMetrics:
    """单类 TP、FP、FN 及其派生指标。"""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def recall(self) -> float:
        """返回 Recall；没有 GT 时按 1.0 处理。"""
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 1.0

    @property
    def fdr(self) -> float:
        """返回 FDR；没有预测时按 0.0 处理。"""
        denominator = self.fp + self.tp
        return self.fp / denominator if denominator else 0.0


@dataclass
class OverallMetrics:
    """总体和分大类评估结果。"""

    recall: float = 0.0
    fdr: float = 0.0
    per_class: dict[str, PerClassMetrics] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """计算两个 xyxy 像素框的 IoU。"""
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("bbox 必须包含 4 个数值")

    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter_area = max(0.0, xb - xa) * max(0.0, yb - ya)

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def evaluate_predictions(
    gt_boxes: dict[int, list[dict[str, Any]]],
    pred_boxes: dict[int, list[dict[str, Any]]],
    class_names: list[str] | None = None,
    category_mapping: dict[int, str] | None = None,
) -> OverallMetrics:
    """按三大类计算比赛指标。

    Args:
        gt_boxes: ``{image_id: [{bbox_xyxy, category_id}, ...]}``。
        pred_boxes: ``{image_id: [{bbox_xyxy, score, category_id}, ...]}``。
        class_names: 参与评估的大类，默认 ship、aircraft、vehicle。
        category_mapping: 数据集 category_id 到大类名称的映射。省略时仅接受
            ``0=ship, 1=aircraft, 2=vehicle`` 的三类输出。

    Raises:
        ValueError: 类别映射缺失或包含未知大类。
    """
    names = class_names or list(IOU_THRESHOLDS)
    unknown_names = set(names) - set(IOU_THRESHOLDS)
    if unknown_names:
        raise ValueError(f"未知评估类别: {sorted(unknown_names)}")

    mapping = (
        {index: name for index, name in enumerate(names)}
        if category_mapping is None
        else category_mapping
    )
    if not mapping:
        raise ValueError("category_mapping 不能为空")
    normalized_mapping = {int(category_id): name for category_id, name in mapping.items()}
    unknown_targets = set(normalized_mapping.values()) - set(names)
    if unknown_targets:
        raise ValueError(f"类别映射包含未参与评估的大类: {sorted(unknown_targets)}")

    normalized_gt = _normalize_records(gt_boxes, normalized_mapping, require_score=False)
    normalized_pred = _normalize_records(pred_boxes, normalized_mapping, require_score=True)

    per_class: dict[str, PerClassMetrics] = {}
    total_tp = total_fp = total_fn = 0
    for class_name in names:
        tp, fp, fn = _evaluate_per_class(
            normalized_gt,
            normalized_pred,
            class_name,
            IOU_THRESHOLDS[class_name],
        )
        per_class[class_name] = PerClassMetrics(tp=tp, fp=fp, fn=fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    recall_denominator = total_tp + total_fn
    fdr_denominator = total_tp + total_fp
    return OverallMetrics(
        recall=total_tp / recall_denominator if recall_denominator else 1.0,
        fdr=total_fp / fdr_denominator if fdr_denominator else 0.0,
        per_class=per_class,
        details={
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "total_gt": recall_denominator,
            "total_pred": fdr_denominator,
            "empty_gt_recall_policy": 1.0,
            "empty_prediction_fdr_policy": 0.0,
        },
    )


def _normalize_records(
    records: dict[int, list[dict[str, Any]]],
    category_mapping: dict[int, str],
    *,
    require_score: bool,
) -> dict[int, list[dict[str, Any]]]:
    """校验记录并把细类 ID 归并为三大类名称。"""
    normalized: dict[int, list[dict[str, Any]]] = {}
    for image_id, items in records.items():
        normalized[image_id] = []
        for item in items:
            category_id = int(item["category_id"])
            if category_id not in category_mapping:
                raise ValueError(f"category_id={category_id} 缺少三大类映射")
            box = [float(value) for value in item["bbox_xyxy"]]
            if (
                len(box) != 4
                or not all(math.isfinite(value) for value in box)
                or box[2] < box[0]
                or box[3] < box[1]
            ):
                raise ValueError(f"非法 xyxy bbox: {box}")
            normalized_item: dict[str, Any] = {
                "bbox_xyxy": box,
                "class_name": category_mapping[category_id],
            }
            if require_score:
                score = float(item["score"])
                if not math.isfinite(score):
                    raise ValueError(f"非法 score: {score}")
                normalized_item["score"] = score
            normalized[image_id].append(normalized_item)
    return normalized


def _evaluate_per_class(
    gt_boxes: dict[int, list[dict[str, Any]]],
    pred_boxes: dict[int, list[dict[str, Any]]],
    class_name: str,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """对一个大类执行跨图像的降序贪心匹配。"""
    tp = fp = fn = 0
    for image_id in set(gt_boxes) | set(pred_boxes):
        gts = [
            item["bbox_xyxy"]
            for item in gt_boxes.get(image_id, [])
            if item["class_name"] == class_name
        ]
        predictions = [
            item for item in pred_boxes.get(image_id, []) if item["class_name"] == class_name
        ]
        predictions.sort(key=lambda item: item["score"], reverse=True)
        matched = [False] * len(gts)

        for prediction in predictions:
            best_index = -1
            best_iou = -1.0
            for index, gt_box in enumerate(gts):
                if matched[index]:
                    continue
                iou = _compute_iou(prediction["bbox_xyxy"], gt_box)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index >= 0:
                matched[best_index] = True
                tp += 1
            else:
                fp += 1

        fn += matched.count(False)

    return tp, fp, fn
