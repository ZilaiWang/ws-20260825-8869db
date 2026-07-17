"""P0-3 固定对象区域上的细类分类指标。

这些是对象级可分性诊断，不是正式检测 Recall/FDR。实现只依赖
NumPy，便于服务器结果回传后在本地独立复算。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from rsdet.data.xh_dataset import FINE_NAMES


def confusion_matrix_fixed(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int = 25,
) -> np.ndarray:
    """返回固定尺寸混淆矩阵，行为真值，列为预测。"""

    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("y_true 与 y_pred 形状不一致")
    if num_classes <= 0:
        raise ValueError("num_classes 必须大于 0")
    if true.size and ((true < 0).any() or (true >= num_classes).any()):
        raise ValueError("y_true 含越界类别")
    if pred.size and ((pred < 0).any() or (pred >= num_classes).any()):
        raise ValueError("y_pred 含越界类别")
    encoded = true * num_classes + pred
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def metrics_from_confusion(
    confusion: np.ndarray,
    *,
    class_names: Sequence[str] = FINE_NAMES,
    macro_class_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """从固定混淆矩阵计算 accuracy、macro recall/F1 和单类指标。

    macro recall/F1 只对本验证集有真实样本的类取平均；未出现类保留
    ``null`` 以防把“不可评估”写成“满分”。
    """

    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion 必须是方阵")
    if (matrix < 0).any():
        raise ValueError("confusion 不得包含负数")
    if len(class_names) != matrix.shape[0]:
        raise ValueError("class_names 数量与 confusion 不一致")

    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    recall = _safe_divide(true_positive, support)
    precision = _safe_divide(true_positive, predicted)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    active = support > 0
    requested = np.ones(matrix.shape[0], dtype=bool)
    if macro_class_ids is not None:
        requested[:] = False
        ids = np.asarray(list(macro_class_ids), dtype=np.int64)
        if ids.size and ((ids < 0).any() or (ids >= matrix.shape[0]).any()):
            raise ValueError("macro_class_ids 越界")
        requested[ids] = True
    macro_mask = active & requested

    total = int(matrix.sum())
    per_class: list[dict[str, Any]] = []
    for class_id, name in enumerate(class_names):
        has_support = bool(active[class_id])
        per_class.append(
            {
                "class_id": class_id,
                "class_name": name,
                "support": int(support[class_id]),
                "predicted_count": int(predicted[class_id]),
                "true_positive": int(true_positive[class_id]),
                "precision": float(precision[class_id]) if has_support else None,
                "recall": float(recall[class_id]) if has_support else None,
                "f1": float(f1[class_id]) if has_support else None,
            }
        )
    return {
        "n_samples": total,
        "accuracy": float(true_positive.sum() / total) if total else None,
        "macro_recall": float(recall[macro_mask].mean()) if macro_mask.any() else None,
        "balanced_accuracy": float(recall[macro_mask].mean()) if macro_mask.any() else None,
        "macro_f1": float(f1[macro_mask].mean()) if macro_mask.any() else None,
        "n_macro_classes": int(macro_mask.sum()),
        "macro_class_ids": np.flatnonzero(macro_mask).astype(int).tolist(),
        "per_class": per_class,
    }


def topk_accuracy(logits: np.ndarray, y_true: Sequence[int] | np.ndarray, k: int) -> float:
    """计算固定 k 的 top-k accuracy。"""

    scores = np.asarray(logits, dtype=np.float64)
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if scores.ndim != 2 or scores.shape[0] != true.size:
        raise ValueError("logits 必须是 [N,C] 且与 y_true 对齐")
    if not 1 <= k <= scores.shape[1]:
        raise ValueError("k 必须在 [1,C] 内")
    if not true.size:
        raise ValueError("空样本无法计算 top-k")
    top_indices = np.argpartition(scores, kth=scores.shape[1] - k, axis=1)[:, -k:]
    return float(np.any(top_indices == true[:, None], axis=1).mean())


def evaluate_classification(
    y_true: Sequence[int] | np.ndarray,
    logits: np.ndarray,
    *,
    class_names: Sequence[str] = FINE_NAMES,
    training_class_counts: Mapping[int, int] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """计算 P0-3 完整指标，同时给出飞机 20 类子集和头/中/尾诊断。"""

    scores = np.asarray(logits, dtype=np.float64)
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if scores.ndim != 2 or scores.shape != (true.size, len(class_names)):
        raise ValueError("logits 形状必须是 [N, len(class_names)]")
    if not np.isfinite(scores).all():
        raise ValueError("logits 含 NaN/Inf")
    pred = scores.argmax(axis=1)
    matrix = confusion_matrix_fixed(true, pred, len(class_names))
    result = metrics_from_confusion(matrix, class_names=class_names)
    result["top5_accuracy"] = topk_accuracy(scores, true, min(5, scores.shape[1]))

    subgroup_ids = {
        "ship4": list(range(0, 4)),
        "aircraft20": list(range(4, 24)),
        "vehicle1": [24],
    }
    subgroups: dict[str, Any] = {}
    for name, ids in subgroup_ids.items():
        mask = np.isin(true, ids)
        subgroup_matrix = confusion_matrix_fixed(true[mask], pred[mask], len(class_names))
        subgroup_metrics = metrics_from_confusion(
            subgroup_matrix, class_names=class_names, macro_class_ids=ids
        )
        subgroup_metrics.pop("per_class")
        subgroups[name] = subgroup_metrics
    result["subgroups"] = subgroups

    if training_class_counts:
        present = sorted(
            ((int(class_id), int(count)) for class_id, count in training_class_counts.items()),
            key=lambda item: (item[1], item[0]),
        )
        tiers: dict[str, list[int]] = {"tail": [], "middle": [], "head": []}
        for rank, (class_id, _) in enumerate(present):
            tier_index = min(2, (3 * rank) // max(1, len(present)))
            tiers[("tail", "middle", "head")[tier_index]].append(class_id)
        result["frequency_tiers"] = {}
        for tier_name, ids in tiers.items():
            tier_metrics = metrics_from_confusion(
                matrix, class_names=class_names, macro_class_ids=ids
            )
            result["frequency_tiers"][tier_name] = {
                "class_ids": ids,
                "class_names": [class_names[index] for index in ids],
                "macro_recall": tier_metrics["macro_recall"],
                "macro_f1": tier_metrics["macro_f1"],
                "n_samples": int(matrix[ids, :].sum()) if ids else 0,
            }
    return result, matrix
