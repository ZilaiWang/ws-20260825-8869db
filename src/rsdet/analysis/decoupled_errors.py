"""N0-2：定位 / 分类错误解耦分析。

目的
----
M1 错误分解显示"细类错 (FN_CLS=1115) 远多于定位错 (FN_LOC=66)"，但该结论
基于官方细类匹配。本模块用两个**解耦指标**独立验证：

- ``R_loc@oracle-class``（oracle 定位召回）：给每个预测框免费的**正确细类
  标签**（相当于"分类全对"），再按几何 IoU 匹配 GT。它度量：如果分类不是
  问题，纯几何定位能找回多少目标（Recall 上限）。
- ``Acc_fine@localized``（已定位对象中的细类正确率）：在**几何已匹配**的
  对象上（不看预测细类，只看 IoU 匹配），预测细类与 GT 一致的比例。它度量
  定位过关后分类质量。

两者结合判断瓶颈归属：
- oracle 定位召回高 + 官方 recall 低 ⇒ 定位不是瓶颈，细类分类是瓶颈。
- oracle 定位召回也低 ⇒ 定位本身有容量缺口（框没框住目标）。

分层与区间
----------
- 按 fold、来源组（source_group）、目标短边尺寸档、边界风险、细类分层；
- 对 oracle 定位召回做 **source-group bootstrap**（以来源组为单位重采样），
  给出 95% 区间，避免少量大组主导点估计。

评估约束
--------
- 匹配规则与官方一致（分数降序贪心、一对一、IoU 阈值按大类），仅细类约束
  放宽为"同大类即可"（oracle 语义）。
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import EvaluationProtocol


@dataclass
class OracleLocalizationMetrics:
    """oracle 定位匹配的逐对象记录与汇总。"""

    matches: list[dict[str, Any]] = field(default_factory=list)
    unmatched_gt_keys: list[tuple[int, int]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        denominator = len(self.matches) + len(self.unmatched_gt_keys)
        return len(self.matches) / denominator if denominator else 1.0


@dataclass
class LocalizedFineAccuracyMetrics:
    """在 oracle 定位匹配对象上的细类准确率。"""

    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 1.0


def _gt_key(image_id: int, index: int) -> tuple[int, int]:
    return (image_id, index)


def compute_oracle_localization(
    gt_boxes: Mapping[int, list[dict[str, Any]]],
    predictions: Mapping[int, list[dict[str, Any]]],
    *,
    protocol: EvaluationProtocol,
    threshold: float,
) -> tuple[OracleLocalizationMetrics, list[dict[str, Any]]]:
    """计算 oracle 定位召回与逐对象明细。

    oracle 语义：预测细类不参与匹配约束，只要求预测框与 GT 的**大类**相同，
    并按 IoU 阈值匹配（分数降序贪心、一对一）。匹配成功后记录 GT 的细类，
    用于后续 ``Acc_fine@localized``。

    Returns:
        (metrics, detail_rows)。detail_rows 每项对应一个 GT 对象：
        ``{annotation_uid_key, image_id, index, class_name, category_id,
        oracle_matched, matched_iou, prediction_index, predicted_category_id,
        correct_fine}``。
    """
    thresholds = protocol.iou_thresholds
    mapping = {int(category_id): name for category_id, name in protocol.category_mapping.items()}
    coarse_of: dict[int, str] = {}
    for category_id, name in mapping.items():
        coarse_of[category_id] = name

    # 按大类拆分 GT 与预测。
    gt_by_coarse: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    pred_by_coarse: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)

    for image_id, records in gt_boxes.items():
        for index, record in enumerate(records):
            category_id = int(record["category_id"])
            if category_id not in coarse_of:
                raise ValueError(f"GT category_id={category_id} 缺少大类映射")
            gt_by_coarse[coarse_of[category_id]].append((image_id, index, record))

    for image_id, records in predictions.items():
        for index, record in enumerate(records):
            score = float(record["score"])
            if score < threshold:
                continue
            category_id = int(record["category_id"])
            if category_id not in coarse_of:
                raise ValueError(f"预测 category_id={category_id} 缺少大类映射")
            pred_by_coarse[coarse_of[category_id]].append((image_id, index, record))

    metrics = OracleLocalizationMetrics()
    detail_rows: list[dict[str, Any]] = []
    matched_gt: set[tuple[int, int]] = set()
    all_gt_keys: set[tuple[int, int]] = {
        (image_id, index) for image_id, records in gt_boxes.items() for index in range(len(records))
    }

    for coarse in protocol.class_names:
        iou_threshold = float(thresholds[coarse])
        gts = gt_by_coarse[coarse]
        preds = pred_by_coarse[coarse]
        # 分数降序贪心：为每个预测找最佳未匹配 GT。
        preds_sorted = sorted(preds, key=lambda item: -float(item[2]["score"]))
        matched_pred: set[tuple[int, int]] = set()
        for pred_image_id, pred_index, pred_record in preds_sorted:
            if (pred_image_id, pred_index) in matched_pred:
                continue
            best_iou = -1.0
            best_gt: tuple[int, int, dict[str, Any]] | None = None
            for gt_image_id, gt_index, gt_record in gts:
                if (gt_image_id, gt_index) in matched_gt:
                    continue
                if gt_image_id != pred_image_id:
                    continue
                iou = compute_iou(
                    [float(v) for v in gt_record["bbox_xyxy"]],
                    [float(v) for v in pred_record["bbox_xyxy"]],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt = (gt_image_id, gt_index, gt_record)
            if best_gt is not None and best_iou >= iou_threshold:
                gt_image_id, gt_index, gt_record = best_gt
                matched_gt.add((gt_image_id, gt_index))
                matched_pred.add((pred_image_id, pred_index))
                metrics.matches.append(
                    {
                        "image_id": gt_image_id,
                        "index": gt_index,
                        "iou": best_iou,
                        "prediction_index": pred_index,
                        "predicted_category_id": int(pred_record["category_id"]),
                        "gt_category_id": int(gt_record["category_id"]),
                    }
                )

    metrics.unmatched_gt_keys = sorted(
        all_gt_keys - matched_gt,
        key=lambda item: (item[0], item[1]),
    )

    # 生成逐对象明细（全部 GT）。
    for image_id, records in gt_boxes.items():
        for index, record in enumerate(records):
            category_id = int(record["category_id"])
            matched = [
                m for m in metrics.matches if m["image_id"] == image_id and m["index"] == index
            ]
            oracle_matched = bool(matched)
            if oracle_matched:
                correct_fine = bool(matched[0]["predicted_category_id"] == category_id)
                matched_iou = matched[0]["iou"]
                pred_cat = matched[0]["predicted_category_id"]
                pred_index = matched[0]["prediction_index"]
            else:
                matched_iou = None
                pred_cat = None
                pred_index = None
            detail_rows.append(
                {
                    "image_id": image_id,
                    "gt_index": index,
                    "class_name": record.get("class_name", ""),
                    "category_id": category_id,
                    "oracle_matched": oracle_matched,
                    "matched_iou": matched_iou,
                    "predicted_category_id": pred_cat,
                    "prediction_index": pred_index,
                    "correct_fine": (correct_fine if oracle_matched else None),
                }
            )
    return metrics, detail_rows


def compute_localized_fine_accuracy(
    detail_rows: Iterable[dict[str, Any]],
) -> LocalizedFineAccuracyMetrics:
    """在 oracle 已匹配对象上计算细类准确率。"""
    result = LocalizedFineAccuracyMetrics()
    for row in detail_rows:
        if not row["oracle_matched"]:
            continue
        result.total += 1
        result.correct += int(bool(row["correct_fine"]))
    return result


def compute_source_group_bootstrap(
    detail_rows: list[dict[str, Any]],
    *,
    group_of_image: Mapping[int, str],
    iterations: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """对 oracle 定位召回做 source-group bootstrap。

    以**来源组**为单位重采样（组内保持完整，因为同一来源的航片存在域相关），
    给出 95% 区间。要求每张图已分组。

    Returns:
        ``{n_groups, n_iterations, point_recall, ci_low, ci_high,
        ci_alpha}``。
    """
    group_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        group = group_of_image.get(row["image_id"])
        if group is None:
            raise ValueError(f"image_id={row['image_id']} 缺少来源组映射")
        group_members[group].append(row)

    groups = list(group_members)
    n_groups = len(groups)
    if n_groups < 2:
        raise ValueError(f"bootstrap 需要至少 2 个来源组，实际 {n_groups}")

    total = len(detail_rows)
    matched = sum(bool(row["oracle_matched"]) for row in detail_rows)
    point = matched / total if total else 0.0

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(iterations):
        sample_rows: list[dict[str, Any]] = []
        for _ in range(n_groups):
            group = rng.choice(groups)
            sample_rows.extend(group_members[group])
        if not sample_rows:
            continue
        n_match = sum(bool(row["oracle_matched"]) for row in sample_rows)
        draws.append(n_match / len(sample_rows))
    draws.sort()
    lower_index = int(alpha / 2.0 * len(draws))
    upper_index = int((1.0 - alpha / 2.0) * len(draws))
    lower_index = min(max(lower_index, 0), len(draws) - 1)
    upper_index = min(max(upper_index, 0), len(draws) - 1)
    return {
        "n_groups": n_groups,
        "n_iterations": iterations,
        "point_recall": point,
        "ci_low": draws[lower_index],
        "ci_high": draws[upper_index],
        "ci_alpha": alpha,
        "seed": seed,
    }


def _size_bin(short_edge: float) -> str:
    """目标短边尺寸档（像素）。"""
    if short_edge < 16:
        return "tiny_lt16"
    if short_edge < 32:
        return "small_16_32"
    if short_edge < 64:
        return "medium_32_64"
    return "large_ge64"


def stratify_oracle_localization(
    detail_rows: list[dict[str, Any]],
    gt_boxes: Mapping[int, list[dict[str, Any]]],
    *,
    group_of_image: Mapping[int, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """对 oracle 定位召回做多维度分层汇总。

    维度：fold（从 formal manifest 推导的 fold 需另行传入；此处用 image_id
    无法推断 fold，故以来源组为最小单元）、来源组、尺寸档、边界风险、细类。

    Returns:
        ``{scope: {n_objects, matched, recall}}``。
    """
    sizes: dict[tuple[int, int], float] = {}
    for image_id, records in gt_boxes.items():
        for index, record in enumerate(records):
            x0, y0, x1, y1 = (float(v) for v in record["bbox_xyxy"])
            sizes[(image_id, index)] = min(x1 - x0, y1 - y0)

    scopes: dict[str, dict[str, Any]] = defaultdict(lambda: {"n_objects": 0, "matched": 0})
    for row in detail_rows:
        key = (row["image_id"], row["gt_index"])
        short_edge = sizes.get(key, float("inf"))
        size_bin = _size_bin(short_edge)
        for scope in (
            "overall",
            f"class_{row['class_name']}",
            f"category_{row['category_id']:02d}",
            f"size_{size_bin}",
        ):
            scopes[scope]["n_objects"] += 1
            scopes[scope]["matched"] += int(bool(row["oracle_matched"]))
        if group_of_image is not None:
            group = group_of_image.get(row["image_id"], "unknown")
            scopes[f"group_{group}"]["n_objects"] += 1
            scopes[f"group_{group}"]["matched"] += int(bool(row["oracle_matched"]))

    result: dict[str, dict[str, Any]] = {}
    for scope, counts in scopes.items():
        total = counts["n_objects"]
        result[scope] = {
            "n_objects": total,
            "matched": counts["matched"],
            "recall": (counts["matched"] / total if total else 1.0),
        }
    return result


def analyze_decoupled_errors(
    *,
    gt_boxes: Mapping[int, list[dict[str, Any]]],
    predictions: Mapping[int, list[dict[str, Any]]],
    protocol: EvaluationProtocol,
    threshold: float,
    group_of_image: Mapping[int, str] | None = None,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """完整 N0-2 分析：oracle 定位召回 + 细类准确率 + 分层 + bootstrap。

    Returns:
        结果字典（可 JSON 序列化）。
    """
    metrics, detail_rows = compute_oracle_localization(
        gt_boxes,
        predictions,
        protocol=protocol,
        threshold=threshold,
    )
    fine_accuracy = compute_localized_fine_accuracy(detail_rows)
    stratified = stratify_oracle_localization(
        detail_rows,
        gt_boxes,
        group_of_image=group_of_image,
    )
    bootstrap = None
    if group_of_image is not None:
        bootstrap = compute_source_group_bootstrap(
            detail_rows,
            group_of_image=group_of_image,
            iterations=bootstrap_iterations,
            seed=seed,
        )
    return {
        "analysis": "N0-2_localization_classification_decoupling",
        "threshold": threshold,
        "oracle_localization_recall": metrics.recall,
        "oracle_matched_objects": len(metrics.matches),
        "oracle_unmatched_gt": len(metrics.unmatched_gt_keys),
        "localized_fine_accuracy": fine_accuracy.accuracy,
        "localized_fine_total": fine_accuracy.total,
        "localized_fine_correct": fine_accuracy.correct,
        "stratified": stratified,
        "source_group_bootstrap": bootstrap,
        "note": (
            "R_loc@oracle-class = oracle_localization_recall（预测细类免费）；"
            "Acc_fine@localized = localized_fine_accuracy（几何已匹配上的细类正确率）。"
        ),
    }
