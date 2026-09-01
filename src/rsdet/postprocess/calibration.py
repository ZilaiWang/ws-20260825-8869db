"""全局置信度阈值扫描与工作点选择。

这里暂不实现 Platt scaling 等分数变换，只提供所有模型都能直接使用的全局
阈值基线。每个扫描点复用官方评估器，避免出现第二套匹配规则。
"""

import math
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from statistics import fmean
from typing import Any

from rsdet.evaluation.official_metric import (
    OverallMetrics,
    RankingMetrics,
    evaluate_predictions,
    evaluate_ranking_metrics,
)
from rsdet.evaluation.platform_protocol import (
    COARSE_ORDER,
    PLATFORM_OBSERVED_PROTOCOL,
)

# This module is an active formal threshold-selection entrypoint.  Keeping the
# binding explicit makes protocol audits fail closed when the platform contract
# changes.
FORMAL_METRIC_PROTOCOL = PLATFORM_OBSERVED_PROTOCOL


@dataclass(frozen=True)
class ThresholdSweepPoint:
    """一个阈值及其官方评估结果。"""

    threshold: float
    detections_kept: int
    metrics: OverallMetrics
    ranking_metrics: RankingMetrics


@dataclass(frozen=True)
class OperatingPointSelection:
    """一个可复现的阈值工作点。"""

    point: ThresholdSweepPoint
    policy: str
    passed: bool | None


def build_threshold_grid(start: float, stop: float, step: float) -> list[float]:
    """用十进制步长生成闭区间内的阈值，避免 ``0.1 + 0.2`` 漂移。"""
    values = {"start": start, "stop": stop, "step": step}
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} 必须是有限数")
    if not 0.0 <= start <= 1.0 or not 0.0 <= stop <= 1.0:
        raise ValueError("start 和 stop 必须在 [0, 1] 内")
    if start > stop:
        raise ValueError("start 不能大于 stop")
    if step <= 0.0:
        raise ValueError("step 必须大于 0")

    start_decimal = Decimal(str(start))
    stop_decimal = Decimal(str(stop))
    step_decimal = Decimal(str(step))
    count = (
        int(((stop_decimal - start_decimal) / step_decimal).to_integral_value(rounding=ROUND_FLOOR))
        + 1
    )
    if count > 10_001:
        raise ValueError("阈值点超过 10001 个，请增大 step")
    return [float(start_decimal + index * step_decimal) for index in range(count)]


def filter_predictions_by_score(
    pred_boxes: dict[int, list[dict[str, Any]]],
    threshold: float,
) -> dict[int, list[dict[str, Any]]]:
    """保留 ``score >= threshold`` 的预测，不修改输入。"""
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold 必须是 [0, 1] 内的有限数")
    filtered: dict[int, list[dict[str, Any]]] = {}
    for image_id, items in pred_boxes.items():
        filtered[image_id] = []
        for item in items:
            score = float(item["score"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"预测 score 必须是 [0, 1] 内的有限数: {score}")
            if score >= threshold:
                filtered[image_id].append(item)
    return filtered


def sweep_global_thresholds(
    gt_boxes: dict[int, list[dict[str, Any]]],
    pred_boxes: dict[int, list[dict[str, Any]]],
    thresholds: list[float],
    *,
    class_names: list[str],
    category_mapping: dict[int, str],
    iou_thresholds: dict[str, float],
    require_complete_taxonomy: bool = True,
) -> list[ThresholdSweepPoint]:
    """在一组全局阈值上调用官方评估器。"""
    if not thresholds:
        raise ValueError("thresholds 不能为空")

    points: list[ThresholdSweepPoint] = []
    for threshold in thresholds:
        filtered = filter_predictions_by_score(pred_boxes, threshold)
        result = evaluate_predictions(
            gt_boxes,
            filtered,
            class_names=class_names,
            category_mapping=category_mapping,
            iou_thresholds=iou_thresholds,
        )
        ranking = evaluate_ranking_metrics(
            gt_boxes,
            filtered,
            class_names=class_names,
            category_mapping=category_mapping,
            iou_thresholds=iou_thresholds,
            require_complete_taxonomy=require_complete_taxonomy,
        )
        points.append(
            ThresholdSweepPoint(
                threshold=threshold,
                detections_kept=sum(len(items) for items in filtered.values()),
                metrics=result,
                ranking_metrics=ranking,
            )
        )
    return points


def select_operating_points(
    points: list[ThresholdSweepPoint],
    *,
    official_recall_min: float,
    official_fdr_max: float,
    internal_recall_min: float = 0.88,
    internal_fdr_max: float = 0.17,
) -> dict[str, OperatingPointSelection]:
    """选择官方最优、内部稳健和 Recall 上限三个工作点。

    官方和内部工作点先满足各自 FDR 上限，再按 Recall 高、FDR 低、阈值高
    排序；若没有任何点满足 FDR 上限，则退回 FDR 最低的点并标记未通过。
    Recall 上限不施加 FDR 约束，仅用于诊断。
    """
    if not points:
        raise ValueError("points 不能为空")
    limits = {
        "official_recall_min": official_recall_min,
        "official_fdr_max": official_fdr_max,
        "internal_recall_min": internal_recall_min,
        "internal_fdr_max": internal_fdr_max,
    }
    for name, value in limits.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} 必须是 [0, 1] 内的有限数")

    def platform_values(point: ThresholdSweepPoint) -> tuple[float, float]:
        missing = set(COARSE_ORDER) - set(point.ranking_metrics.per_coarse)
        if missing:
            # Partial-taxonomy/unit diagnostics cannot define the platform
            # macro-over-three gate.  Preserve their historical diagnostic
            # behavior; formal callers are separately required to supply the
            # complete 25-class taxonomy before admission.
            return (
                float(point.ranking_metrics.overall_recall),
                float(point.ranking_metrics.overall_fdr),
            )
        coarse = point.ranking_metrics.per_coarse
        return (
            fmean(coarse[name].macro_recall for name in COARSE_ORDER),
            fmean(coarse[name].macro_fdr for name in COARSE_ORDER),
        )

    def best_under_fdr(fdr_max: float) -> ThresholdSweepPoint:
        def recall(point: ThresholdSweepPoint) -> float:
            return platform_values(point)[0]

        def fdr(point: ThresholdSweepPoint) -> float:
            return platform_values(point)[1]

        feasible = [point for point in points if fdr(point) <= fdr_max]
        candidates = feasible or points
        if feasible:
            return max(
                candidates,
                key=lambda point: (
                    recall(point),
                    -fdr(point),
                    point.threshold,
                ),
            )
        return max(
            candidates,
            key=lambda point: (
                -fdr(point),
                recall(point),
                point.threshold,
            ),
        )

    official = best_under_fdr(official_fdr_max)
    internal = best_under_fdr(internal_fdr_max)
    recall_ceiling = max(
        points,
        key=lambda point: (
            platform_values(point)[0],
            -platform_values(point)[1],
            point.threshold,
        ),
    )
    return {
        "official_best": OperatingPointSelection(
            point=official,
            policy="FDR 不超过官方上限时 Recall 最高；再按 FDR 低、阈值高选择",
            passed=(
                platform_values(official)[0] >= official_recall_min
                and platform_values(official)[1] <= official_fdr_max
            ),
        ),
        "internal_best": OperatingPointSelection(
            point=internal,
            policy="FDR 不超过内部上限时 Recall 最高；再按 FDR 低、阈值高选择",
            passed=(
                platform_values(internal)[0] >= internal_recall_min
                and platform_values(internal)[1] <= internal_fdr_max
            ),
        ),
        "recall_ceiling": OperatingPointSelection(
            point=recall_ceiling,
            policy="不限制 FDR，Recall 最高；再按 FDR 低、阈值高选择",
            passed=None,
        ),
    }
