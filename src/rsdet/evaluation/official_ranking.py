"""官方评分方案 V1.6 七项排名与二次排序模拟器。

官方机制（比赛评分方案 V1.6，2026-08-04 版）：

- 三大类各自的 Recall/FDR = 大类内细类指标的简单平均（船 4 型各 1/4、
  飞机 20 型各 1/20、车辆 1 型即 FSC 本身）。
- 每支队伍共 **7 项排名**：船 Recall、船 FDR、飞机 Recall、飞机 FDR、
  车辆 Recall、车辆 FDR、总时效性（10000×10000 大图推理时延，越小越好）。
- 对 7 个排名求和后进行二次排序（和越小越靠前）。
- 二次排序的从前往后百分比 ``p`` 决定初赛方案合理性、技术创新程度、
  工程可落地性三项打分区间：``(100% - p) ± 20%``，每项下限 0、上限 10。

本模块只负责"给定多支队伍的 7 项原始指标，计算每支队伍的排名、排名和、
二次排序百分比与打分区间"。指标本身的合法来源是
:func:`rsdet.evaluation.official_metric.evaluate_ranking_metrics` 的
``per_coarse`` 输出，加上 E 分工的 10K 时延实测。单队伍评估见
``scripts/evaluate.py``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# 7 项排名的官方顺序；``lower_is_better`` 标记时延与 FDR 这类"越小越好"的项。
SEVEN_RANKING_METRICS: tuple[str, ...] = (
    "ship_recall",
    "ship_fdr",
    "aircraft_recall",
    "aircraft_fdr",
    "vehicle_recall",
    "vehicle_fdr",
    "latency_seconds",
)
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {"ship_fdr", "aircraft_fdr", "vehicle_fdr", "latency_seconds"}
)


@dataclass(frozen=True)
class RankingItem:
    """一支队伍在 7 项指标上的原始得分。

    时延项（``latency_seconds``）允许缺失：缺失时该项不参与该队伍的排名
    与求和，其余 6 项照常计算；排名表会标记该队伍 ``incomplete=True``，
    防止把缺项队伍与完整队伍直接比排名和。
    """

    team_id: str
    ship_recall: float
    ship_fdr: float
    aircraft_recall: float
    aircraft_fdr: float
    vehicle_recall: float
    vehicle_fdr: float
    latency_seconds: float | None = None

    def metric_values(self) -> dict[str, float]:
        """返回 {指标名: 原始值}；值为 ``None`` 的指标不包含。

        与 :func:`_rank_metric` 的 None 过滤保持一致：缺失指标（例如单大类
        评估时其他大类为 ``None``、或未提供时延）一律不进入排名计算。
        """
        values: dict[str, float] = {}
        for key, value in (
            ("ship_recall", self.ship_recall),
            ("ship_fdr", self.ship_fdr),
            ("aircraft_recall", self.aircraft_recall),
            ("aircraft_fdr", self.aircraft_fdr),
            ("vehicle_recall", self.vehicle_recall),
            ("vehicle_fdr", self.vehicle_fdr),
            ("latency_seconds", self.latency_seconds),
        ):
            if value is not None:
                values[key] = value
        return values

    @property
    def incomplete(self) -> bool:
        """是否缺失时延项（当前唯一允许缺失的指标）。"""
        return self.latency_seconds is None


@dataclass(frozen=True)
class TeamRankingResult:
    """一支队伍的完整排名结果。"""

    team_id: str
    metric_values: dict[str, float] = field(default_factory=dict)
    metric_ranks: dict[str, int] = field(default_factory=dict)
    rank_sum: int = 0
    rank_count: int = 0
    second_order_position: int | None = None
    second_order_percentile: float | None = None
    score_min: float = 0.0
    score_max: float = 10.0
    incomplete: bool = False

    def to_dict(self) -> dict[str, Any]:
        """序列化为便于写 JSON 的字典。"""
        return {
            "team_id": self.team_id,
            "metric_values": dict(self.metric_values),
            "metric_ranks": dict(self.metric_ranks),
            "rank_sum": self.rank_sum,
            "rank_count": self.rank_count,
            "second_order_position": self.second_order_position,
            "second_order_percentile": self.second_order_percentile,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "incomplete": self.incomplete,
        }


def _rank_metric(
    values: dict[str, float | None], metric: str
) -> dict[str, int]:
    """对单个指标给所有队伍排名（竞赛排名：并列同分、名次 1,2,2,4）。

    ``None`` 表示该项缺失，直接排除不参与该指标排名。

    Returns:
        {team_id: rank}，rank 从 1 开始。
    """
    present = {team_id: value for team_id, value in values.items() if value is not None}
    if not present:
        return {}
    lower_is_better = metric in LOWER_IS_BETTER
    # lower_is_better: 值越小越靠前，直接按值升序；
    # higher_is_better: 值越大越靠前，取负后升序等价于按值降序。
    ordered = sorted(
        present.items(),
        key=lambda pair: (pair[1] if lower_is_better else -pair[1], pair[0]),
    )
    ranks: dict[str, int] = {}
    for index, (team_id, value) in enumerate(ordered, start=1):
        if index == 1:
            ranks[team_id] = 1
            continue
        previous_id, previous_value = ordered[index - 2]
        if value == previous_value:
            ranks[team_id] = ranks[previous_id]
        else:
            ranks[team_id] = index
    return ranks


def compute_official_rankings(
    items: list[RankingItem],
    *,
    percentile_fraction_scale: float = 0.5,
) -> list[TeamRankingResult]:
    """计算所有队伍的 7 项排名与二次排序结果。

    Args:
        items: 各队伍原始指标。
        percentile_fraction_scale: 二次排序"从前往后百分比"的取法。
            官方只给出示例（第 30% -> 区间 5-9 分），未明确是
            ``position / n`` 还是 ``(position - 1) / (n - 1)``。默认
            使用 ``position / n``（队伍越靠前百分比越小、打分区间越高，
            与官方示例一致）；如需保守取法可传 ``0.0`` 变为
            ``(position - 1) / (n - 1)``。

    Returns:
        按二次排序名次升序排列的逐队结果。
    """
    if not items:
        return []
    if len({item.team_id for item in items}) != len(items):
        raise ValueError("team_id 必须唯一")

    # 构建 {metric: {team_id: value_or_None}}。
    metric_values: dict[str, dict[str, float | None]] = {
        metric: {} for metric in SEVEN_RANKING_METRICS
    }
    for item in items:
        present = item.metric_values()
        for metric in SEVEN_RANKING_METRICS:
            metric_values[metric][item.team_id] = present.get(metric)

    metric_ranks: dict[str, dict[str, int]] = {}
    for metric in SEVEN_RANKING_METRICS:
        metric_ranks[metric] = _rank_metric(metric_values[metric], metric)

    n_teams = len(items)
    rank_sums: dict[str, tuple[int, int]] = {}
    for item in items:
        total = 0
        count = 0
        present = item.metric_values()
        for metric in SEVEN_RANKING_METRICS:
            if metric not in present:
                continue
            total += metric_ranks[metric][item.team_id]
            count += 1
        rank_sums[item.team_id] = (total, count)

    ordered = sorted(items, key=lambda item: (rank_sums[item.team_id][0], item.team_id))
    position_map = {item.team_id: position for position, item in enumerate(ordered, start=1)}

    results: list[TeamRankingResult] = []
    for item in ordered:
        total, count = rank_sums[item.team_id]
        position = position_map[item.team_id]
        if n_teams > 1:
            if percentile_fraction_scale == 0.0:
                percentile = (position - 1) / (n_teams - 1)
            else:
                percentile = position / n_teams
        else:
            percentile = 0.5
        # 官方公式：(100% - p) ± 20%，每项下限 0、上限 10。
        score_min = max(0.0, (1.0 - percentile - 0.20) * 10.0)
        score_max = min(10.0, (1.0 - percentile + 0.20) * 10.0)
        results.append(
            TeamRankingResult(
                team_id=item.team_id,
                metric_values=item.metric_values(),
                metric_ranks={
                    metric: metric_ranks[metric][item.team_id]
                    for metric in SEVEN_RANKING_METRICS
                    if metric in item.metric_values()
                },
                rank_sum=total,
                rank_count=count,
                second_order_position=position,
                second_order_percentile=percentile,
                score_min=score_min,
                score_max=score_max,
                incomplete=item.incomplete,
            )
        )
    return results


def ranking_result_to_dicts(results: list[TeamRankingResult]) -> list[dict[str, Any]]:
    """兼容旧 JSON 输出的便捷序列化。"""
    return [result.to_dict() for result in results]


def format_ranking_table(results: list[TeamRankingResult]) -> list[str]:
    """生成适合打印的排名表文本行。"""
    header = (
        f"{'team':<12} {'2nd':>4} {'pct':>6} {'rank_sum':>8} "
        f"{'score_range':>14}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        range_text = (
            f"[{result.score_min:.1f}, {result.score_max:.1f}]"
            if not result.incomplete
            else "[incomplete]"
        )
        lines.append(
            f"{result.team_id:<12} {str(result.second_order_position):>4} "
            f"{result.second_order_percentile:>6.3f} {result.rank_sum:>8} "
            f"{range_text:>14}"
        )
    return lines


def compute_score_range_from_percentile(
    percentile: float, *, lower_bound: float = 0.0, upper_bound: float = 10.0
) -> tuple[float, float]:
    """按官方公式计算打分区间：``(100% - p) ± 20%``，并夹取到 [0, 10]。"""
    if not math.isfinite(percentile) or not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile 必须在 [0, 1] 内")
    score_min = max(lower_bound, (1.0 - percentile - 0.20) * 10.0)
    score_max = min(upper_bound, (1.0 - percentile + 0.20) * 10.0)
    if score_max < score_min:
        raise ValueError("打分区间非法：max 小于 min")
    return score_min, score_max
