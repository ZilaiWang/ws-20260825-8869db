"""N2-CFG：粗类条件式前景门控的核心逻辑。

本模块不强依赖 PyTorch，纯 CPU 即可单测，负责门控的全部**数值逻辑**：

1. ``context_1.25`` 候选框扩展（与 formal crop manifest 的 context_1p25 语义一致）；
2. 粗类映射（ship / aircraft / vehicle）；首轮正式门控只启用 ship / vehicle，
   飞机只做 shadow 旁路，输出逐条不变；
3. 低容量、单调的风险校准器
   ``q = sigmoid(alpha * logit(s) + beta * fg_logit + gamma_c)``，
   其中 ``alpha, beta >= 0`` 保证单调，``gamma_c`` 为三个粗类偏置；
4. 单一全局删除阈值 ``tau_drop``；
5. S0 / S1 / S2 三种快筛模式的分数构造。

设计边界严格继承《改进方案 1》第 2 节与第 5 节：

- 只做 ``background_reject``，不做 reclassify / joint；
- 不改 bbox、不改 fine class、不恢复新候选；
- 只允许人工确认的 ``clear_background`` 作负样本（样本构造在 manifest 层，
  本模块不关心）；
- 不引入 25 个类别阈值、source-specific 阈值或尺寸/位置分位。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

# 三个粗类（官方协议排序）。
COARSE_CLASSES: tuple[str, ...] = ("ship", "aircraft", "vehicle")

# 首轮正式门控只启用 ship / vehicle；aircraft 旁路 shadow。
ACTIVE_COARSE_FIRST_ROUND: frozenset[str] = frozenset({"ship", "vehicle"})

# 快筛模式。
MODE_S0 = "S0"  # 仅 R1-6 score 的单调校准器（排除"只是重新调阈值"）
MODE_S1 = "S1"  # shared foreground head + score
MODE_S2 = "S2"  # coarse-conditioned head + score（主候选）

ALLOWED_MODES: frozenset[str] = frozenset({MODE_S0, MODE_S1, MODE_S2})

CONTEXT_EXPANSION: float = 1.25
INPUT_RESOLUTION: int = 224

_EPSILON = 1e-7


def coarse_of_category_id(category_id: int) -> str:
    """官方粗类映射：0-3 ship，4-23 aircraft，24 vehicle。"""
    if category_id < 0 or category_id > 24:
        raise ValueError(f"非法 category_id={category_id}（合法 0..24）")
    if category_id < 4:
        return "ship"
    if category_id < 24:
        return "aircraft"
    return "vehicle"


def _coarse_index(coarse: str) -> int:
    try:
        return COARSE_CLASSES.index(coarse)
    except ValueError as error:
        raise ValueError(f"未知粗类 {coarse!r}（合法 {COARSE_CLASSES}）") from error


def _logit(probability: float) -> float:
    """数值稳定的 logit，输出 clip 到 [-60, 60]。"""
    value = min(1.0 - _EPSILON, max(_EPSILON, float(probability)))
    return max(-60.0, min(60.0, math.log(value / (1.0 - value))))


def _sigmoid(logit_value: float) -> float:
    value = max(-60.0, min(60.0, float(logit_value)))
    return 1.0 / (1.0 + math.exp(-value))


def expand_context_bbox(
    bbox_xyxy: Sequence[float],
    *,
    ratio: float = CONTEXT_EXPANSION,
) -> tuple[float, float, float, float]:
    """以候选框长边 ``* ratio`` 为正方形边长、中心不变地扩展。

    与 formal crop manifest 的 ``context_1p25`` 语义一致：长边乘 ratio 得到
    正方形边长，中心点保持不变。不做裁剪——越界部分交给 ``render_crop``
    的 ``EXTENT`` 变换用 ``fillcolor`` 填充（对应 ``outside_policy=pad``）。
    """
    if not math.isfinite(ratio) or ratio <= 1.0:
        raise ValueError(f"ratio 必须 > 1 的有限数，实际 {ratio}")
    box = tuple(float(value) for value in bbox_xyxy)
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError(f"非法 bbox_xyxy={bbox_xyxy}")
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"非法 bbox_xyxy（宽高非正）: {bbox_xyxy}")
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * ratio
    half = side / 2.0
    return (center_x - half, center_y - half, center_x + half, center_y + half)


@dataclass(frozen=True)
class GateCalibration:
    """低容量单调校准器参数。

    ``gamma`` 为三个粗类的偏置。S0 / S1 要求三者相等（coarse-agnostic），
    S2 允许三者不同（coarse-conditioned）。``beta=0`` 表示不使用前景证据
    （S0）。
    """

    alpha: float
    beta: float
    gamma: Mapping[str, float]
    tau_drop: float
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in ALLOWED_MODES:
            raise ValueError(f"未知 mode={self.mode!r}")
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha 必须是 >= 0 的有限数")
        if not math.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta 必须是 >= 0 的有限数")
        if not math.isfinite(self.tau_drop) or not 0.0 <= self.tau_drop <= 1.0:
            raise ValueError("tau_drop 必须是 [0, 1] 内的有限数")
        missing = set(COARSE_CLASSES) - set(self.gamma)
        extra = set(self.gamma) - set(COARSE_CLASSES)
        if missing or extra:
            raise ValueError(f"gamma 必须恰好覆盖 {COARSE_CLASSES}，缺 {missing} 多 {extra}")
        if any(not math.isfinite(value) for value in self.gamma.values()):
            raise ValueError("gamma 必须是有限数")
        if self.mode in {MODE_S0, MODE_S1}:
            values = {self.gamma[name] for name in COARSE_CLASSES}
            if len(values) != 1:
                raise ValueError(f"{self.mode} 要求三个粗类 gamma 相等（coarse-agnostic）")


def calibrate_q(
    score: float,
    fg_logit: float,
    coarse: str,
    calibration: GateCalibration,
) -> float:
    """计算融合后的风险分数 ``q``。

    ``fg_logit`` 直接取模型的 raw logit（即 ``logit(p_fg)``），与 score 在
    logit 域融合。``alpha, beta >= 0`` 保证 q 对 score 与 fg_logit 单调不减。
    """
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"score 必须是 [0, 1] 内有限数: {score}")
    if not math.isfinite(fg_logit):
        raise ValueError(f"fg_logit 必须是有限数: {fg_logit}")
    if coarse not in calibration.gamma:
        raise ValueError(f"gamma 未覆盖粗类 {coarse!r}")
    z = (
        calibration.alpha * _logit(score)
        + calibration.beta * fg_logit
        + calibration.gamma[coarse]
    )
    return _sigmoid(z)


def _shared_gamma(value: float) -> dict[str, float]:
    return {name: value for name in COARSE_CLASSES}


def make_calibration(
    *,
    mode: str,
    alpha: float,
    beta: float,
    tau_drop: float,
    gamma: Mapping[str, float] | float,
) -> GateCalibration:
    """构造校准器。``gamma`` 可为单一浮点（S0/S1）或三粗类映射（S2）。"""
    if isinstance(gamma, Mapping):
        gamma_map = {name: float(gamma[name]) for name in COARSE_CLASSES}
    else:
        gamma_map = _shared_gamma(float(gamma))
    return GateCalibration(
        alpha=float(alpha),
        beta=float(beta),
        gamma=gamma_map,
        tau_drop=float(tau_drop),
        mode=mode,
    )


def apply_gate(
    candidates: Sequence[Mapping[str, Any]],
    fg_logits: Mapping[str, float],
    calibration: GateCalibration,
    *,
    active_coarse: frozenset[str] = ACTIVE_COARSE_FIRST_ROUND,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """对候选执行保留/删除，返回 ``(kept, removed)``。

    每个候选须含 ``proposal_uid`` / ``category_id`` / ``score``。只有
    ``active_coarse`` 内的候选参与删除；其余（含飞机 shadow）逐条原样保留。
    S0 的 ``fg_logits`` 可传空（``beta=0`` 时不影响结果）。
    """
    if calibration.mode not in ALLOWED_MODES:
        raise ValueError(f"未知 mode={calibration.mode!r}")
    kept: list[Mapping[str, Any]] = []
    removed: list[Mapping[str, Any]] = []
    for candidate in candidates:
        proposal_uid = str(candidate["proposal_uid"])
        category_id = int(candidate["category_id"])
        score = float(candidate["score"])
        coarse = coarse_of_category_id(category_id)
        if coarse not in active_coarse:
            kept.append(candidate)
            continue
        fg_logit = fg_logits.get(proposal_uid)
        if calibration.beta > 0.0 and fg_logit is None:
            raise ValueError(f"候选 {proposal_uid} 缺少 fg_logit")
        fg_value = float(fg_logit) if fg_logit is not None else 0.0
        q = calibrate_q(score, fg_value, coarse, calibration)
        record = dict(candidate)
        record["_gate_q"] = q
        record["_gate_coarse"] = coarse
        if q < calibration.tau_drop:
            removed.append(record)
        else:
            kept.append(record)
    return kept, removed


def summarize_gate(
    candidates: Sequence[Mapping[str, Any]],
    removed: Sequence[Mapping[str, Any]],
    *,
    coarse: str | None = None,
) -> dict[str, int]:
    """统计删除数量（按粗类过滤，便于逐类报告）。"""
    removed_count = 0
    total_count = 0
    for candidate in candidates:
        if coarse is not None and coarse_of_category_id(int(candidate["category_id"])) != coarse:
            continue
        total_count += 1
    for item in removed:
        if coarse is not None and str(item.get("_gate_coarse", "")) != coarse:
            continue
        removed_count += 1
    return {"total": total_count, "removed": removed_count, "kept": total_count - removed_count}


__all__ = [
    "ACTIVE_COARSE_FIRST_ROUND",
    "ALLOWED_MODES",
    "COARSE_CLASSES",
    "CONTEXT_EXPANSION",
    "INPUT_RESOLUTION",
    "MODE_S0",
    "MODE_S1",
    "MODE_S2",
    "GateCalibration",
    "apply_gate",
    "calibrate_q",
    "coarse_of_category_id",
    "expand_context_bbox",
    "make_calibration",
    "summarize_gate",
]
