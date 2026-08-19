"""N2-CFG 校准器拟合与 S0/S1/S2 差异测试（纯 CPU）。"""

from __future__ import annotations

from scripts.evaluate_bg_gate import FitSample, fit_gate_calibration


def _samples() -> list[FitSample]:
    """构造一个可区分的拟合集。

    背景 FP_BG 有两种：低分背景（score 低，S0 能删）与高分背景
    （score 高但 fg_logit 极低，只有 S1/S2 能删）。真目标 TP 分数高、
    fg_logit 高，不应被删。
    """
    samples: list[FitSample] = []
    # 10 个低分背景（S0 就能删）。
    for index in range(10):
        samples.append(
            FitSample(f"bg_low_{index}", 0.1, 0.0, "ship", False, True)
        )
    # 10 个高分背景（fg 证据才能删）。
    for index in range(10):
        samples.append(
            FitSample(f"bg_high_{index}", 0.9, -5.0, "ship", False, True)
        )
    # 20 个真目标（不可删）。
    for index in range(20):
        samples.append(
            FitSample(f"tp_{index}", 0.7, 3.0, "ship", True, False)
        )
    return samples


def test_s0_cannot_remove_high_score_background() -> None:
    samples = _samples()
    calib, stats = fit_gate_calibration(samples, mode="S0", recall_budget=0)
    # S0 只看 score：高分背景（score=0.9）排在高分真目标之后，预算为 0 时
    # 只能删低分背景，删不到高分背景。
    assert stats["removed_fp_bg"] < 20
    assert stats["removed_tp"] == 0


def test_s2_removes_high_score_background_without_tp_loss() -> None:
    samples = _samples()
    calib, stats = fit_gate_calibration(samples, mode="S2", recall_budget=0)
    # S2 用 coarse fg 证据，能删高分背景（fg=-5），且不删真目标（fg=3）。
    assert stats["removed_fp_bg"] == 20
    assert stats["removed_tp"] == 0


def test_fit_respects_recall_budget() -> None:
    samples = _samples()
    # 预算 1：允许删 1 个 TP。
    calib, stats = fit_gate_calibration(samples, mode="S2", recall_budget=1)
    assert stats["removed_tp"] <= 1


def test_s1_matches_s2_on_shared_gamma() -> None:
    samples = _samples()
    # S1 用 shared head，gamma 单一；该拟合集只有 ship 粗类，S1 与 S2 应能
    # 删除相同的高分背景。
    _, s1 = fit_gate_calibration(samples, mode="S1", recall_budget=0)
    _, s2 = fit_gate_calibration(samples, mode="S2", recall_budget=0)
    assert s1["removed_fp_bg"] == s2["removed_fp_bg"] == 20
