"""N2-CFG 前景门控核心逻辑测试（纯 CPU，无 torch 依赖）。"""

from __future__ import annotations

import pytest

from rsdet.analysis.background_gate import (
    ACTIVE_COARSE_FIRST_ROUND,
    COARSE_CLASSES,
    MODE_S0,
    MODE_S1,
    MODE_S2,
    apply_gate,
    calibrate_q,
    coarse_of_category_id,
    expand_context_bbox,
    make_calibration,
)


def test_coarse_mapping_boundaries() -> None:
    assert coarse_of_category_id(0) == "ship"
    assert coarse_of_category_id(3) == "ship"
    assert coarse_of_category_id(4) == "aircraft"
    assert coarse_of_category_id(23) == "aircraft"
    assert coarse_of_category_id(24) == "vehicle"


def test_coarse_mapping_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        coarse_of_category_id(-1)
    with pytest.raises(ValueError):
        coarse_of_category_id(25)


def test_context_expansion_square_centered() -> None:
    # GT 长边 109 -> side 136.25，中心 (239.5, 121) 不变。
    box = (185.0, 72.0, 294.0, 170.0)
    result = expand_context_bbox(box, ratio=1.25)
    assert result == pytest.approx((171.375, 52.875, 307.625, 189.125))
    center_x = (result[0] + result[2]) / 2.0
    center_y = (result[1] + result[3]) / 2.0
    assert center_x == pytest.approx(239.5)
    assert center_y == pytest.approx(121.0)
    assert result[2] - result[0] == pytest.approx(result[3] - result[1])


def test_context_expansion_rejects_bad_ratio() -> None:
    assert expand_context_bbox((0, 0, 10, 20), ratio=1.0) == (0.0, 0.0, 10.0, 20.0)
    with pytest.raises(ValueError):
        expand_context_bbox((0, 0, 10, 10), ratio=0.5)


def test_calibrate_q_monotonic_in_score() -> None:
    calib = make_calibration(mode=MODE_S2, alpha=1.0, beta=1.0, tau_drop=0.5, gamma=0.0)
    low = calibrate_q(0.2, 0.0, "ship", calib)
    high = calibrate_q(0.8, 0.0, "ship", calib)
    assert high > low


def test_calibrate_q_monotonic_in_fg_logit() -> None:
    calib = make_calibration(mode=MODE_S2, alpha=1.0, beta=1.0, tau_drop=0.5, gamma=0.0)
    low = calibrate_q(0.5, -2.0, "vehicle", calib)
    high = calibrate_q(0.5, 2.0, "vehicle", calib)
    assert high > low


def test_calibrate_q_in_unit_interval() -> None:
    calib = make_calibration(mode=MODE_S2, alpha=2.0, beta=2.0, tau_drop=0.5, gamma=3.0)
    for score in (0.01, 0.5, 0.99):
        for fg in (-5.0, 0.0, 5.0):
            q = calibrate_q(score, fg, "aircraft", calib)
            assert 0.0 <= q <= 1.0


def test_s0_s1_require_shared_gamma() -> None:
    with pytest.raises(ValueError):
        make_calibration(
            mode=MODE_S0,
            alpha=1.0,
            beta=0.0,
            tau_drop=0.5,
            gamma={"ship": 0.0, "aircraft": 0.0, "vehicle": 1.0},
        )
    # 单一浮点应自动扩展为三者相等。
    calib = make_calibration(mode=MODE_S1, alpha=1.0, beta=1.0, tau_drop=0.5, gamma=0.3)
    assert set(calib.gamma.values()) == {0.3}


def test_s2_allows_coarse_gamma() -> None:
    calib = make_calibration(
        mode=MODE_S2,
        alpha=1.0,
        beta=1.0,
        tau_drop=0.5,
        gamma={"ship": -1.0, "aircraft": 0.0, "vehicle": 1.0},
    )
    assert calib.gamma["ship"] == -1.0
    assert calib.gamma["vehicle"] == 1.0


def test_calibration_rejects_negative_slopes() -> None:
    with pytest.raises(ValueError):
        make_calibration(mode=MODE_S2, alpha=-0.5, beta=1.0, tau_drop=0.5, gamma=0.0)
    with pytest.raises(ValueError):
        make_calibration(mode=MODE_S2, alpha=1.0, beta=-1.0, tau_drop=0.5, gamma=0.0)


def test_apply_gate_bypasses_aircraft() -> None:
    calib = make_calibration(mode=MODE_S2, alpha=1.0, beta=1.0, tau_drop=0.99, gamma=-5.0)
    candidates = [
        {"proposal_uid": "a1", "category_id": 4, "score": 0.5},  # aircraft
        {"proposal_uid": "s1", "category_id": 0, "score": 0.5},  # ship
    ]
    fg_logits = {"a1": -10.0, "s1": -10.0}
    kept, removed = apply_gate(candidates, fg_logits, calib)
    # 飞机无论多像背景都旁路保留；舰船在极低 fg 下被删除。
    assert {c["proposal_uid"] for c in kept} == {"a1"}
    assert {c["proposal_uid"] for c in removed} == {"s1"}


def test_apply_gate_requires_fg_when_beta_positive() -> None:
    calib = make_calibration(mode=MODE_S2, alpha=1.0, beta=1.0, tau_drop=0.5, gamma=0.0)
    candidates = [{"proposal_uid": "s1", "category_id": 0, "score": 0.5}]
    with pytest.raises(ValueError):
        apply_gate(candidates, {}, calib)


def test_s0_does_not_need_fg() -> None:
    calib = make_calibration(mode=MODE_S0, alpha=1.0, beta=0.0, tau_drop=0.5, gamma=0.0)
    candidates = [
        {"proposal_uid": "s1", "category_id": 0, "score": 0.1},
        {"proposal_uid": "s2", "category_id": 0, "score": 0.9},
    ]
    kept, removed = apply_gate(candidates, {}, calib)
    # S0 等价于 score 阈值：低分删，高分留。
    assert {c["proposal_uid"] for c in kept} == {"s2"}
    assert {c["proposal_uid"] for c in removed} == {"s1"}


def test_active_coarse_first_round() -> None:
    assert ACTIVE_COARSE_FIRST_ROUND == frozenset({"ship", "vehicle"})
    assert set(COARSE_CLASSES) == {"ship", "aircraft", "vehicle"}
