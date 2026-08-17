"""run_sealed_admission 的单元测试（合成小 evaluate JSON 验证门禁判定与 bootstrap）。"""

from __future__ import annotations

from pathlib import Path

from scripts.run_sealed_admission import (
    _applied_fp_bg_removed,
    _paired_bootstrap,
    run_admission,
)


def _make_evaluate(
    mode: str,
    removed_fp_bg_per_fold: list[int],
    *,
    removed_tp_per_fold: list[int] | None = None,
    heldout_fp_bg: int = 100,
    fp_bg_groups: list[str] | None = None,
) -> dict:
    """构造最小 evaluate JSON。"""
    removed_tp_per_fold = removed_tp_per_fold or [0, 0, 0]
    per_fold = []
    for f, (n_fp, n_tp) in enumerate(zip(removed_fp_bg_per_fold, removed_tp_per_fold)):
        fp_by_coarse = {"ship": 0, "vehicle": 0, "aircraft": 0}
        # 简化：所有删除归入 ship/vehicle，验证逐类门禁
        if n_fp >= 20:
            fp_by_coarse["ship"] = n_fp - 10
            fp_by_coarse["vehicle"] = 10
        else:
            fp_by_coarse["ship"] = n_fp
        applied_removed = []
        groups = fp_bg_groups or [f"g{i}" for i in range(5)]
        for i in range(n_fp):
            applied_removed.append(
                {
                    "proposal_uid": f"p{f}-{i}",
                    "image_id": f + 1,
                    "category_id": 0 if i % 2 else 24,
                    "coarse": "ship" if i % 2 else "vehicle",
                    "source_group": groups[i % len(groups)],
                    "score": 0.5,
                    "q": 0.1,
                    "is_tp": False,
                    "is_fp_bg": True,
                }
            )
        per_fold.append(
            {
                "held_out_fold": f,
                "fit": {"removed_fp_bg": n_fp, "removed_tp": n_tp},
                "heldout_counts": {
                    "fp_bg": heldout_fp_bg,
                    "tp": 10,
                    "fp_bg_by_coarse": {"ship": heldout_fp_bg // 2, "vehicle": heldout_fp_bg // 4, "aircraft": 0},
                    "tp_by_coarse": {"ship": 5, "vehicle": 3, "aircraft": 2},
                },
                "applied_removed": applied_removed,
            }
        )
    return {"mode": mode, "per_fold": per_fold}


def test_paired_bootstrap_ci_covers_point() -> None:
    """bootstrap 的 CI 应覆盖点估计，且一致 delta 时 CI 下界 > 0。"""
    delta = {f"g{i}": 2.0 for i in range(20)}
    out = _paired_bootstrap(delta, n_bootstrap=500, seed=202625)
    assert out["n_groups"] == 20
    assert out["point"] == 40.0
    assert out["ci_low"] > 0  # 全正 delta → 下界 > 0


def test_admission_pass_when_s2_clearly_better() -> None:
    """S2 大幅删除（>10%）、逐类达标、fold 同向、来源分散、零 TP → overall PASS。"""
    s0 = _make_evaluate("S0", [20, 20, 20])
    s1 = _make_evaluate("S1", [25, 25, 25])
    s2 = _make_evaluate("S2", [60, 60, 60], heldout_fp_bg=300, fp_bg_groups=[f"g{i}" for i in range(10)])
    payload = run_admission(
        s0, s1, s2, recall_budget=0, output=Path("/tmp/sealed_admission_pass.json")
    )
    assert payload["overall"]["pass"] is True
    gates = payload["gates"]
    assert gates["g1_pooled_fp_bg_reduction"]["pass"] is True
    assert gates["g3_fold_consistency"]["pass"] is True
    assert gates["g4_source_concentration"]["pass"] is True
    assert gates["g5_paired_bootstrap"]["pass"] is True
    assert gates["g7_zero_tp_loss"]["pass"] is True


def test_admission_fail_when_s2_not_better() -> None:
    """S2 与 S0 无差异（甚至更差）→ overall FAIL。"""
    s0 = _make_evaluate("S0", [30, 30, 30])
    s1 = _make_evaluate("S1", [30, 30, 30])
    s2 = _make_evaluate("S2", [30, 30, 30])
    payload = run_admission(
        s0, s1, s2, recall_budget=0, output=Path("/tmp/sealed_admission_fail.json")
    )
    assert payload["overall"]["pass"] is False
    gates = payload["gates"]
    # delta = 0 → bootstrap CI 包含 0，不显著
    assert gates["g5_paired_bootstrap"]["pass"] is False
    # fold 同向：S2 == S0（非 >）→ 0 个 fold 同向
    assert gates["g3_fold_consistency"]["consistent_folds"] == 0


def test_admission_fail_source_concentration() -> None:
    """单一 source group 删除占比 >40% → G4 FAIL。"""
    # 60 个删除全部来自 1 个 group → share=1.0 > 0.40
    s0 = _make_evaluate("S0", [10, 10, 10])
    s1 = _make_evaluate("S1", [10, 10, 10])
    s2 = _make_evaluate("S2", [60, 60, 60], heldout_fp_bg=300, fp_bg_groups=["g0"])
    payload = run_admission(
        s0, s1, s2, recall_budget=0, output=Path("/tmp/sealed_admission_conc.json")
    )
    gates = payload["gates"]
    assert gates["g4_source_concentration"]["pass"] is False
    assert gates["g4_source_concentration"]["max_share"] == 1.0
    assert payload["overall"]["pass"] is False


def test_admission_fail_tp_loss() -> None:
    """S2 删除 TP → G7 FAIL。"""
    s0 = _make_evaluate("S0", [20, 20, 20])
    s1 = _make_evaluate("S1", [20, 20, 20])
    s2 = _make_evaluate("S2", [60, 60, 60], removed_tp_per_fold=[1, 0, 0], heldout_fp_bg=300)
    payload = run_admission(
        s0, s1, s2, recall_budget=0, output=Path("/tmp/sealed_admission_tp.json")
    )
    gates = payload["gates"]
    assert gates["g7_zero_tp_loss"]["pass"] is False
    assert payload["overall"]["pass"] is False


def test_applied_fp_bg_removed_counts() -> None:
    """applied 删除统计正确。"""
    data = _make_evaluate("S2", [5, 7, 9])
    assert _applied_fp_bg_removed(data) == 21
