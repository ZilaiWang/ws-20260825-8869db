"""方案6 §十一 工程不变量测试。

覆盖：
1.  deploy 包禁止读 GT
2.  detector 预测严格 OOF
3.  router 预测嵌套 OOF
4.  候选数不增加不变量
5.  max 聚合有界（防求和聚合候选爆炸再犯）
6.  动作变换幂等
7.  快/慢 frontier scorer 一致
8.  baseline 确定性重放
9.  tie-break 确定性
10. scene/机场不跨折

数据依赖：涉及真实 manifest 的测试（7/10）使用项目本地已冻结的
``outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv``；若缺失则跳过，
不阻塞核心不变量测试。
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

from scope_router.actions import (  # noqa: E402
    Action,
    ActionKind,
    Candidate,
    apply_action,
    assert_candidate_count_invariant,
)
from scope_router.calibration import GroupwiseConformalLCB  # noqa: E402
from scope_router.decode import ProposedAction, safe_greedy_decode  # noqa: E402
from scope_router.provenance import OOFRecord, validate_cross_fit  # noqa: E402


# --------------------------------------------------------------------------
# 1. deploy 包禁止读 GT
# --------------------------------------------------------------------------
DEPLOY_MODULES = [
    "actions.py",
    "decode.py",
    "features.py",
    "calibration.py",
    "model.py",
    "losses.py",
    "__init__.py",
]
GT_MODULES = {
    "rsdet.scope.official_scorer",   # FrontierScorer 依赖 GT boxes
    "rsdet.analysis.oof_detection",  # load_formal_ground_truth
    "scope_router.counterfactual",   # CounterfactualLabelBuilder 依赖 GT
}


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_deploy_cannot_read_ground_truth() -> None:
    for mod in DEPLOY_MODULES:
        path = ROOT / "scope_router" / mod
        if not path.exists():
            continue
        imported = _imports_of(path)
        leaked = imported & GT_MODULES
        assert not leaked, f"deploy 模块 {mod} import 了 GT 相关模块: {leaked}"


def test_counterfactual_is_offline_only() -> None:
    """counterfactual.py 是离线标签模块，deploy 包不得 import 它。"""
    for mod in DEPLOY_MODULES:
        path = ROOT / "scope_router" / mod
        if not path.exists():
            continue
        assert "scope_router.counterfactual" not in _imports_of(path), (
            f"deploy 模块 {mod} 不应 import 离线 counterfactual"
        )


# --------------------------------------------------------------------------
# 2/3. 严格 OOF 与嵌套 OOF
# --------------------------------------------------------------------------
def test_detector_prediction_is_strict_oof() -> None:
    ok = OOFRecord(
        sample_id="s1",
        group_id="airport-A",
        detector_model_id="d0",
        detector_train_groups=frozenset({"airport-B", "airport-C"}),
    )
    validate_cross_fit([ok])  # 不抛异常

    bad = OOFRecord(
        sample_id="s2",
        group_id="airport-A",
        detector_model_id="d0",
        detector_train_groups=frozenset({"airport-A"}),
    )
    with pytest.raises(AssertionError):
        validate_cross_fit([bad])


def test_router_prediction_is_nested_oof() -> None:
    ok = OOFRecord(
        sample_id="s1",
        group_id="airport-A",
        detector_model_id="d0",
        detector_train_groups=frozenset({"airport-B"}),
        router_train_groups=frozenset({"airport-B"}),
    )
    validate_cross_fit([ok], require_router=True)

    bad = OOFRecord(
        sample_id="s2",
        group_id="airport-A",
        detector_model_id="d0",
        detector_train_groups=frozenset({"airport-B"}),
        router_train_groups=frozenset({"airport-A"}),
    )
    with pytest.raises(AssertionError):
        validate_cross_fit([bad], require_router=True)


# --------------------------------------------------------------------------
# 4. 候选数不变量
# --------------------------------------------------------------------------
def _cand(cid: str, cls_id: int = 1, score: float = 0.5) -> Candidate:
    return Candidate(cid, (0.0, 0.0, 10.0, 10.0), cls_id, score)


def test_candidate_count_never_increases() -> None:
    baseline = [_cand("a"), _cand("b"), _cand("c")]
    # DROP / RELABEL 不增加候选
    out = apply_action(baseline, "a", Action(ActionKind.DROP))
    assert_candidate_count_invariant(baseline, out, allow_drop=True)
    assert len(out) == 2

    out = apply_action(baseline, "a", Action(ActionKind.RELABEL, cls_id=3))
    assert_candidate_count_invariant(baseline, out, allow_drop=False)
    assert len(out) == 3

    # 扩展被禁止
    with pytest.raises(AssertionError):
        assert_candidate_count_invariant(
            baseline, baseline + [_cand("d")], allow_drop=True
        )


def test_decoder_never_expands_candidates() -> None:
    baseline = [_cand("a"), _cand("b")]
    proposals = [
        ProposedAction("a", Action(ActionKind.RELABEL, cls_id=2), 0.1, 0.2),
        ProposedAction("b", Action(ActionKind.DROP), 0.05, 0.08),
    ]
    output, selected = safe_greedy_decode(baseline, proposals)
    assert len(output) <= len(baseline)
    assert len(selected) == 2


# --------------------------------------------------------------------------
# 5. max 聚合有界
# --------------------------------------------------------------------------
def test_max_aggregation_is_bounded() -> None:
    torch = pytest.importorskip("torch")
    from rsdet.innovation.aggregation import aggregate_group_scores

    torch.manual_seed(0)
    logits = torch.randn(3, 5, 25)
    groups = [[0, 1, 2, 3], list(range(4, 24)), [24]]

    out = aggregate_group_scores(logits, groups)
    assert out.shape == (3, 5, 3)

    # 等于组内 max
    for g, idx in enumerate(groups):
        expected = logits[..., idx].max(dim=-1).values
        assert torch.allclose(out[..., g], expected)

    # 有界：max <= 求和（求和聚合是被禁止的候选爆炸源）
    sum_out = torch.stack(
        [logits[..., idx].sum(dim=-1) for idx in groups], dim=-1
    )
    assert torch.all(out <= sum_out + 1e-6)

    # 求和聚合被明确拒绝
    with pytest.raises(ValueError):
        aggregate_group_scores(logits, groups, reduction="sum")  # type: ignore[arg-type]

    # 越界 / 重复索引被拒绝
    with pytest.raises(ValueError):
        aggregate_group_scores(logits, [[0, 25]])
    with pytest.raises(ValueError):
        aggregate_group_scores(logits, [[0, 1], [1, 2]])


# --------------------------------------------------------------------------
# 6. 动作变换幂等
# --------------------------------------------------------------------------
def test_keep_is_idempotent() -> None:
    cands = [_cand("a")]
    once = apply_action(cands, "a", Action(ActionKind.KEEP))
    twice = apply_action(once, "a", Action(ActionKind.KEEP))
    assert once == cands
    assert twice == cands


# --------------------------------------------------------------------------
# 7/8/9. frontier scorer 一致性 + 确定性
# --------------------------------------------------------------------------
def _build_proto():
    from rsdet.evaluation.protocol import parse_evaluation_protocol
    from rsdet.utils.config import load_config

    return parse_evaluation_protocol(load_config(str(ROOT / "configs/project.yaml")))


def _synthetic_data():
    from rsdet.scope.official_scorer import CandidateView

    # 单 image，3 候选 + 1 GT
    gt_boxes = {1: [{"category_id": 5, "bbox_xyxy": [0.0, 0.0, 100.0, 100.0]}]}
    image_ids = {1}
    cands = [
        CandidateView(0, 1, 5, 0.9, [0.0, 0.0, 100.0, 100.0]),   # 命中 GT
        CandidateView(1, 1, 5, 0.8, [2.0, 2.0, 98.0, 98.0]),     # 冗余同类
        CandidateView(2, 1, 6, 0.5, [200.0, 200.0, 300.0, 300.0]),  # 错类
    ]
    return gt_boxes, image_ids, cands


def test_fast_metric_delta_matches_full_scorer() -> None:
    from rsdet.scope.incremental_scorer import IncrementalFrontierScorer
    from rsdet.scope.official_scorer import CandidateView, FrontierScorer

    proto = _build_proto()
    gt_boxes, image_ids, cands = _synthetic_data()

    full = FrontierScorer(proto, gt_boxes, image_ids)
    inc = IncrementalFrontierScorer(proto, gt_boxes, image_ids, cands)

    # base 一致
    assert abs(full.score(cands) - inc.score()) < 1e-9

    # drop 候选 1
    d_full = full.score([c for c in cands if c._idx != 1])
    d_inc, _, _ = inc.score_after(1, "drop")
    assert abs(d_full - d_inc) < 1e-9

    # relabel 候选 2 -> 5
    relabeled = [cands[0], cands[1], CandidateView(2, 1, 5, 0.5, [200.0, 200.0, 300.0, 300.0])]
    r_full = full.score(relabeled)
    r_inc, _, _ = inc.score_after(2, "relabel", 5)
    assert abs(r_full - r_inc) < 1e-9


def test_baseline_replay_is_deterministic() -> None:
    from rsdet.scope.official_scorer import FrontierScorer

    proto = _build_proto()
    gt_boxes, image_ids, cands = _synthetic_data()
    scorer = FrontierScorer(proto, gt_boxes, image_ids)

    r1 = scorer.frontier(cands).recall_at_fdr[0.12]
    r2 = scorer.frontier(cands).recall_at_fdr[0.12]
    assert r1 == r2  # 确定性重放


def test_tie_breaking_is_deterministic() -> None:
    from rsdet.scope.official_scorer import CandidateView, FrontierScorer

    proto = _build_proto()
    gt_boxes = {1: [{"category_id": 5, "bbox_xyxy": [0.0, 0.0, 100.0, 100.0]}]}
    image_ids = {1}
    # 相同分数、重叠的同类候选：NMS 的 tie-break 必须确定
    cands = [
        CandidateView(0, 1, 5, 0.9, [0.0, 0.0, 100.0, 100.0]),
        CandidateView(1, 1, 5, 0.9, [0.0, 0.0, 100.0, 100.0]),
    ]
    scorer = FrontierScorer(proto, gt_boxes, image_ids)
    r1 = scorer.frontier(cands).recall_at_fdr[0.12]
    r2 = scorer.frontier(list(reversed(cands))).recall_at_fdr[0.12]
    assert r1 == r2


# --------------------------------------------------------------------------
# 10. scene/机场不跨折
# --------------------------------------------------------------------------
FORMAL_MANIFEST = ROOT / "outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv"


def test_no_scene_or_near_duplicate_crosses_folds() -> None:
    if not FORMAL_MANIFEST.exists():
        pytest.skip("formal_crop_manifest.csv 不在本地")

    from rsdet.analysis.oof_detection import load_formal_ground_truth

    formal = load_formal_ground_truth(str(FORMAL_MANIFEST))
    group_fold: dict[str, int] = {}
    for (img, _), o in formal.objects.items():
        if o.group_id in group_fold:
            assert group_fold[o.group_id] == o.fold, (
                f"机场 {o.group_id} 跨越了多个 fold（{group_fold[o.group_id]} vs {o.fold}）"
            )
        else:
            group_fold[o.group_id] = o.fold


# --------------------------------------------------------------------------
# 附：conformal LCB 保守性（scaffold 原测试保留）
# --------------------------------------------------------------------------
def test_conformal_lcb_is_more_conservative() -> None:
    pred = np.linspace(-0.1, 0.2, 100)
    truth = pred - 0.03
    cal = GroupwiseConformalLCB(alpha=0.1).fit(pred, truth)
    corrected = cal.transform(pred)
    assert np.all(corrected <= pred + 1e-12)
