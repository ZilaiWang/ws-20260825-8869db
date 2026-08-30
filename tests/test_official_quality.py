from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rsdet.innovation.group_dro import GroupDROController  # noqa: E402
from rsdet.innovation.official_quality import (  # noqa: E402
    OfficialMatchQualityHead,
    active_pair_rank_loss,
    official_iou_thresholds,
    official_quality_loss,
    soft_official_match_target,
)


def test_vehicle_uses_lower_official_iou_threshold() -> None:
    coarse = torch.tensor([0, 1, 2])
    thresholds = official_iou_thresholds(coarse)
    assert torch.allclose(thresholds, torch.tensor([0.50, 0.50, 0.35]))
    targets = soft_official_match_target(torch.tensor([0.50, 0.50, 0.35]), coarse)
    assert torch.allclose(targets, torch.full((3,), 0.5), atol=1e-6)


def test_quality_head_is_identity_initialized() -> None:
    torch.manual_seed(0)
    head = OfficialMatchQualityHead(8, hidden_dim=32, dropout=0.0)
    features = torch.randn(4, 8)
    score = torch.tensor([0.05, 0.15, 0.40, 0.80])
    output = head(features, score)
    assert torch.allclose(output.residual, torch.zeros_like(output.residual), atol=1e-7)
    assert torch.allclose(torch.sigmoid(output.final_logit), score, atol=1e-6)


def test_active_pair_loss_compares_tp_to_background_fp() -> None:
    final_logit = torch.tensor([0.0, 1.0], requires_grad=True)
    detector_score = torch.tensor([0.15, 0.20])
    protected = torch.tensor([1, 0])
    active_fp = torch.tensor([0, 1])
    coarse = torch.tensor([2, 2])
    groups = torch.tensor([3, 3])
    loss = active_pair_rank_loss(
        final_logit,
        detector_score,
        protected,
        active_fp,
        coarse,
        groups,
    )
    assert float(loss.detach()) > 1.0
    loss.backward()
    assert final_logit.grad is not None
    assert float(final_logit.grad[0]) < 0.0
    assert float(final_logit.grad[1]) > 0.0


def test_active_pair_loss_does_not_silently_cross_groups() -> None:
    final_logit = torch.tensor([0.0, 1.0], requires_grad=True)
    detector_score = torch.tensor([0.15, 0.20])
    protected = torch.tensor([1, 0])
    active_fp = torch.tensor([0, 1])
    coarse = torch.tensor([2, 2])
    groups = torch.tensor([3, 4])
    strict = active_pair_rank_loss(
        final_logit, detector_score, protected, active_fp, coarse, groups
    )
    relaxed = active_pair_rank_loss(
        final_logit,
        detector_score,
        protected,
        active_fp,
        coarse,
        groups,
        relax_group_if_empty=True,
    )
    assert float(strict.detach()) == 0.0
    assert float(relaxed.detach()) > 1.0


def test_final_quality_supervision_reaches_identity_initialized_residual() -> None:
    torch.manual_seed(1)
    head = OfficialMatchQualityHead(8, hidden_dim=32, dropout=0.0)
    features = torch.randn(4, 8)
    score = torch.tensor([0.05, 0.15, 0.40, 0.80])
    output = head(features, score)
    losses = official_quality_loss(
        output,
        detector_score=score,
        soft_match_target=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        protected_tp=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        active_fp=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        active_mask=torch.tensor([0.0, 0.0, 1.0, 1.0]),
        coarse_ids=torch.tensor([0, 0, 2, 2]),
        group_ids=torch.tensor([0, 0, 1, 1]),
        rank_enabled=False,
    )
    losses["total"].backward()
    gradient = head.residual_head[-1].weight.grad
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0.0


def test_group_dro_increases_weight_on_hard_group() -> None:
    controller = GroupDROController(2, step_size=0.5)
    result = controller(
        torch.tensor([0.1, 0.1, 2.0, 2.0]),
        torch.tensor([0, 0, 1, 1]),
        update=True,
    )
    assert float(result.group_weights[1]) > float(result.group_weights[0])
