from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from rsdet.innovation.quality_aware_loss import quality_aware_classification_loss


def test_quality_aware_loss_matches_frozen_elementwise_formula() -> None:
    logits = torch.tensor([[[-2.0, 0.5], [1.0, -0.25]]], requires_grad=True)
    targets = torch.tensor([[[0.0, 0.8], [0.4, 0.0]]])
    alpha = 0.75
    gamma = 2.0
    module = quality_aware_classification_loss(alpha=alpha, gamma=gamma)
    actual = module(logits, targets)
    labels = targets.gt(0).to(logits.dtype)
    expected_weights = (
        alpha * logits.sigmoid().pow(gamma) * (1.0 - labels) + targets * labels
    )
    expected = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ) * expected_weights
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0)
    actual.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_quality_aware_loss_downweights_easy_negative_more_than_hard_negative() -> None:
    module = quality_aware_classification_loss(alpha=0.75, gamma=2.0)
    logits = torch.tensor([[[-5.0], [2.0]]])
    targets = torch.zeros_like(logits)
    values = module(logits, targets).flatten()
    assert values[0] < values[1]


@pytest.mark.parametrize(
    ("alpha", "gamma"),
    [(-0.01, 2.0), (1.01, 2.0), (0.75, -0.01)],
)
def test_quality_aware_loss_rejects_invalid_hyperparameters(
    alpha: float, gamma: float
) -> None:
    with pytest.raises(ValueError):
        quality_aware_classification_loss(alpha=alpha, gamma=gamma)


def test_quality_aware_loss_rejects_invalid_positive_weighting() -> None:
    with pytest.raises(ValueError, match="positive_weighting"):
        quality_aware_classification_loss(positive_weighting="bad")


def test_unit_positive_weighting_preserves_soft_positive_bce_strength() -> None:
    logits = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([[0.25, 0.0]])
    quality = quality_aware_classification_loss(positive_weighting="quality")(
        logits, targets
    )
    unit = quality_aware_classification_loss(positive_weighting="unit")(
        logits, targets
    )
    assert torch.isclose(unit[0, 0], quality[0, 0] * 4.0)
    assert torch.isclose(unit[0, 1], quality[0, 1])


def test_focused_loss_leaves_nonfocused_classes_as_plain_bce() -> None:
    logits = torch.tensor([[1.2, -0.7, 0.3]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 0.6, 0.0]], dtype=torch.float32)
    actual = quality_aware_classification_loss(
        positive_weighting="unit", focus_class_indices=(0, 2)
    )(logits, targets)
    plain = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    assert actual[0, 1].item() == pytest.approx(plain[0, 1].item())
    assert actual[0, 0].item() < plain[0, 0].item()
    assert actual[0, 2].item() < plain[0, 2].item()


def test_focused_loss_rejects_out_of_range_class_at_runtime() -> None:
    module = quality_aware_classification_loss(focus_class_indices=(3,))
    with pytest.raises(ValueError, match="exceeds"):
        module(torch.zeros(1, 3), torch.zeros(1, 3))


def test_selective_trainer_rejects_partial_projection_contract() -> None:
    from rsdet.innovation.quality_aware_loss import selective_classifier_trainer

    class Base:
        pass

    with pytest.raises(ValueError, match="set together"):
        selective_classifier_trainer(
            focus_class_indices=(0,),
            base_trainer=Base,
            max_weight_relative_delta=0.05,
        )


@pytest.mark.parametrize(
    ("branch_delta", "weight_delta", "bias_delta"),
    [(0.0, 0.05, 0.25), (0.01, 0.0, 0.25), (0.01, 0.05, 0.0)],
)
def test_spatial_residual_trainer_rejects_nonpositive_bounds(
    branch_delta: float, weight_delta: float, bias_delta: float
) -> None:
    from rsdet.innovation.quality_aware_loss import (
        spatial_classifier_residual_trainer,
    )

    class Base:
        pass

    with pytest.raises(ValueError):
        spatial_classifier_residual_trainer(
            focus_class_indices=(0,),
            base_trainer=Base,
            max_branch_relative_delta=branch_delta,
            max_final_weight_relative_delta=weight_delta,
            max_final_bias_delta=bias_delta,
        )
