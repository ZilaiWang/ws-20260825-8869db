from __future__ import annotations

import numpy as np
import torch

from rsdet.innovation.in_model_agreement import (
    AgreementResidualHead,
    apply_logit_residual,
    boxes_to_letterbox,
    letterbox_geometry,
    normalized_feature_anchor,
    pairwise_support_ranking_loss,
    proposal_metadata,
)
from scripts.train_in_model_dfine_agreement import _select_rows


def test_fresh_residual_head_is_exact_identity() -> None:
    head = AgreementResidualHead(12, hidden_dim=16)
    features = torch.randn(5, 12)
    scores = torch.tensor([0.01, 0.1, 0.5, 0.8, 0.99])
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]] * 5)
    metadata = proposal_metadata(scores, boxes, image_size=100)
    residual = head(features, metadata)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.allclose(apply_logit_residual(scores, residual, alpha=0.5), scores)


def test_letterbox_geometry_and_box_transform() -> None:
    geometry = letterbox_geometry(800, 400, 1024)
    assert geometry.scale == 1.28
    assert geometry.resized_width == 1024
    assert geometry.resized_height == 512
    assert geometry.pad_left == 0.0
    assert geometry.pad_top == 256.0
    boxes = boxes_to_letterbox(
        torch.tensor([[0.0, 0.0, 800.0, 400.0]]), geometry
    )
    assert torch.allclose(boxes, torch.tensor([[0.0, 256.0, 1024.0, 768.0]]))


def test_ranking_and_anchor_have_expected_direction() -> None:
    target = torch.tensor([0.8, 0.4, 0.0, 0.0])
    good = pairwise_support_ranking_loss(torch.tensor([2.0, 1.0, -1.0, -2.0]), target)
    bad = pairwise_support_ranking_loss(torch.tensor([-1.0, -2.0, 2.0, 1.0]), target)
    assert good < bad
    first = [torch.ones(1, 2, 3, 3), torch.ones(1, 4, 2, 2)]
    assert normalized_feature_anchor(first, first).item() == 0.0
    changed = [value + 0.5 for value in first]
    assert normalized_feature_anchor(changed, first).item() > 0.0


def test_row_selection_balances_supported_and_unsupported() -> None:
    grouped = _select_rows(
        np.asarray([7, 7, 7, 7, 8]),
        np.asarray([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32),
        np.asarray([0.0, 0.6, 0.4, 0.0, 0.0], dtype=np.float32),
        np.asarray([True, True, True, True, True]),
        max_per_image=2,
    )
    assert grouped[7].tolist() == [0, 1]
    assert grouped[8].tolist() == [4]


def test_row_selection_never_drops_priority_risk_rows_when_under_cap() -> None:
    grouped = _select_rows(
        np.asarray([7, 7, 7, 7]),
        np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        np.asarray([0.9, 0.8, 0.7, 0.0], dtype=np.float32),
        np.asarray([True, True, True, True]),
        max_per_image=3,
        priority=np.asarray([False, True, False, True]),
    )
    assert {1, 3}.issubset(set(grouped[7].tolist()))
