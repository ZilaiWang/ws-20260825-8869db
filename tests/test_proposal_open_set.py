from __future__ import annotations

import pytest

from rsdet.analysis.proposal_open_set import (
    OPEN_FOREGROUND,
    OPEN_IGNORE,
    OPEN_ORDINARY_BACKGROUND,
    OPEN_STRUCTURED_BACKGROUND,
    proposal_open_set_label,
)


def test_three_way_label_preserves_matchable_nonwinner() -> None:
    assert proposal_open_set_label(is_valid=True, fine_correct=True, crop_top1_class=2) == OPEN_FOREGROUND
    assert proposal_open_set_label(is_valid=False, fine_correct=True, crop_top1_class=2) == OPEN_IGNORE
    assert proposal_open_set_label(is_valid=False, fine_correct=False, crop_top1_class=2) == OPEN_STRUCTURED_BACKGROUND
    assert proposal_open_set_label(is_valid=False, fine_correct=False, crop_top1_class=10) == OPEN_ORDINARY_BACKGROUND


def test_open_set_head_contract() -> None:
    torch = pytest.importorskip("torch")
    from rsdet.innovation.proposal_open_set import ProposalOpenSetHead

    head = ProposalOpenSetHead(8, hidden_dim=16, dropout=0.0)
    logits = head(torch.zeros(4, 8), torch.ones(4, 8), torch.eye(3)[[0, 1, 2, 0]])
    assert logits.shape == (4, 3)
    with pytest.raises(ValueError):
        head(torch.zeros(4, 7), torch.ones(4, 8), torch.zeros(4, 3))
