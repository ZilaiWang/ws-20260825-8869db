"""Shape and routing tests for the gated Y3 sampling pair."""

from __future__ import annotations

import torch

from rsdet.models.ibs_sampling import (
    IBSDown,
    IBSUp,
    _channels_to_space,
    _space_to_channels,
)


def test_spatial_reorganization_is_exactly_invertible() -> None:
    source = torch.arange(2 * 8 * 6 * 10, dtype=torch.float32).view(2, 8, 6, 10)
    packed = _space_to_channels(source, 2)
    restored = _channels_to_space(packed, 2)
    assert packed.shape == (2, 32, 3, 5)
    assert torch.equal(restored, source)


def test_ibs_pair_preserves_p2_neck_shapes() -> None:
    up = IBSUp(128, 128, factor=2, expansion_ratio=2)
    down = IBSDown(64, 64, factor=2, expansion_ratio=2)
    assert up(torch.randn(2, 128, 16, 16)).shape == (2, 128, 32, 32)
    assert down(torch.randn(2, 64, 32, 32)).shape == (2, 64, 16, 16)
