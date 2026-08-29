from __future__ import annotations

import pytest

from scripts.classify_coarse_proposals_with_p03 import allowed_fine_ids


def test_allowed_fine_ids() -> None:
    assert allowed_fine_ids(0) == tuple(range(4))
    assert allowed_fine_ids(4) == tuple(range(4, 24))
    assert allowed_fine_ids(24) == (24,)
    with pytest.raises(ValueError):
        allowed_fine_ids(1)
