from __future__ import annotations

from scripts.extract_pseudo_candidate_dino_features import _xywh_to_xyxy


def test_xywh_to_xyxy() -> None:
    assert _xywh_to_xyxy([2.0, 3.0, 5.0, 7.0]) == (2.0, 3.0, 7.0, 10.0)
