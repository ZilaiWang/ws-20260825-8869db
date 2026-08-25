from __future__ import annotations

import pytest

from scripts.b_stage_source_rerank import size_bin, source_family


@pytest.mark.parametrize(
    ("group_id", "expected"),
    [
        ("mar20-airport-proxy-000", "aircraft_source_family"),
        ("mar20-airport-proxy-059", "aircraft_source_family"),
        ("scene_L00000010882", "ship_source_family"),
        ("site_001", "vehicle_source_family"),
        ("unexpected", "unknown_source_family"),
    ],
)
def test_source_family_is_the_documented_three_family_collapse(
    group_id: str,
    expected: str,
) -> None:
    assert source_family(group_id) == expected


@pytest.mark.parametrize(
    ("short_edge", "expected"),
    [
        (0.0, "<32"),
        (31.999, "<32"),
        (32.0, "32-64"),
        (64.0, "64-128"),
        (128.0, "128-256"),
        (256.0, ">=256"),
    ],
)
def test_size_bin_boundaries_are_frozen(short_edge: float, expected: str) -> None:
    assert size_bin(short_edge) == expected
