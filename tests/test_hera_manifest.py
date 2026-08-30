import math

import pytest

from rsdet.hera_guard.manifest import (
    PAV_METADATA_COLUMNS,
    metadata_from_node,
    square_crop_box,
)


def test_square_crop_preserves_center_and_requested_scale() -> None:
    crop = square_crop_box([0, 10, 20, 20], scale=1.5)
    assert crop == pytest.approx([-5, 0, 25, 30])


def test_metadata_contract_is_finite_and_ordered() -> None:
    node = {
        "y5_score": 0.4,
        "crop_top1": 0.8,
        "crop_margin": 0.6,
        "crop_entropy": 1.0,
        "detector_crop_agree": 1,
        "short_edge": 10,
        "area": 200,
        "aspect": 2,
        "local_density": 4,
    }
    result = metadata_from_node(node, coarse_name="ship")
    assert tuple(result) == PAV_METADATA_COLUMNS
    assert all(math.isfinite(value) for value in result.values())
    assert result["meta_coarse_ship"] == 1
