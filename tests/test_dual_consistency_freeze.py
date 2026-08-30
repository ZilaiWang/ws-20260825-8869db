# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rsdet.innovation.dual_consistency import (
    blend_probability,
    deserialize_model,
    quality_features,
    serialize_model,
)


def test_frozen_logistic_roundtrip() -> None:
    model = (
        np.arange(12, dtype=np.float64),
        np.arange(1, 13, dtype=np.float64),
        np.linspace(-1.0, 1.0, 12),
        0.25,
    )
    restored = deserialize_model(
        serialize_model(model), "quality_features_v1_12d"
    )
    for actual, expected in zip(restored[:3], model[:3], strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert restored[3] == model[3]


def test_frozen_logistic_rejects_zero_scale() -> None:
    payload = serialize_model(
        (np.zeros(12), np.ones(12), np.zeros(12), 0.0)
    )
    payload["scale"][3] = 0.0
    with pytest.raises(ValueError, match="invalid values"):
        deserialize_model(payload, "quality_features_v1_12d")


def test_v2_features_have_expected_finite_shape() -> None:
    rows = [
        {
            "score": 0.4,
            "nearby_identity_score": 0.6,
            "novel_same_fine_iou": 0.5,
            "bbox_xyxy": [0.0, 0.0, 20.0, 10.0],
            "category_id": 4,
        }
    ]
    features = quality_features(rows, "quality_features_v2_22d")
    assert features.shape == (1, 22)
    assert np.isfinite(features).all()


def test_logit_blend_endpoints() -> None:
    raw = np.asarray([0.2, 0.8])
    quality = np.asarray([0.7, 0.3])
    np.testing.assert_allclose(blend_probability(raw, quality, 0.0), raw)
    np.testing.assert_allclose(blend_probability(raw, quality, 1.0), quality)
