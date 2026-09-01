from __future__ import annotations

import math

import numpy as np

from scripts.evaluate_quality_route_frozen_benchmark import build_base_crop_features


def test_base_crop_feature_contract_is_63d_and_score_aligned() -> None:
    predictions = [
        {
            "image_id": 1,
            "category_id": 4,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "score": 0.4,
            "crop_top1": 0.8,
            "crop_margin": 0.6,
            "crop_entropy": 0.5,
            "crop_top1_class": 4,
            "detector_crop_agree": 1,
        },
        {
            "image_id": 1,
            "category_id": 24,
            "bbox": [100.0, 100.0, 20.0, 10.0],
            "score": 0.2,
            "crop_top1": 0.7,
            "crop_margin": 0.4,
            "crop_entropy": 0.9,
            "crop_top1_class": 23,
            "detector_crop_agree": 0,
        },
    ]
    coarse = {**{index: "ship" for index in range(4)}, **{index: "aircraft" for index in range(4, 24)}, 24: "vehicle"}
    features, scores = build_base_crop_features(
        predictions,
        score_field="score",
        coarse_mapping=coarse,
        density_radius=1024.0,
    )
    assert features.shape == (2, 63)
    assert np.allclose(scores, [0.4, 0.2])
    assert math.isclose(float(features[0, 0]), 0.4, rel_tol=1e-6)
    assert math.isclose(float(features[0, 9]), math.log1p(1.0), rel_tol=1e-6)
    assert features[0, 10:13].tolist() == [0.0, 1.0, 0.0]
    assert features[1, 10:13].tolist() == [0.0, 0.0, 1.0]
