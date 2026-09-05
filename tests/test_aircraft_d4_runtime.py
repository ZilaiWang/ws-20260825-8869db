from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from rsdet.contracts import Prediction
from rsdet.features.p04_inputs import D4_VIEW_IDS, apply_d4_view
from rsdet.submission.aircraft_d4 import (
    _normalize,
    _tensorized_d4_views,
    apply_aircraft_probabilities,
    filter_prediction_by_score,
)


def _probability(label20: int, confidence: float) -> list[float]:
    values = [(1.0 - confidence) / 19.0] * 20
    values[label20] = confidence
    return values


def test_aircraft_d4_relabels_only_aircraft_and_repeats_same_class_nms() -> None:
    prediction = Prediction(
        image_id=7,
        boxes_xyxy=[
            [0, 0, 10, 10],
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
        ],
        scores=[0.8, 0.7, 0.6, 0.5],
        labels=[4, 5, 24, 0],
    )
    result = apply_aircraft_probabilities(
        prediction,
        [_probability(2, 0.95), _probability(2, 0.91)],
        min_probability=0.9,
        nms_iou=0.5,
    )
    assert result.labels == [6, 24, 0]
    assert result.scores == [0.8, 0.6, 0.5]
    assert prediction.labels == [4, 5, 24, 0]


def test_aircraft_d4_rejects_invalid_probability_rows() -> None:
    prediction = Prediction(1, [[0, 0, 10, 10]], [0.5], [4])
    with pytest.raises(ValueError, match="normalized"):
        apply_aircraft_probabilities(
            prediction,
            [[0.0] * 20],
            min_probability=0.9,
            nms_iou=0.5,
        )


def test_score_prefilter_is_stable_and_does_not_mutate_input() -> None:
    prediction = Prediction(
        9,
        [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]],
        [0.7, 0.5359, 0.536],
        [4, 24, 5],
    )
    filtered = filter_prediction_by_score(prediction, 0.536)
    assert filtered.boxes_xyxy == [[0, 0, 1, 1], [2, 2, 3, 3]]
    assert filtered.scores == [0.7, 0.536]
    assert filtered.labels == [4, 5]
    assert prediction.labels == [4, 24, 5]


def test_tensorized_d4_views_match_object_major_contract() -> None:
    torch = pytest.importorskip("torch")
    images = torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    actual = _tensorized_d4_views(images)
    expected = []
    for image in images:
        flipped = torch.flip(image, dims=(-1,))
        expected.extend(
            [
                image,
                torch.rot90(image, 1, dims=(-2, -1)),
                torch.rot90(image, 2, dims=(-2, -1)),
                torch.rot90(image, 3, dims=(-2, -1)),
                flipped,
                torch.rot90(flipped, 1, dims=(-2, -1)),
                torch.rot90(flipped, 2, dims=(-2, -1)),
                torch.rot90(flipped, 3, dims=(-2, -1)),
            ]
        )
    assert torch.equal(actual, torch.stack(expected))


def test_tensorized_views_are_bitwise_equal_to_legacy_pil_views() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pixels = np.arange(3 * 7 * 7, dtype=np.uint8).reshape(7, 7, 3)
    image = Image.fromarray(pixels, mode="RGB")
    legacy = torch.stack([_normalize(apply_d4_view(image, view)) for view in D4_VIEW_IDS])
    tensorized = _tensorized_d4_views(_normalize(image).unsqueeze(0))
    assert torch.equal(tensorized, legacy)
