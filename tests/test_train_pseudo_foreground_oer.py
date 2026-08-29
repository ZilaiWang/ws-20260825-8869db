import numpy as np
import pytest

from scripts.train_pseudo_foreground_oer import (
    FOREGROUND_COLUMNS,
    build_foreground_features,
    normalize_records,
)


def _row(**updates):
    row = {
        "image_id": 1,
        "category_id": 4,
        "bbox": [10.0, 20.0, 30.0, 15.0],
        "score": 0.25,
        "detector_score": 0.4,
        "foreground_probability": 0.8,
        "source_fold": 1,
        "source_model": "M3",
    }
    row.update(updates)
    return row


def test_normalize_and_feature_contract() -> None:
    records = normalize_records([_row()])
    assert records[0]["bbox_xyxy"] == [10.0, 20.0, 40.0, 35.0]
    features = build_foreground_features(
        records, category_mapping={4: "aircraft"}
    )
    assert features.shape == (1, len(FOREGROUND_COLUMNS))
    assert np.isfinite(features).all()
    assert features[0, FOREGROUND_COLUMNS.index("model_m3")] == 1.0
    assert features[0, FOREGROUND_COLUMNS.index("coarse_aircraft")] == 1.0
    assert features[0, FOREGROUND_COLUMNS.index("score_x_foreground")] == pytest.approx(0.32)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_invalid_foreground_probability_rejected(probability: float) -> None:
    with pytest.raises(ValueError, match="foreground_probability"):
        normalize_records([_row(foreground_probability=probability)])


def test_invalid_source_model_rejected() -> None:
    with pytest.raises(ValueError, match="source_model"):
        normalize_records([_row(source_model="unknown")])
