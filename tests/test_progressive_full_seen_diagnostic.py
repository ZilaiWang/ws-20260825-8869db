from __future__ import annotations

import json

import pytest

from scripts.run_progressive_full_seen_diagnostic import SAFETY, THRESHOLD, metrics


def fixture_paths(tmp_path):
    gt = {"images": [{"id": 1, "width": 1000, "height": 1000}],
          "categories": [{"id": i, "name": str(i)} for i in range(25)],
          "annotations": [{"id": i + 1, "image_id": 1, "category_id": i,
                           "bbox": [i * 20, 0, 10, 10]} for i in range(25)]}
    predictions = [{"image_id": 1, "category_id": i, "bbox": [i * 20, 0, 10, 10],
                    "score": THRESHOLD} for i in range(25)]
    # A below-threshold false alarm must not affect the frozen working point.
    predictions.append({"image_id": 1, "category_id": 24,
                        "bbox": [800, 800, 10, 10], "score": THRESHOLD - 0.001})
    g, p = tmp_path / "gt.json", tmp_path / "pred.json"
    g.write_text(json.dumps(gt))
    p.write_text(json.dumps(predictions))
    return g, p


def test_full_seen_has_no_generalization_or_admission_claim():
    assert SAFETY["data_leakage"] is True
    for field in ["is_oof", "eligible_for_admission", "official_score_prediction"]:
        assert SAFETY[field] is False


def test_frozen_threshold_inclusive_platform_score(tmp_path):
    g, p = fixture_paths(tmp_path)
    result = metrics(g, p, 4.0)
    assert result["predictions_after_threshold"] == 25
    assert result["platform"]["gate_recall"] == 1.0
    assert result["platform"]["gate_fdr"] == 0.0
    assert result["platform"]["absolute_score"] == pytest.approx(100 - 4 / 7)
    assert result["threshold_retuned"] is False


def test_normal_without_latency_does_not_invent_score(tmp_path):
    g, p = fixture_paths(tmp_path)
    assert metrics(g, p, None)["platform"]["absolute_score"] is None


@pytest.mark.parametrize("key,value", [("image_id", 999), ("category_id", 25),
                                       ("score", float("nan")), ("score", 1.1),
                                       ("bbox", [0, 0, -1, 10])])
def test_bad_predictions_fail_closed(tmp_path, key, value):
    g, p = fixture_paths(tmp_path)
    rows = json.loads(p.read_text())
    rows[0][key] = value
    p.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        metrics(g, p, None)
