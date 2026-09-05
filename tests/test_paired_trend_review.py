import copy
import json
import subprocess
import sys

import pytest

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions
from scripts.compare_candidate_trend import compare


def _payload(recall=0.9, fdr=0.1, latency=3.0):
    return {
        "input_sha256": {"gt": "same-gt"},
        "latency_seconds": latency,
        "image_coverage": {"negative_coverage_known": True},
        "platform": {
            "metric_protocol": "platform_observed_20260831",
            "per_coarse": {
                name: {"macro_recall": recall, "macro_fdr": fdr}
                for name in ("ship", "aircraft", "vehicle")
            },
        },
    }


def test_empty_images_are_retained_and_false_positives_counted(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"images": [{"id": 1}, {"id": 2}], "annotations": [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]}
    ]}))
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps([
        {"image_id": i, "category_id": 0, "bbox": [0, 0, 10, 10], "score": .9}
        for i in (1, 2)
    ]))
    gt = load_coco_ground_truth(path)
    assert gt[2] == []
    raw = load_coco_predictions(pred_path)
    filtered = {i: raw.get(i, []) for i in gt}  # Existing downstream pattern.
    result = evaluate_predictions(gt, filtered, class_names=["ship"])
    assert result.details["tp"] == 1
    assert result.details["fp"] == 1
    assert result.fdr == .5


@pytest.mark.parametrize("images,annotations", [
    ([{"id": 1}, {"id": 1}], []),
    ([{"id": 1}], [{"image_id": 2, "category_id": 0, "bbox": [0, 0, 1, 1]}]),
])
def test_invalid_gt_universe_fails_closed(tmp_path, images, annotations):
    p = tmp_path / "gt.json"
    p.write_text(json.dumps({"images": images, "annotations": annotations}))
    with pytest.raises(ValueError):
        load_coco_ground_truth(p)


def test_legacy_annotation_only_input_is_still_readable(tmp_path):
    p = tmp_path / "gt.json"
    p.write_text('{"annotations": []}')
    assert load_coco_ground_truth(p) == {}


def test_score_tradeoff_not_rejected_because_recall_dropped():
    result = compare(_payload(.90, .32), _payload(.89, .10))
    assert result["direction"] == "positive"
    assert result["per_coarse_delta"]["ship"]["recall_pp"] < 0
    assert result["automatic_full_training_or_submission_admission"] is False


def test_missing_timing_does_not_invent_total():
    result = compare(_payload(), _payload(.92, latency=None))
    assert result["delta_total_score"] is None
    assert result["candidate"]["total_score"] is None
    assert result["warnings"]


def test_same_inputs_small_changes_and_seen_diagnostics():
    x = _payload()
    x["diagnostic_only"] = True
    result = compare(x, copy.deepcopy(x))
    assert result["direction"] == "small_or_uncertain"
    assert result["delta_total_score"] == 0
    assert any("Seen-source" in warning for warning in result["warnings"])


def test_mismatched_gt_or_metric_protocol_rejected():
    x = _payload()
    x["input_sha256"]["gt"] = "other"
    with pytest.raises(ValueError, match="same GT"):
        compare(_payload(), x)
    x = _payload()
    x["platform"]["metric_protocol"] = "legacy_pooled"
    with pytest.raises(ValueError, match="platform_observed"):
        compare(_payload(), x)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_latency_rejected(value):
    with pytest.raises(ValueError):
        compare(_payload(), _payload(latency=value))


def test_fixed_evaluation_cli_counts_negative_and_emits_trend_input(tmp_path):
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    output = tmp_path / "metrics.json"
    annotations = [
        {"image_id": i, "category_id": i, "bbox": [0, 0, 10, 10]}
        for i in range(25)
    ]
    gt_path.write_text(json.dumps({
        "images": [{"id": i} for i in range(26)], "annotations": annotations,
    }))
    predictions = [{**a, "score": .9} for a in annotations]
    predictions.append({"image_id": 25, "category_id": 24,
                        "bbox": [0, 0, 10, 10], "score": .9})
    pred_path.write_text(json.dumps(predictions))
    command = [sys.executable, "scripts/evaluate_fixed_score_threshold.py",
               "--gt", str(gt_path), "--pred", str(pred_path),
               "--threshold", ".5", "--latency-seconds", "3",
               "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["image_coverage"]["empty_gt_images"] == 1
    assert result["per_coarse"]["vehicle"]["fp"] == 1
    assert result["platform"]["per_coarse"]["vehicle"]["macro_fdr"] == .5
    assert compare(result, result)["delta_total_score"] == 0
    predictions[-1]["image_id"] = 999
    pred_path.write_text(json.dumps(predictions))
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "outside GT" in failed.stderr
