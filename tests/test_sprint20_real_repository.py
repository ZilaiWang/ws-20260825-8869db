"""Server-side integration checks. Explicitly skipped without the real repository."""

import json
from pathlib import Path

import numpy as np
import pytest

from rsdet.contracts import Prediction
from rsdet.evaluation.absolute_score import platform_confirmed_score
from rsdet.postprocess.nms import nms
from rsdet.submission.aircraft_d4 import apply_aircraft_probabilities
from sprint20.evaluation import evaluate
from sprint20.policy import _apply_aircraft_labels, prediction_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def test_original_nms_helper_parity():
    p = Prediction(
        1, [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], [0.9, 0.8, 0.7], [4, 5, 24]
    )
    probabilities = np.eye(20)[[1, 1]]
    original = apply_aircraft_probabilities(p, probabilities, min_probability=0.9, nms_iou=0.5)
    new = _apply_aircraft_labels(p, [1, 1], nms, nms_iou=0.5)
    assert prediction_fingerprint(original) == prediction_fingerprint(new)


def test_actual_platform_scorer_v2_regression():
    rows = {
        "ship": {"recall": 0.809894, "fdr": 0.067489},
        "aircraft": {"recall": 0.945967, "fdr": 0.037265},
        "vehicle": {"recall": 79 / 95, "fdr": 0.21},
    }
    assert platform_confirmed_score(rows, 3.551833)["total_score"] == pytest.approx(
        76.6010, abs=0.0001
    )


def test_actual_matcher_complete_taxonomy():
    gt = {1: [{"bbox_xyxy": [c * 20, 0, c * 20 + 10, 10], "category_id": c} for c in range(25)]}
    pred = {1: [{**x, "score": 0.9} for x in gt[1]]}
    result = evaluate(gt, pred, 4.0)
    assert result["score"]["total_score"] == pytest.approx(100 - 4 / 7)
    pred[1].append(dict(pred[1][-1]))
    worse = evaluate(gt, pred, 4.0)
    assert worse["per_fine"]["24"]["fp"] == 1
    assert worse["score"]["total_score"] < result["score"]["total_score"]


def test_ship23_candidate_changes_only_explicit_sprint20_ownership():
    base = json.loads(
        (ROOT / "configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json").read_text()
    )
    candidate = json.loads(
        (ROOT / "configs/experiments/hera_sprint20_p40_d4_otm_ship23_candidate_v1.json").read_text()
    )
    assert candidate["sprint20"] == {
        "mode": "shared",
        "bounded_d4": False,
        "allow_experimental": True,
        "otm_labels": [2, 3],
        "otm_threshold": 0.536,
    }
    for key in (
        "contract_version",
        "metric_protocol",
        "device",
        "model",
        "pipeline",
        "post_fusion_score_threshold",
        "aircraft_classifier_model",
    ):
        assert candidate[key] == base[key]
