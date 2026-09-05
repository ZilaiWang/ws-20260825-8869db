#!/usr/bin/env python3
"""Diagnose missed objects in A development; never fit or read confirmation.

Two existing working points only: the baseline's selected threshold and its
already-decoded 0.001 floor. This is a candidate-coverage diagnostic, NOT an
achievable recall/score forecast or an oracle threshold recommendation.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rsdet.analysis.oof_detection import decompose_official_errors
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.experiments.paired_trend import read, safe_path, sha, validate_bundle, write
from rsdet.utils.config import load_config
from scripts import run_paired_trend as pipeline
from scripts.analyze_paired_fine_error_surface import _formal_ground_truth


def analyze(bundle: Path, evaluation: Path) -> dict:
    contract = validate_bundle(bundle, pipeline.PROJECT)
    review = read(evaluation / "review.json")
    if review["bundle_sha256"] != sha(bundle / "contract.json"):
        raise ValueError("wrong evaluation bundle")
    # Do not read confirmation predictions or metrics, including in cache checks.
    for name in ("development/predictions_low.json", "development/inference.json",
                 "development_metrics.json", "threshold.json"):
        if sha(safe_path(evaluation, name)) != review["artifacts"][name]:
            raise ValueError("development artifact changed")
    gt_path = bundle / "development_gt.json"
    pred_path = evaluation / "development/predictions_low.json"
    receipt = read(evaluation / "development/inference.json")
    selected = read(evaluation / "threshold.json")
    threshold = selected["threshold"]
    raw = read(gt_path)
    if (selected["development_gt_sha256"] != sha(gt_path)
            or selected["development_predictions_sha256"] != sha(pred_path)
            or selected["checkpoint_sha256"] != receipt["request"]["checkpoint_sha256"]
            or receipt["image_ids"] != sorted(i["id"] for i in raw["images"])
            or receipt["request"]["config"] != contract["inference"]):
        raise ValueError("development coverage or threshold provenance mismatch")
    gt, predictions = pipeline.load_predictions(gt_path, pred_path)
    groups = {s["image_id"]: s["group_id"] for s in read(bundle / "manifest.json")["samples"]
              if s["split"] == "development"}
    # The shared object-conservation helper expects fold metadata; 0 is merely
    # its unused bookkeeping field here, not a claim this test is CV3.
    raw["images"] = [{**i, "fold": 0, "group_id": groups[i["id"]]} for i in raw["images"]]
    protocol = parse_evaluation_protocol(load_config(pipeline.PROJECT))
    formal = _formal_ground_truth(raw, gt, protocol)
    high, _, high_fn = decompose_official_errors(formal, predictions, threshold=threshold,
                            protocol=protocol, model_key="selected_development", include_cases=False)
    low, _, low_fn = decompose_official_errors(formal, predictions, threshold=contract["inference"]["score_floor"],
                            protocol=protocol, model_key="decoded_floor_diagnostic", include_cases=False)
    recovered = set(high_fn) - set(low_fn)
    counts = Counter(formal.objects[k].category_id for k in recovered)
    fine = {}
    for label, name in enumerate(pipeline.FINE_NAMES):
        keys = {k for k, obj in formal.objects.items() if obj.category_id == label}
        fine[str(label)] = {
            "fine_name": name, "gt": len(keys),
            "selected_fn": len(keys & set(high_fn)),
            "fn_recovered_at_decoded_floor": counts[label],
            "remaining_fn_at_decoded_floor": len(keys & set(low_fn)),
            "selected_error_roles": high["per_fine_category"][str(label)],
            "floor_error_roles": low["per_fine_category"][str(label)],
        }
    return {
        "status": "development_diagnostic_only", "bundle_sha256": sha(bundle / "contract.json"),
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "development_gt_sha256": sha(gt_path), "prediction_sha256": sha(pred_path),
        "selected_threshold": threshold, "decoded_floor": contract["inference"]["score_floor"],
        "selected": high, "floor": low, "per_fine": fine,
        "selected_fn_recovered_at_floor": len(recovered),
        "selected_matches_lost_at_floor": len(set(low_fn) - set(high_fn)),
        "confirmation_used": False, "thresholds_fitted": False,
        "limitations": [
            "Floor includes post-NMS/max_det truncation; it is not the detector's theoretical ceiling.",
            "Recoverable FN comes with actual extra FP; not achievable free recall.",
            "FP_BG is an unmatched-label role, not proof of visual background or a training-negative label.",
            "Do not train on development cases. Use analogous training-only cases for any new loss/data recipe.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=ROOT / "data/splits/paired_trend_v1")
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.bundle, args.evaluation)
    write(args.output, result)
    print(json.dumps({k: result[k] for k in ("status", "selected_fn_recovered_at_floor",
                                           "confirmation_used", "thresholds_fitted")}))


if __name__ == "__main__":
    main()
