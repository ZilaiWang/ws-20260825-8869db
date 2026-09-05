#!/usr/bin/env python3
"""Current P40 OOF miss audit at its frozen workpoint and decoded floor.

This diagnostic uses Normal only. Floor matches are not free achievable recall:
their accompanying false positives and post-decoding limits are reported.
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
from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config
from scripts.analyze_p40_vehicle_zoom import metrics, read, sha
from scripts.analyze_paired_fine_error_surface import _formal_ground_truth


def analyze(normal: Path) -> dict:
    gt_path = normal / "ground_truth.json"
    pred_path = normal / "predictions_low.json"
    frontier_path = normal / "crossfit_frontier.json"
    expected = {
        "ground_truth.json": "c4290b542ffdafe62d5dbcb575f0b3431d46721bbcb366f8ef05291653fcb975",
        "predictions_low.json": "e96870c9e10bdd8022846b03ed40ec7700c822be81433d74c4245cad7cedfdbc",
        "crossfit_frontier.json": "545e02b2d252909400ff5cf8f9ea7768bb8438dd99c6e26269fc0807132c81be",
    }
    for name, digest in expected.items():
        if sha(normal / name) != digest:
            raise ValueError(f"wrong frozen P40 input: {name}")
    raw = read(gt_path)
    folds = {int(r["id"]): int(r["fold"]) for r in raw["images"]}
    if len(folds) != 4481 or set(folds.values()) != {0, 1, 2}:
        raise ValueError("incomplete CV3 image universe")
    gt, pred = load_coco_ground_truth(gt_path), load_coco_predictions(pred_path)
    if set(pred) - set(folds):
        raise ValueError("predictions outside Normal")
    thresholds = {int(k): v for k, v in read(frontier_path)["frontiers"]["0.150"]["crossfit_thresholds"].items()}
    selected = {i: [r for r in pred.get(i, []) if r["score"] >= thresholds[folds[i]]] for i in folds}
    protocol = parse_evaluation_protocol(load_config(ROOT / "configs/project.yaml"))
    formal = _formal_ground_truth(raw, gt, protocol)
    high, _, high_fn = decompose_official_errors(formal, selected, threshold=0.001,
        protocol=protocol, model_key="p40_crossfit_selected", include_cases=False)
    low, _, low_fn = decompose_official_errors(formal, pred, threshold=0.001,
        protocol=protocol, model_key="p40_decoded_floor", include_cases=False)
    recovered = set(high_fn) - set(low_fn)
    counts = Counter(formal.objects[k].category_id for k in recovered)
    per_fine = {}
    for label in range(25):
        keys = {k for k, obj in formal.objects.items() if obj.category_id == label}
        per_fine[str(label)] = {
            "name": FINE_NAMES[label], "gt": len(keys), "selected_fn": len(keys & set(high_fn)),
            "recovered_at_floor": counts[label], "remaining_floor_fn": len(keys & set(low_fn)),
            "selected_roles": high["per_fine_category"][str(label)],
            "floor_roles": low["per_fine_category"][str(label)],
        }
    return {
        "status": "normal_diagnostic_complete", "thresholds": thresholds,
        "selected": metrics(gt, selected, protocol), "decoded_floor": metrics(gt, pred, protocol),
        "per_fine": per_fine, "selected_matches_lost_at_floor": len(set(low_fn)-set(high_fn)),
        "inputs_sha256": {p.name: sha(p) for p in (gt_path, pred_path, frontier_path)},
        "hard_sentinel_used": False, "thresholds_fitted": False,
        "limitations": ["Decoded floor is post-NMS and max_det, not a theoretical upper bound.",
            "Extra floor TP come with FP; no claim all are recoverable by a valid model.",
            "Research CV3 P40 is 40+40 epochs, not deployed full 160+40.",
            "FP_BG denotes unmatched annotation, not a verified training negative."],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--normal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.normal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result["per_fine"].items() if int(k) in (0, 1, 2, 3, 24)}, indent=2))


if __name__ == "__main__":
    main()
