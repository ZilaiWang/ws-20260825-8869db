#!/usr/bin/env python3
"""Apply a Normal-only frozen rescue model on one fixed stress proxy."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analyze_p40_vehicle_zoom import metrics, read, sha
from calibrate_p40_supported_rescue import apply_models

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.experiments.fixed_proxy import review_quality_delta
from rsdet.utils.config import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--condition", choices=("hard", "sentinel"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    fit = read(args.models)
    config_path = Path("configs/experiments/p40_rot90_supported_rescue_v1.json")
    if fit["config_sha256"] != sha(config_path) or fit["fit_hard_sentinel"]:
        raise ValueError("fit contract differs or used stress labels")
    parent = read(fit["config"]["parent_contract"])
    if sha(args.gt) != parent["ground_truth_sha256"][args.condition]:
        raise ValueError("stress GT differs")
    folds = {r["id"]: r["fold"] for r in read(args.gt)["images"]}
    base, auxiliary = {}, {}
    timing = {}
    for view in ("base", "rot90"):
        summary = read(args.views / view / "run_summary.json")
        path = args.views / view / "predictions.json"
        if sha(path) != summary["predictions_sha256"]:
            raise ValueError("view predictions changed")
        if [r["weight_sha256"] for r in summary["folds"]] != parent["weights_sha256"]:
            raise ValueError("view used wrong fold weights")
        for row in read(path):
            if row["source_fold"] != folds[row["image_id"]]:
                raise ValueError("wrong heldout fold")
        if view == "base":
            base = load_coco_predictions(path)
        else:
            auxiliary = load_coco_predictions(path)
        timing[view] = statistics.mean(t["wall_seconds"] for f in summary["folds"] for t in f["image_timings"])
    thresholds = {int(k): v for k, v in fit["thresholds"].items()}
    baseline = {i: [r for r in base.get(i, []) if r["score"] >= thresholds[folds[i]]]
                for i in folds}
    candidate, counts = apply_models(base, auxiliary, folds, thresholds, fit["models"])
    for i in folds:
        if candidate[i][:len(baseline[i])] != baseline[i]:
            raise AssertionError("incumbent changed")
    protocol = parse_evaluation_protocol(load_config("configs/project.yaml"))
    gt = load_coco_ground_truth(args.gt)
    b = metrics(gt, baseline, protocol, timing["base"])
    c = metrics(gt, candidate, protocol, sum(timing.values()))
    review = review_quality_delta(b["platform"], c["platform"], stage=args.condition,
                                  minimum=fit["config"][f"{args.condition}_quality_delta_min_exclusive"])
    result = {"status": "complete", "condition": args.condition,
              "experiment_id": fit["config"]["experiment_id"],
              "baseline": b, "candidate": c, "counts": counts, "review": review,
              "timing": {"base": timing["base"], "two_view_component_sum": sum(timing.values()),
                         "not_docker_not_total_calibrator_latency": True},
              "models_sha256": sha(args.models), "gt_sha256": sha(args.gt),
              "no_hard_sentinel_parameter_selection": True,
              "calibration_code_sha256": sha("scripts/calibrate_p40_supported_rescue.py"),
              "evaluation_code_sha256": sha(__file__)}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"review": review, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
