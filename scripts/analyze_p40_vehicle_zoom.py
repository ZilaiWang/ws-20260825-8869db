#!/usr/bin/env python3
"""Fixed Hard/Sentinel P40 Vehicle rescue replay, without threshold fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.experiments.fixed_proxy import review_quality_delta
from rsdet.postprocess.complementary_rescue import append_class_rescue
from rsdet.postprocess.vehicle_rescue import append_vehicle_rescue
from rsdet.utils.config import load_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def metrics(gt, pred, protocol, latency=None):
    kwargs = dict(class_names=protocol.class_names, category_mapping=protocol.category_mapping,
                  iou_thresholds=protocol.iou_thresholds)
    pooled = evaluate_predictions(gt, pred, **kwargs)
    ranking = evaluate_ranking_metrics(gt, pred, require_complete_taxonomy=True, **kwargs)
    return {
        "platform": platform_metrics_payload(build_platform_observed_metrics(
            ranking, latency_seconds=latency)),
        "pooled_diagnostic": asdict(pooled),
        "per_fine": {str(k): asdict(v) for k, v in ranking.per_fine.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=("hard", "sentinel"), required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--frontier", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--historical-baseline", type=Path)
    p.add_argument("--other-gt", type=Path, required=True)
    p.add_argument("--config", type=Path,
                   default=Path("configs/experiments/p40_vehicle_zoom_rescue_v1.json"))
    p.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = p.parse_args()
    config = read(args.config)
    frozen = read(args.run_dir / "preflight.json")
    if sha(args.config) != frozen["contract_sha256"] or config != frozen["contract"]:
        raise ValueError("contract changed since inference preflight")
    if sha(args.gt) != config["ground_truth_sha256"][args.condition]:
        raise ValueError("GT differs from frozen experiment")
    other_condition = "sentinel" if args.condition == "hard" else "hard"
    if sha(args.other_gt) != config["ground_truth_sha256"][other_condition]:
        raise ValueError("other stress set differs from frozen experiment")
    if sha(args.frontier) != config["frontier_sha256"]:
        raise ValueError("threshold frontier differs from frozen experiment")
    for relative, expected in frozen["code_sha256"].items():
        if sha(relative) != expected:
            raise ValueError(f"inference code changed: {relative}")
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    thresholds = {int(k): float(v) for k, v in frozen["thresholds"].items()}
    if set(thresholds) != {0, 1, 2} or any(
        not math.isfinite(v) or not 0 <= v <= 1 for v in thresholds.values()
    ):
        raise ValueError("invalid thresholds")
    gt_raw = read(args.gt)
    image_folds = {int(r["id"]): int(r["fold"]) for r in gt_raw["images"]}
    gt = load_coco_ground_truth(args.gt)
    if set(image_folds.values()) != {0, 1, 2} or len(image_folds) != 6:
        raise ValueError("incomplete fixed 6-image CV3 proxy")
    def sources(data):
        return {Path(s).name for r in data["images"] for s in r["source_images"]}
    source_set, other_sources = sources(gt_raw), sources(read(args.other_gt))
    if len(source_set) != 600 or len(other_sources) != 600 or source_set & other_sources:
        raise ValueError("Hard/Sentinel source inventory/disjointness mismatch")
    predictions, timings, audit = {}, {}, {}
    auxiliary_view = config.get("auxiliary_view", "zoom")
    for view in ("base", auxiliary_view):
        directory = args.run_dir / view
        pred_path = directory / "predictions.json"
        summary = read(directory / "run_summary.json")
        if (sha(pred_path) != summary["predictions_sha256"]
                or summary["selected_folds"] != [0, 1, 2]
                or summary["imgsz"] != config[view]["imgsz"]):
            raise ValueError("inference summary integrity/coverage mismatch")
        for key in ("tile_size", "overlap"):
            if summary["pipeline"][key] != config[view][key]:
                raise ValueError("inference geometry mismatch")
        if summary.get("tile_rotation", 0) != config[view].get("tile_rotation", 0):
            raise ValueError("inference rotation mismatch")
        for row in read(pred_path):
            if int(row["image_id"]) not in image_folds:
                raise ValueError("prediction outside image universe")
            if int(row["source_fold"]) != image_folds[int(row["image_id"])]:
                raise ValueError("wrong fold weight on held-out image")
        for fold, row in enumerate(summary["folds"]):
            if row["fold"] != fold or row["weight_sha256"] != config["weights_sha256"][fold]:
                raise ValueError("wrong P40 weight")
        raw = load_coco_predictions(pred_path)
        predictions[view] = {
            i: [r for r in raw.get(i, []) if r["score"] >= thresholds[image_folds[i]]]
            for i in image_folds
        }
        rows = [t for fold in summary["folds"] for t in fold["image_timings"]]
        timings[view] = {r["file_name"]: r["wall_seconds"] for r in rows}
        if set(timings[view]) != {r["file_name"] for r in gt_raw["images"]}:
            raise ValueError("missing/duplicate timing image")
        audit[view] = {"predictions_sha256": sha(pred_path), "summary_sha256": sha(directory / "run_summary.json")}
    started = time.perf_counter()
    if config.get("rescue_categories") == "all_fine":
        merged, counts = append_class_rescue(predictions["base"], predictions[auxiliary_view],
            category_iou={i: (0.35 if i == 24 else 0.5) for i in range(25)})
    else:
        merged, counts = append_vehicle_rescue(predictions["base"], predictions[auxiliary_view],
                                              dedup_iou=config["dedup_iou"])
    merge_seconds = time.perf_counter() - started
    base_time = statistics.mean(timings["base"].values())
    candidate_time = base_time + statistics.mean(timings[auxiliary_view].values()) + merge_seconds / len(gt)
    base = metrics(gt, predictions["base"], protocol, base_time)
    candidate = metrics(gt, merged, protocol, candidate_time)
    # Semantic invariant: both non-Vehicle records and all incumbent records survive.
    for i in image_folds:
        if merged[i][:len(predictions["base"][i])] != predictions["base"][i]:
            raise AssertionError("incumbent predictions changed")
        if config.get("rescue_categories") != "all_fine" and [r for r in merged[i] if r["category_id"] != 24] != [
            r for r in predictions["base"][i] if r["category_id"] != 24
        ]:
            raise AssertionError("non-Vehicle predictions changed")
    historical = None
    if args.historical_baseline:
        old_raw = load_coco_predictions(args.historical_baseline)
        if set(old_raw) - set(gt):
            raise ValueError("historical predictions outside proxy")
        old = {i: [r for r in old_raw.get(i, []) if r["score"] >= thresholds[image_folds[i]]]
               for i in image_folds}
        historical = {"sha256": sha(args.historical_baseline),
                      "thresholded_predictions_equal": old == predictions["base"],
                      "metrics": metrics(gt, old, protocol)}
    review = review_quality_delta(base["platform"], candidate["platform"],
        stage=args.condition, minimum=config[f"{args.condition}_quality_delta_min_exclusive"])
    result = {
        "status": "complete", "experiment_id": config["experiment_id"],
        "condition": args.condition, "thresholds": thresholds,
        "source_audit": {"images": len(gt), "source_images": len(source_set),
                         "other_source_images": len(other_sources), "shared_sources": 0,
                         "historically_inspected_not_blind": True},
        "baseline": base, "candidate": candidate, "historical_baseline": historical,
        "merge_counts": counts, "incumbents_unchanged": True,
        "ship_aircraft_unchanged": config.get("rescue_categories") != "all_fine",
        "review": review, "artifact_sha256": audit,
        "evaluation_code_sha256": {str(Path(__file__).relative_to(Path.cwd())): sha(__file__),
            "src/rsdet/postprocess/vehicle_rescue.py": sha("src/rsdet/postprocess/vehicle_rescue.py"),
            "src/rsdet/experiments/fixed_proxy.py": sha("src/rsdet/experiments/fixed_proxy.py")},
        "timing": {"base_mean_seconds": base_time, "combined_estimate_seconds": candidate_time,
                   "same_gpu_sequential_component_sum_not_docker": True,
                   "delta_proxy_score_with_component_timing": candidate["platform"]["absolute_score"] - base["platform"]["absolute_score"],
                   "candidate_exceeds_20_seconds": candidate_time > 20},
        "full_training_docker_submission_authorized": False,
    }
    (args.run_dir / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    rows = [{"image_id": i, "category_id": r["category_id"], "score": r["score"],
             "source_fold": image_folds[i],
             "bbox": [r["bbox_xyxy"][0], r["bbox_xyxy"][1],
                      r["bbox_xyxy"][2]-r["bbox_xyxy"][0], r["bbox_xyxy"][3]-r["bbox_xyxy"][1]]}
            for i, items in merged.items() for r in items]
    (args.run_dir / "candidate_predictions.json").write_text(json.dumps(rows, indent=2)+"\n")
    print(json.dumps({"review": review, "counts": counts, "timing": result["timing"]}, indent=2))


if __name__ == "__main__":
    main()
