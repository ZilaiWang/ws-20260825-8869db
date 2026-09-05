#!/usr/bin/env python3
"""Normal-only fit followed by a frozen Hard → positive-only Sentinel replay."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rsdet.analysis.oof_detection import build_threshold_curve
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.hierarchical_thresholds import _subprotocol
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.experiments.fixed_proxy import review_quality_delta
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.postprocess.recall_calibration import (
    TARGETS,
    apply_recall_thresholds,
    select_recall_thresholds,
)
from rsdet.utils.config import load_config
from scripts.analyze_p40_vehicle_zoom import metrics, read, sha

CONFIG = ROOT / "configs/experiments/p40_recall_calibration_v1.json"


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def fit(normal, group_map, output):
    config = read(CONFIG)
    for name, key in (("ground_truth.json", "normal_gt_sha256"), ("predictions_low.json", "normal_pred_sha256"),
                      ("crossfit_frontier.json", "frontier_sha256")):
        if sha(normal / name) != config[key]:
            raise ValueError(f"wrong input: {name}")
    if sha(group_map) != config["group_map_sha256"]:
        raise ValueError("wrong group map")
    groups = {int(r["image_id"]): r["group_id"] for r in csv.DictReader(group_map.open())}
    gt_path, pred_path = normal / "ground_truth.json", normal / "predictions_low.json"
    gt, pred = load_coco_ground_truth(gt_path), load_coco_predictions(pred_path)
    folds = {r["id"]: r["fold"] for r in read(gt_path)["images"]}
    if len(folds) != 4481 or set(folds.values()) != {0, 1, 2} or set(folds)-set(groups) or set(pred)-set(folds):
        raise ValueError("invalid Normal coverage")
    gf = defaultdict(set)
    for i, f in folds.items():
        gf[groups[i]].add(f)
    if any(len(fs) != 1 for fs in gf.values()):
        raise ValueError("source groups cross folds")
    thresholds = {int(k): v for k, v in read(normal / "crossfit_frontier.json")["frontiers"]["0.150"]["crossfit_thresholds"].items()}
    protocol = parse_evaluation_protocol(load_config(ROOT / "configs/project.yaml"))
    fits = {}
    for heldout in range(3):
        ids = {i for i in folds if folds[i] != heldout}
        curves, support = {}, {}
        grid = sorted(set(build_threshold_grid(*config["grid"])) | {thresholds[heldout]})
        for label in TARGETS:
            fine_gt = {i: [r for r in gt[i] if r["category_id"] == label] for i in ids}
            fine_pred = {i: [r for r in pred.get(i, []) if r["category_id"] == label] for i in ids}
            support[label] = (sum(map(len, fine_gt.values())), len({groups[i] for i in ids if fine_gt[i]}))
            curves[label], _ = build_threshold_curve(fine_gt, fine_pred, thresholds=grid,
                protocol=_subprotocol(protocol, class_name=protocol.category_mapping[label], category_ids={label}))
        fits[heldout] = select_recall_thresholds(curves, incumbent=thresholds[heldout], support=support)
        fits[heldout]["fit_images"] = len(ids)
        fits[heldout]["heldout_images"] = len(folds)-len(ids)
        print(json.dumps({"fold": heldout, "weak_thresholds": {c: fits[heldout]["thresholds"][c] for c in TARGETS}}), flush=True)
    maps = {f: fits[f]["thresholds"] for f in fits}
    base = {i: [r for r in pred.get(i, []) if r["score"] >= thresholds[folds[i]]] for i in folds}
    candidate = apply_recall_thresholds(pred, folds, maps)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "fit.json", {"status": "fit_complete", "config": config, "config_sha256": sha(CONFIG),
        "code_sha256": {"script": sha(__file__), "core": sha(ROOT / "src/rsdet/postprocess/recall_calibration.py")},
        "fits": fits, "baseline_thresholds": thresholds, "normal_baseline": metrics(gt, base, protocol),
        "normal_candidate": metrics(gt, candidate, protocol), "historical_OOF_not_nested": True,
        "hard_sentinel_used_in_fit": False})


def evaluate(stage, gt_path, pred_path, output):
    fitted = read(output / "fit.json")
    if fitted["config_sha256"] != sha(CONFIG) or fitted["code_sha256"] != {
        "script": sha(__file__), "core": sha(ROOT / "src/rsdet/postprocess/recall_calibration.py")
    }:
        raise ValueError("fitted code/config changed")
    parent = read(ROOT / "configs/experiments/p40_vehicle_zoom_rescue_v1.json")
    if sha(gt_path) != parent["ground_truth_sha256"][stage]:
        raise ValueError("wrong fixed stress set")
    summary = read(pred_path.parent / "run_summary.json")
    if sha(pred_path) != summary["predictions_sha256"] or [r["weight_sha256"] for r in summary["folds"]] != parent["weights_sha256"]:
        raise ValueError("wrong stress prediction/weights")
    if stage == "sentinel" and not read(output / "hard.json")["review"]["direction_pass"]:
        raise ValueError("Hard failed; Sentinel is forbidden")
    raw = read(gt_path)
    folds = {r["id"]: r["fold"] for r in raw["images"]}
    if len(folds) != 6 or set(folds.values()) != {0, 1, 2}:
        raise ValueError("incomplete stress set")
    sources = {Path(s).name for r in raw["images"] for s in r["source_images"]}
    if len(sources) != 600:
        raise ValueError("invalid source inventory")
    if stage == "sentinel" and sources & set(read(output / "hard.json")["source_names"]):
        raise ValueError("overlapping Hard/Sentinel sources")
    if any(r["source_fold"] != folds[r["image_id"]] for r in read(pred_path)):
        raise ValueError("wrong heldout weights")
    gt, pred = load_coco_ground_truth(gt_path), load_coco_predictions(pred_path)
    thresholds = {int(k): v for k, v in fitted["baseline_thresholds"].items()}
    base = {i: [r for r in pred.get(i, []) if r["score"] >= thresholds[folds[i]]] for i in folds}
    candidate = apply_recall_thresholds(pred, folds, {int(f): r["thresholds"] for f, r in fitted["fits"].items()})
    for i in folds:
        if [r for r in base[i] if 4 <= r["category_id"] < 24] != [r for r in candidate[i] if 4 <= r["category_id"] < 24]:
            raise AssertionError("aircraft changed")
        if any(r not in candidate[i] for r in base[i]):
            raise AssertionError("incumbent removed")
    protocol = parse_evaluation_protocol(load_config(ROOT / "configs/project.yaml"))
    b, c = metrics(gt, base, protocol), metrics(gt, candidate, protocol)
    result = {"status": "complete", "stage": stage, "baseline": b, "candidate": c,
        "review": review_quality_delta(b["platform"], c["platform"], stage=stage,
            minimum=fitted["config"][f"{stage}_quality_delta_min_exclusive"]),
        "source_names": sorted(sources), "fit_sha256": sha(output / "fit.json"),
        "gt_sha256": sha(gt_path), "predictions_sha256": sha(pred_path),
        "incumbent_preserved": True, "aircraft_bypass": True, "historically_inspected_not_blind": True}
    write(output / f"{stage}.json", result)
    print(json.dumps(result["review"], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fit", "hard", "sentinel"))
    p.add_argument("--normal", type=Path)
    p.add_argument("--group-map", type=Path)
    p.add_argument("--gt", type=Path)
    p.add_argument("--pred", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.action == "fit":
        if a.output.exists():
            raise FileExistsError(a.output)
        fit(a.normal, a.group_map, a.output)
    else:
        if (a.output / f"{a.action}.json").exists():
            raise FileExistsError(a.output / f"{a.action}.json")
        evaluate(a.action, a.gt, a.pred, a.output)


if __name__ == "__main__":
    main()
