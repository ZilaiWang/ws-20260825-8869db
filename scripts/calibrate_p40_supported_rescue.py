#!/usr/bin/env python3
"""Fit only on the other two Normal folds; audit rotation-rescue precision.

No Hard/Sentinel predictions or labels are read by this calibration program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import compute_iou, evaluate_predictions_with_trace
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.complementary_rescue import append_class_rescue
from rsdet.utils.config import load_config

COARSE = ("ship", "aircraft", "vehicle")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coarse(label):
    return "ship" if label < 4 else "aircraft" if label < 24 else "vehicle"


def features(row, raw):
    compatible = [r for r in raw if r["category_id"] == row["category_id"]]
    best = max(compatible, key=lambda r: (compute_iou(row["bbox_xyxy"], r["bbox_xyxy"]),
                                         r["score"]), default=None)
    overlap = compute_iou(row["bbox_xyxy"], best["bbox_xyxy"]) if best else 0.0
    x1, y1, x2, y2 = row["bbox_xyxy"]
    return [row["score"], overlap, best["score"] if best else 0.0,
            math.log1p(x2-x1), math.log1p(y2-y1)]


def candidates(base, auxiliary, ids, threshold):
    incumbent = {i: [r for r in base.get(i, []) if r["score"] >= threshold] for i in ids}
    selected = {i: [r for r in auxiliary.get(i, []) if r["score"] >= threshold] for i in ids}
    merged, _ = append_class_rescue(incumbent, selected,
                                    category_iou={i: .35 if i == 24 else .5 for i in range(25)})
    return incumbent, merged, [(i, index, row, features(row, base.get(i, [])))
        for i in ids for index, row in enumerate(merged[i]) if index >= len(incumbent[i])]


def accept_record(model, feature):
    if model["status"] != "fit":
        return False
    values = (np.asarray(feature)-np.asarray(model["mean"])) / np.asarray(model["scale"])
    logit = float(values @ np.asarray(model["coef"]) + model["intercept"])
    return logit >= math.log(.9/.1)


def apply_models(base, auxiliary, image_folds, thresholds, models):
    output, counts = {}, {"candidates": 0, "accepted": 0}
    for fold in range(3):
        ids = sorted(i for i, f in image_folds.items() if f == fold)
        incumbent, _, rows = candidates(base, auxiliary, ids, thresholds[fold])
        for i, _, row, feature in rows:
            counts["candidates"] += 1
            if accept_record(models[str(fold)][coarse(row["category_id"])], feature):
                incumbent[i].append(row)
                counts["accepted"] += 1
        output.update(incumbent)
    return output, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p40", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    args = parser.parse_args()
    out = args.collection / "calibration"
    out.mkdir(exist_ok=False)
    config_path = Path("configs/experiments/p40_rot90_supported_rescue_v1.json")
    config = json.loads(config_path.read_text())
    started = json.loads((args.collection / "preflight.json").read_text())
    if sha(config_path) != started["contract_sha256"]:
        raise ValueError("calibration contract changed")
    summary = json.loads((args.collection / "summary.json").read_text())
    if sha(args.collection / "predictions.json") != summary["predictions_sha256"]:
        raise ValueError("rot90 predictions changed")
    gt_path = args.p40 / "aggregate/ground_truth.json"
    if sha(gt_path) != config["normal_gt_sha256"]:
        raise ValueError("Normal GT changed")
    gt = load_coco_ground_truth(gt_path)
    doc = json.loads(gt_path.read_text())
    folds = {r["id"]: r["fold"] for r in doc["images"]}
    base_path = args.p40 / "aggregate/predictions_low.json"
    base, auxiliary = load_coco_predictions(base_path), load_coco_predictions(args.collection / "predictions.json")
    if (set(base) | set(auxiliary)) - set(gt):
        raise ValueError("predictions outside Normal universe")
    frontier = json.loads((args.p40 / "aggregate/crossfit_frontier.json").read_text())
    thresholds = {int(k): v for k, v in frontier["frontiers"]["0.150"]["crossfit_thresholds"].items()}
    protocol = parse_evaluation_protocol(load_config("configs/project.yaml"))
    models, support = {}, {}
    for heldout in range(3):
        ids = sorted(i for i in gt if folds[i] != heldout)
        train_gt = {i: gt[i] for i in ids}
        # One threshold fitted without heldout is also used on both fitting folds.
        # Never use each fitting fold's own threshold, which could depend on heldout labels.
        _, merged, rows = candidates(base, auxiliary, ids, thresholds[heldout])
        _, trace = evaluate_predictions_with_trace(train_gt, merged,
            class_names=protocol.class_names, category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds)
        positives = {(m.image_id, m.prediction_index) for m in trace.matches}
        models[str(heldout)], support[str(heldout)] = {}, {}
        for name in COARSE:
            subset = [(i, idx, r, f) for i, idx, r, f in rows if coarse(r["category_id"]) == name]
            x = np.asarray([r[3] for r in subset], dtype=float)
            y = np.asarray([(r[0], r[1]) in positives for r in subset], dtype=int)
            positive, negative = int(y.sum()), int(len(y)-y.sum())
            support[str(heldout)][name] = {"positive": positive, "negative": negative,
                                         "fitting_images": len(ids), "excluded_fold": heldout}
            if min(positive, negative) < 10:
                models[str(heldout)][name] = {"status": "abstain_insufficient_support"}
                continue
            scaler = StandardScaler().fit(x)
            learner = LogisticRegression(C=1, max_iter=1000, random_state=42).fit(scaler.transform(x), y)
            models[str(heldout)][name] = {"status": "fit", "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(), "coef": learner.coef_[0].tolist(),
                "intercept": float(learner.intercept_[0]), "iterations": int(learner.n_iter_[0])}
    result = {"status": "normal_only_crossfit_fit_complete", "models": models,
              "support": support, "thresholds": thresholds, "config": config,
              "config_sha256": sha(config_path),
              "inputs_sha256": {"gt": sha(gt_path), "base": sha(base_path),
                  "auxiliary": sha(args.collection / "predictions.json")},
              "fit_hard_sentinel": False, "not_nested_detector_retraining": True,
              "formal_admission": False}
    (out / "models.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "support": support}, indent=2))


if __name__ == "__main__":
    main()
