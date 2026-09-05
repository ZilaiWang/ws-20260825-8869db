#!/usr/bin/env python3
"""Collect 4481 fold-heldout auxiliary views for Normal-only rescue calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.contracts import InferenceSample
from rsdet.models.rot90_view import Rot90ViewDetector
from rsdet.models.ultralytics_adapter import UltralyticsDetector


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p40", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output / "status.txt"
    status.write_text("preflight\n")
    config_path = Path("configs/experiments/p40_rot90_supported_rescue_v1.json")
    config = json.loads(config_path.read_text())
    parent = json.loads(Path(config["parent_contract"]).read_text())
    gt_path = args.p40 / "aggregate/ground_truth.json"
    if sha(gt_path) != config["normal_gt_sha256"]:
        raise ValueError("Normal GT changed")
    gt = json.loads(gt_path.read_text())
    images = gt["images"]
    if len(images) != 4481 or len({r["id"] for r in images}) != 4481:
        raise ValueError("Normal image coverage mismatch")
    frontier_path = args.p40 / "aggregate/crossfit_frontier.json"
    if sha(frontier_path) != parent["frontier_sha256"]:
        raise ValueError("Normal threshold source mismatch")
    fold_ids = json.loads(frontier_path.read_text())["fold_image_ids"]
    assets, paths = {}, {}
    for fold in range(3):
        actual = {r["id"] for r in images if r["fold"] == fold}
        if actual != set(fold_ids[str(fold)]):
            raise ValueError("Normal fold provenance mismatch")
        path = args.p40 / f"fold_{fold}/adaptation/runs/resolution_adaptation/weights/last.pt"
        if sha(path) != parent["weights_sha256"][fold]:
            raise ValueError("P40 fold weight SHA mismatch")
        paths[fold] = path
        assets[str(path)] = sha(path)
    image_sha = {}
    for r in images:
        path = (args.data_root / r["file_name"]).resolve()
        if not path.is_relative_to(args.data_root.resolve()):
            raise ValueError("image escaped data root")
        image_sha[r["file_name"]] = sha(path)
    preflight = {"contract": config, "contract_sha256": sha(config_path),
                 "normal_gt_sha256": sha(gt_path), "weights": assets,
                 "image_sha256": image_sha, "created_unix": time.time(),
                 "code_sha256": {str(p): sha(p) for directory in ("src", "scripts", "configs")
                     for p in sorted(Path(directory).rglob("*"))
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}}
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2)+"\n")
    predictions, completed_images = [], []
    started = time.perf_counter()
    for fold in range(3):
        detector = UltralyticsDetector(family="yolo", imgsz=1280, confidence=0.001,
            iou=0.70, max_detections=500, half=True, agnostic_nms=False)
        detector.load(str(paths[fold]))
        detector.to("cuda:0")
        detector.eval()
        detector = Rot90ViewDetector(detector)
        records = sorted((r for r in images if r["fold"] == fold), key=lambda r: r["id"])
        fold_pred = []
        for offset in range(0, len(records), 4):
            batch = []
            for r in records[offset:offset+4]:
                with Image.open(args.data_root / r["file_name"]) as im:
                    pixels = np.asarray(im.convert("RGB"))
                height, width = pixels.shape[:2]
                if (width, height) != (r["width"], r["height"]):
                    raise ValueError("Normal image size mismatch")
                batch.append(InferenceSample(r["id"], pixels, width, height))
            for pred in detector.predict(batch):
                completed_images.append(pred.image_id)
                for box, score, label in zip(pred.boxes_xyxy, pred.scores, pred.labels, strict=True):
                    x1, y1, x2, y2 = map(float, box)
                    fold_pred.append({"image_id": pred.image_id, "source_fold": fold,
                        "category_id": int(label), "score": float(score),
                        "bbox": [x1, y1, x2-x1, y2-y1]})
            status.write_text(f"rot90_fold_{fold}_images_{min(offset+4,len(records))}_of_{len(records)}\n")
            if offset % 100 == 0:
                print(status.read_text().strip(), flush=True)
        (args.output / f"fold_{fold}_predictions.json").write_text(json.dumps(fold_pred)+"\n")
        predictions.extend(fold_pred)
        del detector
    if len(completed_images) != 4481 or set(completed_images) != {r["id"] for r in images}:
        raise ValueError("duplicate/missing Normal inference")
    pred_path = args.output / "predictions.json"
    pred_path.write_text(json.dumps(predictions)+"\n")
    summary = {"status": "normal_rot90_collection_complete", "images": len(completed_images),
               "wall_seconds": time.perf_counter()-started,
               "predictions": len(predictions), "predictions_sha256": sha(pred_path),
               "checkpoint_training_performed": False, "hard_sentinel_used_for_fit": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    status.write_text("complete_collection_waiting_normal_only_calibration\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
