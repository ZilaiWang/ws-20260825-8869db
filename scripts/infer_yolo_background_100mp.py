#!/usr/bin/env python3
"""Run one YOLO checkpoint on the frozen Background-100MP image set."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rsdet.contracts import InferenceSample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [int(row["image_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("background manifest must be non-empty with unique image_id values")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, choices=(1024, 1280), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.manifest.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("manifest and checkpoint must exist")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    from rsdet.models.ultralytics_adapter import UltralyticsDetector

    rows = _load_manifest(args.manifest)
    detector = UltralyticsDetector(
        family="yolo",
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        max_detections=args.max_detections,
        half=True,
        agnostic_nms=False,
    )
    detector.load(str(args.checkpoint.resolve()))
    detector.to(args.device)
    detector.eval()

    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for offset in range(0, len(rows), args.batch_size):
        samples: list[InferenceSample] = []
        for row in rows[offset : offset + args.batch_size]:
            image_path = args.root / str(row["file_name"])
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            samples.append(
                InferenceSample(
                    image_id=int(row["image_id"]),
                    image=rgb,
                    width=int(rgb.shape[1]),
                    height=int(rgb.shape[0]),
                )
            )
        for prediction in detector.predict(samples):
            for box, score, label in zip(
                prediction.boxes_xyxy, prediction.scores, prediction.labels
            ):
                x1, y1, x2, y2 = [float(value) for value in box]
                predictions.append(
                    {
                        "image_id": int(prediction.image_id),
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    runtime = {
        "schema_version": "background_100mp_yolo_inference_v1",
        "image_count": len(rows),
        "prediction_count": len(predictions),
        "imgsz": args.imgsz,
        "confidence": args.confidence,
        "iou": args.iou,
        "max_detections": args.max_detections,
        "elapsed_seconds": elapsed,
        "seconds_per_crop": elapsed / len(rows),
        "manifest_sha256": _sha256(args.manifest),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "predictions_sha256": _sha256(args.output),
    }
    runtime_path = args.output.with_name(f"{args.output.stem}.runtime.json")
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    print(
        f"BACKGROUND_INFER_PASS images={len(rows)} proposals={len(predictions)} "
        f"seconds={elapsed:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
