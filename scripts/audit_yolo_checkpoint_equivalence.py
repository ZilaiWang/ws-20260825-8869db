#!/usr/bin/env python3
"""Audit inference equivalence of an original and sanitized YOLO checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_detections(result: Any) -> np.ndarray:
    """Return class/score/xyxy rows in a deterministic comparison order."""

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float64)
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float64, copy=False)
    scores = boxes.conf.detach().cpu().numpy().astype(np.float64, copy=False)
    classes = boxes.cls.detach().cpu().numpy().astype(np.float64, copy=False)
    rows = np.column_stack((classes, scores, xyxy))
    order = np.lexsort(
        (rows[:, 5], rows[:, 4], rows[:, 3], rows[:, 2], -rows[:, 1], rows[:, 0])
    )
    return rows[order]


def compare_detections(
    reference: np.ndarray, candidate: np.ndarray, *, atol: float
) -> dict[str, Any]:
    shape_equal = reference.shape == candidate.shape
    class_equal = bool(
        shape_equal and np.array_equal(reference[:, 0], candidate[:, 0])
    )
    max_score_delta = (
        float(np.max(np.abs(reference[:, 1] - candidate[:, 1])))
        if shape_equal and len(reference)
        else (0.0 if shape_equal else float("inf"))
    )
    max_box_delta = (
        float(np.max(np.abs(reference[:, 2:] - candidate[:, 2:])))
        if shape_equal and len(reference)
        else (0.0 if shape_equal else float("inf"))
    )
    passed = bool(
        shape_equal
        and class_equal
        and max_score_delta <= atol
        and max_box_delta <= atol
    )
    return {
        "passed": passed,
        "reference_detections": int(len(reference)),
        "candidate_detections": int(len(candidate)),
        "shape_equal": shape_equal,
        "class_equal": class_equal,
        "max_score_abs_delta": max_score_delta,
        "max_box_abs_delta": max_box_delta,
        "atol": atol,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--sanitized", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    from ultralytics import YOLO

    for path in (args.original, args.sanitized, *args.images):
        if not path.is_file():
            raise FileNotFoundError(path)
    predict_args = {
        "imgsz": args.imgsz,
        "conf": args.confidence,
        "iou": args.iou,
        "max_det": args.max_det,
        "device": args.device,
        "half": args.device != "cpu",
        "verbose": False,
    }
    original = YOLO(str(args.original))
    sanitized = YOLO(str(args.sanitized))
    reference_results = original.predict([str(path) for path in args.images], **predict_args)
    candidate_results = sanitized.predict([str(path) for path in args.images], **predict_args)
    comparisons = []
    for path, reference, candidate in zip(
        args.images, reference_results, candidate_results, strict=True
    ):
        comparison = compare_detections(
            canonical_detections(reference),
            canonical_detections(candidate),
            atol=args.atol,
        )
        comparisons.append({"image": str(path.resolve()), **comparison})
    passed = all(item["passed"] for item in comparisons)
    payload = {
        "status": "pass" if passed else "fail",
        "protocol": "yolo_sanitized_checkpoint_prediction_equivalence_v1",
        "original_sha256": _sha256(args.original),
        "sanitized_sha256": _sha256(args.sanitized),
        "predict_args": predict_args,
        "images": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
