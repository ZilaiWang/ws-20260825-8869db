#!/usr/bin/env python3
"""Audit YOLO Detect input layer indices, channels, strides and tensor shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rsdet.innovation.yolo_feature_quality import YoloPyramidTap


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    actual_sha = _sha256(args.weights)
    if actual_sha != args.expected_sha256.lower():
        raise ValueError("YOLO checkpoint SHA mismatch")
    from ultralytics import YOLO

    model = YOLO(str(args.weights)).model.to(args.device).eval()
    layers = model.model
    detect = layers[-1]
    indices = tuple(int(value) for value in detect.f)
    strides = tuple(int(value) for value in detect.stride.tolist())
    with YoloPyramidTap(model, indices) as tap, torch.inference_mode():
        model(torch.zeros((1, 3, args.imgsz, args.imgsz), device=args.device))
        features = tap.features(detach=True)
    payload = {
        "status": "pass",
        "protocol": "yolo_detect_input_feature_layer_audit_v1",
        "weight_sha256": actual_sha,
        "layer_count": len(layers),
        "detect_layer_index": len(layers) - 1,
        "feature_layer_indices": indices,
        "feature_strides": strides,
        "feature_shapes": [list(value.shape) for value in features],
        "feature_channels": [int(value.shape[1]) for value in features],
        "imgsz": args.imgsz,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
