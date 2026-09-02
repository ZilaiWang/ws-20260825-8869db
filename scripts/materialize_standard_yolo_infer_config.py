#!/usr/bin/env python3
"""Create a clean low-floor inference config for a standard 25-class YOLO."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def materialize(
    base: dict[str, Any],
    *,
    checkpoint: Path,
    predictions: Path,
    data_root: Path,
    split_view: Path,
    imgsz: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    payload = yaml.safe_load(yaml.safe_dump(base))
    model = payload.get("model")
    if not isinstance(model, dict) or model.get("family") != "yolo":
        raise ValueError("base config must contain model.family=yolo")
    model["checkpoint"] = str(checkpoint.resolve())
    model["imgsz"] = imgsz
    model["confidence"] = 0.001
    model["iou"] = 0.70
    model["max_detections"] = 500
    model["half"] = True
    model["agnostic_nms"] = False
    for key in ("label_map", "drop_labels", "score_transform"):
        model.pop(key, None)
    input_config = payload.get("input")
    if not isinstance(input_config, dict):
        raise ValueError("base config must contain an input mapping")
    input_config["data_root"] = str(data_root.resolve())
    input_config["manifest"] = str(split_view.resolve())
    input_config["split"] = "val"
    payload["device"] = device
    payload["batch_size"] = batch_size
    payload["tiling"] = {"enabled": False, "force": False}
    payload["score_thresholds"] = {
        "ship": 0.001,
        "aircraft": 0.001,
        "vehicle": 0.001,
    }
    payload["fine_score_thresholds"] = {}
    payload["output_json"] = str(predictions.resolve())
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, choices=(1024, 1280), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite inference config: {args.output}")
    if not args.base.is_file() or not args.checkpoint.is_file() or not args.split_view.is_file():
        raise FileNotFoundError("base config, checkpoint, and split view must exist")
    if not args.data_root.is_dir():
        raise NotADirectoryError(args.data_root)
    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("base config must be a YAML object")
    payload = materialize(
        base,
        checkpoint=args.checkpoint,
        predictions=args.predictions,
        data_root=args.data_root,
        split_view=args.split_view,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"STANDARD_YOLO_INFER_CONFIG_PASS output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
