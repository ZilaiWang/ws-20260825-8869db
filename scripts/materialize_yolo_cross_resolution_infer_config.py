#!/usr/bin/env python3
"""Change only YOLO inference resolution and output path in a frozen config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def materialize(
    source: dict[str, Any],
    *,
    imgsz: int,
    output_json: Path,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    payload = yaml.safe_load(yaml.safe_dump(source))
    model = payload.get("model")
    if not isinstance(model, dict) or model.get("family") != "yolo":
        raise ValueError("source config must contain model.family=yolo")
    if imgsz not in {1024, 1280}:
        raise ValueError("imgsz must be 1024 or 1280")
    if float(model.get("confidence", -1.0)) != 0.001:
        raise ValueError("cross-resolution inference requires frozen confidence=0.001")
    if int(model.get("max_detections", 0)) != 500:
        raise ValueError("cross-resolution inference requires frozen max_detections=500")
    tiling = payload.get("tiling")
    if not isinstance(tiling, dict) or bool(tiling.get("enabled")):
        raise ValueError("cross-resolution diagnostic requires frozen non-tiled inference")
    model["imgsz"] = imgsz
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model["checkpoint"] = str(checkpoint.resolve())
    payload["output_json"] = str(output_json.resolve())
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, choices=(1024, 1280), required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional replacement checkpoint; all other inference fields remain frozen.",
    )
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.output.exists() or args.predictions.exists():
        raise FileExistsError("refusing to overwrite cross-resolution output")
    source = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("source config must be a YAML object")
    payload = materialize(
        source,
        imgsz=args.imgsz,
        output_json=args.predictions,
        checkpoint=args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"CROSS_RESOLUTION_INFER_CONFIG_PASS output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
