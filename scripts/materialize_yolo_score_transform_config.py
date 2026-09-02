#!/usr/bin/env python3
"""Materialize one frozen YOLO inference config with a score transform."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--transform", choices=("coarse_purity_sqrt",), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("model", {}).get("family") != "yolo":
        raise ValueError("base config must describe a YOLO model")
    if int(payload["model"].get("imgsz", 0)) != 1280:
        raise ValueError("this frozen transform screen requires imgsz=1280")
    if float(payload["model"].get("confidence", -1.0)) != 0.001:
        raise ValueError("this frozen transform screen requires confidence=0.001")
    payload["model"]["score_transform"] = args.transform
    payload["output_json"] = str(args.predictions.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"YOLO_SCORE_TRANSFORM_CONFIG_PASS output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
