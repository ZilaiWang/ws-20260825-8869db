#!/usr/bin/env python3
"""Materialize one frozen YOLO capacity/scale fold0 training contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def materialize(
    template: dict[str, Any],
    *,
    output_dir: Path,
    data_root: Path,
    split_view: Path,
    weights: Path,
    device: str,
) -> dict[str, Any]:
    payload = yaml.safe_load(yaml.safe_dump(template))
    if payload.get("seed") != 42:
        raise ValueError("capacity/scale screen requires seed=42")
    if payload.get("model", {}).get("family") != "yolo":
        raise ValueError("capacity/scale screen requires model.family=yolo")
    stages = payload.get("train", {}).get("stages")
    if not isinstance(stages, list) or len(stages) != 1:
        raise ValueError("capacity/scale screen requires exactly one training stage")
    if stages[0].get("epochs") != 40 or stages[0].get("name") != "foundation":
        raise ValueError("capacity/scale screen requires foundation/40 epochs")
    args = payload["train"].get("args", {})
    if args.get("batch") != 8 or args.get("close_mosaic") is not None:
        raise ValueError("frozen total batch is 8 and close_mosaic belongs to the stage")
    if stages[0].get("args", {}).get("close_mosaic") != 20:
        raise ValueError("frozen close_mosaic is 20")
    if args.get("imgsz") not in (1024, 1280):
        raise ValueError("frozen image size must be 1024 or 1280")

    payload["output_dir"] = str(output_dir.resolve())
    payload["model"]["weights"] = str(weights.resolve())
    payload["data"]["root"] = str(data_root.resolve())
    payload["data"]["manifest"] = str(split_view.resolve())
    payload["train"]["device"] = device
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-view", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite materialized contract: {args.output}")
    for path in (args.template, args.split_view, args.weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.data_root.is_dir():
        raise NotADirectoryError(args.data_root)
    template = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise ValueError("template must be a YAML object")
    payload = materialize(
        template,
        output_dir=args.output_dir,
        data_root=args.data_root,
        split_view=args.split_view,
        weights=args.weights,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"YOLO_CAPACITY_SCALE_CONFIG_PASS output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
