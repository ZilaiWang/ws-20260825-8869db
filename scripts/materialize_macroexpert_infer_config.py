#!/usr/bin/env python3
"""Create a low-floor MacroExpert inference config from a frozen CV3 template."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    payload = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    model = payload["model"]
    model["checkpoint"] = str(args.checkpoint.resolve())
    model["imgsz"] = args.imgsz
    model["confidence"] = 0.001
    model["label_map"] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 24}
    model["drop_labels"] = [5]
    payload["device"] = args.device
    payload["batch_size"] = 1
    payload["output_json"] = str(args.predictions.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
