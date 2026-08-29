#!/usr/bin/env python3
"""把训练 checkpoint 收缩为不依赖增强库的可部署 Ultralytics 权重。"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import torch

from rsdet.data.xh_dataset import FINE_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("model") is None:
        raise ValueError("输入不是 Ultralytics checkpoint")
    model = checkpoint.get("ema") or checkpoint["model"]
    names = {index: name for index, name in enumerate(FINE_NAMES)}
    model.names = names
    if hasattr(model, "yaml") and isinstance(model.yaml, dict):
        model.yaml["nc"] = len(names)
        model.yaml["names"] = names
    model.args = {"task": "detect", "imgsz": 1024, "single_cls": False}
    if hasattr(model, "criterion"):
        model.criterion = None
    model.half().eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    try:
        from ultralytics import __version__ as ultralytics_version
    except ImportError:
        ultralytics_version = "unknown"
    deploy = {
        "date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "version": ultralytics_version,
        "license": checkpoint.get("license", "AGPL-3.0"),
        "docs": checkpoint.get("docs", "https://docs.ultralytics.com"),
        "epoch": -1,
        "best_fitness": None,
        "model": model,
        "ema": None,
        "updates": None,
        "optimizer": None,
        "scaler": None,
        "train_args": {"task": "detect", "imgsz": 1024, "single_cls": False},
        "train_metrics": {},
        "train_results": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deploy, args.output)
    payload = {
        "status": "sanitized",
        "source": str(args.input.resolve()),
        "source_sha256": _sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "output_size": args.output.stat().st_size,
        "fine_names": list(FINE_NAMES),
        "removed_training_state": True,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
