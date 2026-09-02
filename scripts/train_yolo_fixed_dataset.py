#!/usr/bin/env python3
"""Train one frozen YOLO candidate on an already materialized dataset view."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--fraction", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("fraction must be in (0,1]")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {args.output}")
    if not args.weights.is_file() or not args.data.is_file():
        raise FileNotFoundError("weights and dataset.yaml must exist")
    from ultralytics import YOLO

    from rsdet.innovation.trainers import rotate90_augmentations

    contract = {
        "schema_version": "macroexpert_train_contract_v1",
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256(args.weights),
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": str(args.device),
        "seed": args.seed,
        "checkpoint_selection": "last",
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "close_mosaic": max(args.epochs // 2, 1),
        "rotation90": True,
        "fraction": args.fraction,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "train_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    model = YOLO(str(args.weights))
    model.train(
        augmentations=rotate90_augmentations(p=1.0),
        data=str(args.data), project=str(args.output / "runs"), name="foundation",
        exist_ok=False, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=str(args.device), seed=args.seed,
        fraction=args.fraction,
        optimizer="AdamW", lr0=args.lr0, lrf=0.01, weight_decay=0.0005,
        warmup_epochs=3, cos_lr=True, amp=True, deterministic=True,
        patience=0, val=False, plots=False, close_mosaic=max(args.epochs // 2, 1),
    )
    last = args.output / "runs" / "foundation" / "weights" / "last.pt"
    if not last.is_file():
        raise FileNotFoundError(f"training completed without last checkpoint: {last}")
    (args.output / "result.json").write_text(
        json.dumps({**contract, "status": "complete", "last": str(last),
                    "last_sha256": sha256(last)}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
