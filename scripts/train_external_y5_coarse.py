#!/usr/bin/env python3
"""Fixed-epoch four-class external coarse/objectness pretraining for Y5-S."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_contract(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = payload.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    if list(names or []) != ["aircraft", "ship", "vehicle", "other_remote_object"]:
        raise ValueError(f"external dataset must expose frozen four-class names, got {names}")
    return {"dataset_yaml": str(path.resolve()), "dataset_yaml_sha256": _sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if _sha256(args.weights) != args.expected_weight_sha256.lower():
        raise ValueError("initial weight SHA mismatch")
    dataset = _dataset_contract(args.dataset)
    checkpoint = args.output_dir / "runs/foundation/weights/last.pt"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    train_args = {
        "data": str(args.dataset.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "optimizer": "AdamW",
        "lr0": 0.002,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "cos_lr": True,
        "amp": True,
        "deterministic": True,
        "seed": args.seed,
        "patience": 0,
        "val": False,
        "plots": False,
        "close_mosaic": 10,
        "device": args.device,
        "project": str((args.output_dir / "runs").resolve()),
        "name": "foundation",
        "exist_ok": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "status": "dry_run" if args.dry_run else "training_requested",
        "protocol": "external_four_coarse_y5s_fixed_epoch_v1",
        "initial_weight": str(args.weights.resolve()),
        "initial_weight_sha256": _sha256(args.weights),
        "dataset": dataset,
        "checkpoint_selection": "fixed_epoch_last",
        "uses_validation_for_selection": False,
        "train_args": train_args,
    }
    (args.output_dir / "training_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print("EXTERNAL_Y5_COARSE_DRY_RUN_PASS")
        return 0
    from ultralytics import YOLO

    from rsdet.innovation.trainers import rotate90_augmentations

    model = YOLO(str(args.weights.resolve()))
    model.train(augmentations=rotate90_augmentations(p=1.0), **train_args)
    if not checkpoint.is_file():
        raise RuntimeError("external training completed without last.pt")
    result = {
        **contract,
        "status": "complete",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
    }
    (args.output_dir / "training_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("EXTERNAL_Y5_COARSE_TRAINING_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
