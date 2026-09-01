#!/usr/bin/env python3
"""Adapt a mature YOLO26 detector with low-intensity background and a frozen teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distillation-weight", type=float, default=6.0)
    parser.add_argument("--freeze-layers", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.distillation_weight <= 0 or args.freeze_layers <= 0:
        raise ValueError("epochs, distillation weight, and freeze layers must be positive")
    actual_sha = _sha256(args.weights)
    if actual_sha != args.expected_weight_sha256.lower():
        raise ValueError("input weight SHA mismatch")
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
        "lr0": 0.00005,
        "lrf": 0.1,
        "weight_decay": 0.0005,
        "warmup_epochs": 0.5,
        "cos_lr": True,
        "amp": True,
        "deterministic": True,
        "seed": args.seed,
        "patience": 0,
        "val": False,
        "plots": False,
        "mosaic": 0.2,
        "close_mosaic": 2,
        "freeze": args.freeze_layers,
        "distill_model": str(args.weights.resolve()),
        "dis": args.distillation_weight,
        "device": args.device,
        "project": str((args.output_dir / "runs").resolve()),
        "name": "foundation",
        "exist_ok": False,
    }
    contract = {
        "status": "dry_run" if args.dry_run else "training_requested",
        "protocol": "mature_teacher_low_intensity_background_distillation_v1",
        "input_weight": str(args.weights.resolve()),
        "input_weight_sha256": actual_sha,
        "teacher_weight": str(args.weights.resolve()),
        "teacher_weight_sha256": actual_sha,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "checkpoint_selection": "fixed_epoch_last",
        "uses_validation_for_selection": False,
        "train_args": train_args,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print("MATURE_BACKGROUND_DISTILLATION_DRY_RUN_PASS")
        return 0

    from ultralytics import YOLO

    from rsdet.innovation.trainers import rotate90_augmentations

    model = YOLO(str(args.weights.resolve()))
    model.train(augmentations=rotate90_augmentations(p=1.0), **train_args)
    if not checkpoint.is_file():
        raise RuntimeError("distillation training produced no final checkpoint")
    result = {
        **contract,
        "status": "complete",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    (args.output_dir / "training_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("MATURE_BACKGROUND_DISTILLATION_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
