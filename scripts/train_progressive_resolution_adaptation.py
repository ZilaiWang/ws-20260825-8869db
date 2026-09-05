#!/usr/bin/env python3
"""Low-learning-rate 1024→1280 progressive resolution adaptation.

The input must be a mature S1024 checkpoint. This candidate experiment is
deliberately short, disables mosaic, uses a small learning rate, and selects
the fixed final epoch without validation-driven checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=2e-4)
    parser.add_argument("--lrf", type=float, default=0.10)
    parser.add_argument("--rotate90-p", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--weak-rfs", action="store_true", help="experimental 25-class capped weak-image sampling")
    parser.add_argument(
        "--weak-eqlv2",
        action="store_true",
        help="targeted EQLv2 classification loss for Ship/Vehicle weak fine classes",
    )
    args = parser.parse_args()

    if not args.weights.is_file() or not args.data.is_file():
        raise FileNotFoundError("weights and dataset YAML must exist")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {args.output}")
    if min(args.epochs, args.imgsz, args.batch) <= 0 or args.workers < 0:
        raise ValueError("epochs/imgsz/batch must be positive; workers non-negative")
    if args.lr0 <= 0.0 or not 0.0 < args.lrf <= 1.0:
        raise ValueError("lr0 must be positive and lrf in (0, 1]")
    if not 0.0 <= args.rotate90_p <= 1.0:
        raise ValueError("rotate90-p must be in [0, 1]")
    if args.weak_rfs and args.weak_eqlv2:
        raise ValueError("weak-rfs and weak-eqlv2 are separate single-factor experiments")

    args.output.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": "progressive_resolution_adaptation_v1",
        "candidate_only": True,
        "initial_checkpoint": str(args.weights.resolve()),
        "initial_checkpoint_sha256": sha256(args.weights),
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": 0.0005,
        "warmup_epochs": 1,
        "mosaic": 0.0,
        "close_mosaic": args.epochs,
        "rotate90_p": args.rotate90_p,
        "checkpoint_selection": "fixed_last",
        "uses_validation_for_selection": False,
    }
    if args.weak_rfs:
        contract["sampling"] = {"method": "weak_rfs_v1", "target_image_frequency": .10,
            "cap": 3., "samples_per_epoch": "same_as_natural", "replacement": True,
            "targets": [0, 1, 2, 3, 24], "class_head_unchanged": True}
    if args.weak_eqlv2:
        contract["classification_loss"] = {
            "method": "targeted_eqlv2_v1",
            "focus_class_indices": [0, 1, 2, 3, 24],
            "gamma": 12.0,
            "mu": 0.8,
            "alpha": 4.0,
            "soft_task_aligned_targets_preserved": True,
            "gradient_statistics": "weighted absolute logit gradient, accumulated independently in one2many and one2one branches",
            "other_class_elements": "unchanged_bce",
        }
    (args.output / "training_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print("PROGRESSIVE_RESOLUTION_ADAPTATION_DRY_RUN_PASS")
        return 0

    from ultralytics import YOLO

    from rsdet.innovation.trainers import rotate90_augmentations

    model = YOLO(str(args.weights.resolve()))
    trainer_kwargs = {}
    if args.weak_rfs:
        from rsdet.innovation.weak_rfs import weak_rfs_trainer

        trainer_kwargs["trainer"] = weak_rfs_trainer()
    elif args.weak_eqlv2:
        from rsdet.innovation.eqlv2 import eqlv2_trainer

        trainer_kwargs["trainer"] = eqlv2_trainer(
            focus_class_indices=(0, 1, 2, 3, 24), gamma=12.0, mu=0.8, alpha=4.0
        )
    model.train(
        **trainer_kwargs,
        augmentations=rotate90_augmentations(p=args.rotate90_p),
        data=str(args.data.resolve()),
        project=str((args.output / "runs").resolve()),
        name="resolution_adaptation",
        exist_ok=False,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=0.0005,
        warmup_epochs=1,
        cos_lr=True,
        amp=True,
        deterministic=True,
        patience=0,
        val=False,
        plots=False,
        mosaic=0.0,
        close_mosaic=args.epochs,
    )
    checkpoint = args.output / "runs" / "resolution_adaptation" / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise RuntimeError("training completed without last.pt")
    if args.weak_eqlv2:
        from rsdet.innovation.eqlv2 import eqlv2_criterion_audit

        audit = eqlv2_criterion_audit(model.trainer.model)
        (args.output / "runs" / "resolution_adaptation" / "eqlv2_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    result = {
        **contract,
        "status": "complete_candidate",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
    }
    (args.output / "training_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PROGRESSIVE_RESOLUTION_ADAPTATION_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
