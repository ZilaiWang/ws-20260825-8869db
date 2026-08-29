#!/usr/bin/env python3
"""Fit the admitted RT-DETR-L M3 detector on all 4,481 official images.

Formal CV3 has already selected the architecture and 120-epoch contract.  This
final fit therefore has no validation-based checkpoint choice: it starts from
the frozen official pretrained asset and deploys the fixed-epoch ``last.pt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_full_y5 import materialize_full_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_train_args(*, dataset: Path, output_dir: Path, args: argparse.Namespace) -> dict:
    """Build the frozen full-data M3 train call."""

    return {
        "data": str(dataset),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "optimizer": "AdamW",
        "lr0": 0.0002,
        "lrf": 0.01,
        "weight_decay": 0.0001,
        "warmup_epochs": 3,
        "cos_lr": True,
        "amp": True,
        "deterministic": True,
        "seed": int(args.seed),
        "patience": 0,
        "val": False,
        "plots": False,
        "device": "cuda:0",
        "project": str((output_dir / "runs").resolve()),
        "name": "foundation",
        "exist_ok": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for path in (args.manifest, args.data_root, args.weights):
        if not path.exists():
            raise FileNotFoundError(path)
    actual_sha = _sha256(args.weights)
    if actual_sha != args.expected_weight_sha256.lower():
        raise ValueError(f"pretrained RT-DETR-L SHA mismatch: {actual_sha}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        raise ValueError("epochs/imgsz/batch must be positive and workers non-negative")
    checkpoint = args.output_dir / "runs" / "foundation" / "weights" / "last.pt"
    if checkpoint.exists():
        raise FileExistsError("full M3 last.pt already exists; overwrite/resume is forbidden")

    dataset, audit = materialize_full_dataset(args.manifest, args.data_root, args.output_dir)
    if audit["image_count"] != 4481:
        raise ValueError("formal full M3 fit requires exactly 4,481 images")
    train_args = build_train_args(dataset=dataset, output_dir=args.output_dir, args=args)
    contract = {
        "contract_version": "m3_full_fit_v1",
        "status": "dry_run" if args.dry_run else "training_requested",
        "model_key": "M3-FULL-RTDETR-L",
        "model_family": "rtdetr",
        "initial_weight": str(args.weights.resolve()),
        "initial_weight_sha256": actual_sha,
        "checkpoint_selection": "fixed_epoch_last",
        "uses_all_official_training_images": True,
        "uses_validation_for_selection": False,
        "scientific_basis": "formal M3 CV3 OOF fixed-risk dominance and Y5 complementarity",
        "train_args": train_args,
        "dataset_audit": audit,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print("M3_FULL_DRY_RUN_PASS")
        return 0

    from ultralytics import RTDETR

    model = RTDETR(str(args.weights.resolve()))
    model.train(**train_args)
    if not checkpoint.is_file():
        raise RuntimeError("full M3 training completed without last.pt")
    complete = {
        **contract,
        "status": "complete",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
    }
    (args.output_dir / "training_result.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("M3_FULL_TRAINING_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
