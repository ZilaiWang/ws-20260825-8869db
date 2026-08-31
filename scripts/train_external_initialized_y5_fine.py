#!/usr/bin/env python3
"""Fine-tune a reviewed official fold from external coarse Y5 weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from rsdet.data.xh_dataset import FINE_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--external-weights", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--head-warmup-epochs", type=int, default=8)
    parser.add_argument("--freeze-layers", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 1 <= args.head_warmup_epochs <= args.epochs:
        raise ValueError("head-warmup-epochs must be in [1, epochs]")
    if args.freeze_layers <= 0:
        raise ValueError("freeze-layers must be positive")
    if _sha256(args.external_weights) != args.expected_weight_sha256.lower():
        raise ValueError("external weight SHA mismatch")
    dataset_payload = yaml.safe_load(args.dataset.read_text(encoding="utf-8"))
    names = dataset_payload.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    normalized_names = list(names or [])
    accepted_names = (list(FINE_NAMES), [str(index) for index in range(len(FINE_NAMES))])
    if normalized_names not in accepted_names:
        raise ValueError("fine-tune dataset must expose the frozen 25-class order")
    checkpoint = args.output_dir / "runs/foundation/weights/last.pt"
    warmup_checkpoint = args.output_dir / "runs/head_warmup/weights/last.pt"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    transfer_audit = args.output_dir / "head_transfer_audit.json"
    shared_train_args = {
        "data": str(args.dataset.resolve()),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 1,
        "cos_lr": True,
        "amp": True,
        "deterministic": True,
        "seed": args.seed,
        "patience": 0,
        "val": False,
        "plots": False,
        "close_mosaic": 5,
        "device": args.device,
    }
    head_warmup_args = {
        **shared_train_args,
        "epochs": args.head_warmup_epochs,
        "freeze": args.freeze_layers,
        "project": str((args.output_dir / "runs").resolve()),
        "name": "head_warmup",
        "exist_ok": False,
    }
    full_finetune_epochs = args.epochs - args.head_warmup_epochs
    full_finetune_args = {
        **shared_train_args,
        "epochs": full_finetune_epochs,
        "warmup_epochs": min(2, full_finetune_epochs),
        "project": str((args.output_dir / "runs").resolve()),
        "name": "foundation",
        "exist_ok": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "status": "dry_run" if args.dry_run else "training_requested",
        "protocol": "external_backbone_neck_to_reviewed_official_fine_staged_v2",
        "external_weight": str(args.external_weights.resolve()),
        "external_weight_sha256": _sha256(args.external_weights),
        "dataset_yaml_sha256": _sha256(args.dataset),
        "dataset_names": normalized_names,
        "semantic_fine_order": list(FINE_NAMES),
        "head_policy": "reset entire Detect module with deterministic seed",
        "staged_transfer": {
            "head_warmup_epochs": args.head_warmup_epochs,
            "freeze_layers": args.freeze_layers,
            "full_finetune_epochs": full_finetune_epochs,
            "optimizer_is_recreated_between_stages": True,
        },
        "checkpoint_selection": "fixed_epoch_last",
        "uses_validation_for_selection": False,
        "head_warmup_args": head_warmup_args,
        "full_finetune_args": full_finetune_args,
    }
    (args.output_dir / "training_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print("EXTERNAL_INITIALIZED_FINE_DRY_RUN_PASS")
        return 0
    from ultralytics import YOLO

    from rsdet.external.transfer import external_head_transfer_trainer
    from rsdet.innovation.trainers import rotate90_augmentations

    source = YOLO(str(args.external_weights.resolve()))
    source_state = source.model.state_dict()
    source_model_yaml = dict(source.model.yaml)
    trainer = external_head_transfer_trainer(
        source_state,
        transfer_audit,
        expected_target_nc=len(FINE_NAMES),
        reset_seed=args.seed + 90000,
        source_model_yaml=source_model_yaml,
    )
    model = YOLO(str(args.external_weights.resolve()))
    model.train(
        trainer=trainer,
        augmentations=rotate90_augmentations(p=1.0),
        **head_warmup_args,
    )
    if not warmup_checkpoint.is_file() or not transfer_audit.is_file():
        raise RuntimeError("head warmup completed without checkpoint or transfer audit")
    if full_finetune_epochs > 0:
        model = YOLO(str(warmup_checkpoint.resolve()))
        model.train(
            augmentations=rotate90_augmentations(p=1.0),
            **full_finetune_args,
        )
    else:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(warmup_checkpoint, checkpoint)
    if not checkpoint.is_file():
        raise RuntimeError("full fine training completed without final checkpoint")
    result = {
        **contract,
        "status": "complete",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "head_warmup_checkpoint": str(warmup_checkpoint.resolve()),
        "head_warmup_checkpoint_sha256": _sha256(warmup_checkpoint),
        "head_transfer_audit_sha256": _sha256(transfer_audit),
    }
    (args.output_dir / "training_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("EXTERNAL_INITIALIZED_FINE_TRAINING_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
