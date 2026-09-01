#!/usr/bin/env python3
"""Fine-tune only Ship/Vehicle classifier rows from a mature Y5 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--max-weight-relative-delta", type=float)
    parser.add_argument("--max-bias-delta", type=float)
    parser.add_argument(
        "--train-scope",
        choices=("final_rows", "classifier_branches"),
        default="final_rows",
    )
    parser.add_argument("--max-branch-relative-delta", type=float)
    parser.add_argument(
        "--classification-loss",
        choices=("bce", "ship_vehicle_hard_negative_focal"),
        default="ship_vehicle_hard_negative_focal",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    actual_sha = _sha256(args.weights)
    if actual_sha != args.expected_weight_sha256.lower():
        raise ValueError("input weight SHA mismatch")
    checkpoint = args.output_dir / "runs/selective_classifier/weights/last.pt"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)

    focus_classes = (0, 1, 2, 3, 24)
    contract = {
        "status": "training_requested",
        "protocol": "mature_incumbent_ship_vehicle_classifier_rows_only_v1",
        "input_weight": str(args.weights.resolve()),
        "input_weight_sha256": actual_sha,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "fine_names": list(FINE_NAMES),
        "focus_class_indices": list(focus_classes),
        "frozen": "backbone_neck_box_dfl_bn_and_nonfocused_classifier_rows",
        "loss": {
            "name": args.classification_loss,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "positive_weighting": "unit",
        },
        "epochs": args.epochs,
        "checkpoint_selection": "fixed_epoch_last",
        "uses_validation_for_selection": False,
        "projection": {
            "train_scope": args.train_scope,
            "max_branch_relative_delta": args.max_branch_relative_delta,
            "max_weight_relative_delta": args.max_weight_relative_delta,
            "max_bias_delta": args.max_bias_delta,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    from ultralytics import YOLO

    from rsdet.innovation.quality_aware_loss import (
        quality_aware_trainer,
        selective_classifier_trainer,
        spatial_classifier_residual_trainer,
    )
    from rsdet.innovation.trainers import rotate90_augmentations

    if args.classification_loss == "ship_vehicle_hard_negative_focal":
        loss_trainer = quality_aware_trainer(
            alpha=args.alpha,
            gamma=args.gamma,
            positive_weighting="unit",
            focus_class_indices=focus_classes,
        )
    else:
        from ultralytics.models.yolo.detect.train import DetectionTrainer

        loss_trainer = DetectionTrainer
    if args.train_scope == "classifier_branches":
        if args.max_branch_relative_delta is None:
            raise ValueError("classifier_branches requires --max-branch-relative-delta")
        if args.max_weight_relative_delta is None or args.max_bias_delta is None:
            raise ValueError("classifier_branches requires final-row projection bounds")
        trainer = spatial_classifier_residual_trainer(
            focus_class_indices=focus_classes,
            base_trainer=loss_trainer,
            max_branch_relative_delta=args.max_branch_relative_delta,
            max_final_weight_relative_delta=args.max_weight_relative_delta,
            max_final_bias_delta=args.max_bias_delta,
        )
    else:
        if args.max_branch_relative_delta is not None:
            raise ValueError("final_rows does not accept --max-branch-relative-delta")
        trainer = selective_classifier_trainer(
            focus_class_indices=focus_classes,
            base_trainer=loss_trainer,
            max_weight_relative_delta=args.max_weight_relative_delta,
            max_bias_delta=args.max_bias_delta,
        )
    model = YOLO(str(args.weights.resolve()))
    model.train(
        trainer=trainer,
        data=str(args.dataset.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        optimizer="AdamW",
        lr0=0.0002,
        lrf=0.1,
        weight_decay=0.0,
        warmup_epochs=1,
        cos_lr=True,
        amp=True,
        deterministic=True,
        seed=args.seed,
        patience=0,
        val=False,
        plots=False,
        close_mosaic=3,
        device=args.device,
        project=str((args.output_dir / "runs").resolve()),
        name="selective_classifier",
        exist_ok=False,
        augmentations=rotate90_augmentations(p=1.0),
    )
    if not checkpoint.is_file():
        raise RuntimeError("selective classifier training produced no final checkpoint")
    result = {
        **contract,
        "status": "complete",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    (args.output_dir / "training_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SELECTIVE_CLASSIFIER_FINETUNE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
