#!/usr/bin/env python3
"""Train a fold-heldout three-coarse-class Y5 detector.

The experiment isolates label-space simplification from hard-background
training.  It transfers each admitted 25-class Y5-ROT fold checkpoint, rewrites
only the training labels to ship/aircraft/vehicle, and selects the fixed final
epoch without consulting the held-out fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.data.xh_dataset import COARSE_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coarse_id(fine_id: int) -> int:
    if 0 <= fine_id <= 3:
        return 0
    if 4 <= fine_id <= 23:
        return 1
    if fine_id == 24:
        return 2
    raise ValueError(f"invalid fine class id: {fine_id}")


def _source_label(image: Path) -> Path:
    parts = list(image.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _convert_label(source: Path) -> tuple[str, dict[int, int]]:
    counts = {0: 0, 1: 0, 2: 0}
    output: list[str] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{source}:{line_number}: expected five YOLO fields")
        target = coarse_id(int(fields[0]))
        counts[target] += 1
        output.append(" ".join([str(target), *fields[1:]]))
    return ("\n".join(output) + ("\n" if output else "")), counts


def materialize_coarse_dataset(
    manifest: Path, data_root: Path, held_out_fold: int, output_dir: Path
) -> tuple[Path, dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = [item for item in payload["samples"] if int(item["fold"]) != held_out_fold]
    if not rows:
        raise ValueError("no training images remain")
    image_dir = output_dir / "images" / "train"
    label_dir = output_dir / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    train_images: list[Path] = []
    class_counts = {0: 0, 1: 0, 2: 0}
    source_folds: set[int] = set()
    for item in rows:
        source_folds.add(int(item["fold"]))
        source_image = (data_root / str(item["relative_path"])).resolve()
        source_label = _source_label(source_image)
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(source_image if not source_image.is_file() else source_label)
        stem = f"image_{int(item['image_id']):05d}"
        target_image = image_dir / f"{stem}{source_image.suffix.lower()}"
        target_label = label_dir / f"{stem}.txt"
        if target_image.exists() or target_image.is_symlink():
            if target_image.resolve() != source_image:
                raise FileExistsError(target_image)
        else:
            os.symlink(source_image, target_image)
        converted, counts = _convert_label(source_label)
        target_label.write_text(converted, encoding="utf-8")
        for category_id, count in counts.items():
            class_counts[category_id] += count
        # Keep the symlink path in the train list.  Resolving it would make
        # Ultralytics infer the original 25-class label path instead of the
        # coarse label beside this symlink.
        train_images.append(target_image.absolute())
    if held_out_fold in source_folds:
        raise RuntimeError("held-out fold leaked into coarse training data")
    train_list = output_dir / "train_images.txt"
    train_list.write_text("\n".join(map(str, train_images)) + "\n", encoding="utf-8")
    dataset = output_dir / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "path": str(output_dir.resolve()),
                "train": str(train_list.resolve()),
                "val": str(train_list.resolve()),
                "nc": 3,
                "names": list(COARSE_NAMES),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    audit = {
        "status": "coarse_dataset_ready",
        "held_out_fold": held_out_fold,
        "source_folds": sorted(source_folds),
        "image_count": len(train_images),
        "class_counts": {COARSE_NAMES[key]: value for key, value in class_counts.items()},
        "manifest_sha256": _sha256(manifest),
        "train_list_sha256": _sha256(train_list),
        "dataset_sha256": _sha256(dataset),
    }
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return dataset, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--held-out-fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    actual_sha = _sha256(args.weights)
    if actual_sha != args.expected_weight_sha256.lower():
        raise ValueError("initial fold checkpoint SHA mismatch")
    checkpoint = args.output_dir / "runs" / "foundation" / "weights" / "last.pt"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset, audit = materialize_coarse_dataset(
        args.manifest, args.data_root, args.held_out_fold, args.output_dir
    )
    train_args = {
        "data": str(dataset),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "optimizer": "AdamW",
        "lr0": 0.0005,
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
        "device": "cuda:0",
        "project": str((args.output_dir / "runs").resolve()),
        "name": "foundation",
        "exist_ok": False,
    }
    contract = {
        "status": "dry_run" if args.dry_run else "training_requested",
        "protocol": "fold_heldout_y5_three_coarse_class_short_ft_v1",
        "held_out_fold": args.held_out_fold,
        "checkpoint_selection": "fixed_epoch_last",
        "uses_validation_for_selection": False,
        "initial_weight": str(args.weights.resolve()),
        "initial_weight_sha256": actual_sha,
        "dataset_audit": audit,
        "train_args": train_args,
    }
    (args.output_dir / "training_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print("Y5_COARSE_DETECTOR_DRY_RUN_PASS")
        return 0
    from ultralytics import YOLO

    from rsdet.innovation.trainers import rotate90_augmentations

    model = YOLO(str(args.weights.resolve()))
    model.train(augmentations=rotate90_augmentations(p=1.0), **train_args)
    if not checkpoint.is_file():
        raise RuntimeError("training completed without last.pt")
    result = {**contract, "status": "complete", "checkpoint": str(checkpoint.resolve()),
              "checkpoint_sha256": _sha256(checkpoint)}
    (args.output_dir / "training_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Y5_COARSE_DETECTOR_TRAINING_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
