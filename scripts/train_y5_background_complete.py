#!/usr/bin/env python3
"""Short fold-heldout Y5 fine-tune with fold-pure hard-background tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from rsdet.data.xh_dataset import FINE_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def materialize_dataset(
    *,
    manifest: Path,
    data_root: Path,
    hard_negative_summary: Path,
    held_out_fold: int,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    official = [
        (data_root / str(item["relative_path"])).resolve()
        for item in samples
        if int(item["fold"]) != held_out_fold
    ]
    hard_payload = json.loads(hard_negative_summary.read_text(encoding="utf-8"))
    if int(hard_payload["held_out_fold"]) != held_out_fold:
        raise ValueError("hard-negative held-out fold mismatch")
    if held_out_fold in {int(value) for value in hard_payload["source_folds"]}:
        raise ValueError("hard-negative tiles contain held-out fold sources")
    hard = [Path(str(item["image"])) for item in hard_payload["tiles"]]
    paths = official + hard
    if not official or not hard or len(paths) != len(set(paths)):
        raise ValueError("dataset requires unique official and hard-negative images")
    missing_images = [path for path in paths if not path.is_file()]
    missing_labels = [_label_path(path) for path in paths if not _label_path(path).is_file()]
    if missing_images or missing_labels:
        raise FileNotFoundError(
            f"dataset incomplete: images={len(missing_images)} labels={len(missing_labels)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_list = output_dir / "train_images.txt"
    train_list.write_text("\n".join(map(str, paths)) + "\n", encoding="utf-8")
    dataset = output_dir / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "path": str(data_root.resolve()),
                "train": str(train_list.resolve()),
                "val": str(train_list.resolve()),
                "nc": len(FINE_NAMES),
                "names": list(FINE_NAMES),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    audit = {
        "status": "background_complete_dataset_ready",
        "held_out_fold": held_out_fold,
        "official_training_images": len(official),
        "hard_negative_tiles": len(hard),
        "total_training_images": len(paths),
        "manifest_sha256": _sha256(manifest),
        "hard_negative_summary_sha256": _sha256(hard_negative_summary),
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
    parser.add_argument("--hard-negative-summary", type=Path, required=True)
    parser.add_argument("--held-out-fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    actual_sha = _sha256(args.weights)
    if actual_sha != args.expected_weight_sha256.lower():
        raise ValueError("initial fold checkpoint SHA mismatch")
    checkpoint = args.output_dir / "runs" / "foundation" / "weights" / "last.pt"
    if checkpoint.exists():
        raise FileExistsError("background-complete checkpoint already exists")
    dataset, audit = materialize_dataset(
        manifest=args.manifest,
        data_root=args.data_root,
        hard_negative_summary=args.hard_negative_summary,
        held_out_fold=args.held_out_fold,
        output_dir=args.output_dir,
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
        "protocol": "fold_heldout_y5_background_complete_short_ft_v1",
        "held_out_fold": args.held_out_fold,
        "checkpoint_selection": "fixed_epoch_last",
        "initial_weight": str(args.weights.resolve()),
        "initial_weight_sha256": actual_sha,
        "uses_validation_for_selection": False,
        "dataset_audit": audit,
        "train_args": train_args,
    }
    (args.output_dir / "training_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print("Y5_BACKGROUND_COMPLETE_DRY_RUN_PASS")
        return 0

    from ultralytics import YOLO

    from rsdet.innovation.trainers import rotate90_augmentations

    model = YOLO(str(args.weights.resolve()))
    model.train(augmentations=rotate90_augmentations(p=1.0), **train_args)
    if not checkpoint.is_file():
        raise RuntimeError("training completed without last.pt")
    result = {
        **contract,
        "status": "complete",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    (args.output_dir / "training_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Y5_BACKGROUND_COMPLETE_TRAINING_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
