#!/usr/bin/env python3
"""用全部官方训练图固定 epoch 拟合最终 Y5-Rot90 检测器。

该脚本不使用验证集选择 checkpoint：CV3 只负责冻结设计与超参数，最终训练将
全部 4481 张图放入 train，``val=False``，部署只采用固定 epoch 的 ``last.pt``。
"""

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


def _load_samples(manifest: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("CV3 manifest 缺少 samples")
    image_ids = [int(sample["image_id"]) for sample in samples]
    paths = [str(sample["relative_path"]) for sample in samples]
    if len(image_ids) != len(set(image_ids)) or len(paths) != len(set(paths)):
        raise ValueError("CV3 manifest 的 image_id 或 relative_path 不唯一")
    return sorted(samples, key=lambda item: int(item["image_id"]))


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def materialize_full_dataset(
    manifest: Path, data_root: Path, output_dir: Path
) -> tuple[Path, dict[str, Any]]:
    samples = _load_samples(manifest)
    paths = [(data_root / str(item["relative_path"])).resolve() for item in samples]
    missing_images = [str(path) for path in paths if not path.is_file()]
    missing_labels = [str(_label_path(path)) for path in paths if not _label_path(path).is_file()]
    if missing_images or missing_labels:
        raise FileNotFoundError(
            f"全量数据不完整: missing_images={len(missing_images)} "
            f"missing_labels={len(missing_labels)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_list = output_dir / "all_train_images.txt"
    train_list.write_text("\n".join(map(str, paths)) + "\n", encoding="utf-8")
    dataset = output_dir / "dataset_full.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "path": str(data_root.resolve()),
                "train": str(train_list.resolve()),
                # Ultralytics 会校验字段存在；val=False 保证不运行验证或据此选模。
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
        "status": "full_dataset_materialized",
        "image_count": len(paths),
        "unique_image_count": len(set(paths)),
        "label_count": len(paths) - len(missing_labels),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "train_list": str(train_list.resolve()),
        "train_list_sha256": _sha256(train_list),
        "dataset_yaml": str(dataset.resolve()),
        "dataset_yaml_sha256": _sha256(dataset),
        "uses_validation_for_selection": False,
    }
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return dataset, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", default="Y5-FULL-S")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Ultralytics device contract, e.g. cuda:0, 0, or 0,1,2 for DDP",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rotate90-p", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for path in (args.manifest, args.data_root, args.weights):
        if not path.exists():
            raise FileNotFoundError(path)
    actual_sha = _sha256(args.weights)
    if actual_sha != args.expected_weight_sha256.lower():
        raise ValueError(f"预训练权重 SHA 不匹配: {actual_sha}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        raise ValueError("epochs/imgsz/batch 必须 >0，workers 必须 >=0")
    if not str(args.device).strip():
        raise ValueError("device 不能为空")
    if not 0.0 <= args.rotate90_p <= 1.0:
        raise ValueError("rotate90-p 必须在 [0,1]")
    if (args.output_dir / "runs" / "foundation" / "weights" / "last.pt").exists():
        raise FileExistsError("固定全量训练已存在 last.pt；禁止静默覆盖或 resume")
    dataset, audit = materialize_full_dataset(args.manifest, args.data_root, args.output_dir)
    if audit["image_count"] != 4481:
        raise ValueError(f"正式全量训练必须恰好 4481 图，实际 {audit['image_count']}")
    train_args = {
        "data": str(dataset),
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
        "close_mosaic": 20,
        "device": str(args.device),
        "project": str((args.output_dir / "runs").resolve()),
        "name": "foundation",
        "exist_ok": False,
    }
    contract = {
        "contract_version": "y5_full_fit_v1",
        "status": "dry_run" if args.dry_run else "training_requested",
        "model_key": args.model_key,
        "initial_weight": str(args.weights.resolve()),
        "initial_weight_sha256": actual_sha,
        "checkpoint_selection": "fixed_epoch_last",
        "uses_all_official_training_images": True,
        "uses_validation_for_selection": False,
        "rotate90_p": args.rotate90_p,
        "train_args": train_args,
        "dataset_audit": audit,
    }
    (args.output_dir / "training_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print("Y5_FULL_DRY_RUN_PASS")
        return 0
    from ultralytics import YOLO

    from rsdet.innovation.trainers import rotate90_augmentations

    model = YOLO(str(args.weights.resolve()))
    model.train(
        augmentations=rotate90_augmentations(p=args.rotate90_p),
        **train_args,
    )
    checkpoint = args.output_dir / "runs" / "foundation" / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise RuntimeError("全量训练结束但 last.pt 不存在")
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
    print("Y5_FULL_TRAINING_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
