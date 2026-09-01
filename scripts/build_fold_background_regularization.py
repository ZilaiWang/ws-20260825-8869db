#!/usr/bin/env python3
"""Build a leakage-safe, low-intensity Background-100MP fold training view."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from rsdet.data.background_regularization import (
    load_nonempty_lines,
    make_source_diverse_groups,
    materialize_background_mosaics,
    read_jsonl,
    select_training_background_rows,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--background-root", type=Path, required=True)
    parser.add_argument("--frozen-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    decision = json.loads(args.frozen_decision.read_text(encoding="utf-8"))
    if decision.get("formal_admission") is not True or decision.get("status") != "frozen":
        raise ValueError("Background-100MP is not formally frozen")
    manifest = args.background_root / "background_100mp_manifest.jsonl"
    manifest_sha = sha256_file(manifest)
    if manifest_sha != decision["manifest_sha256"]:
        raise ValueError("frozen Background-100MP manifest SHA mismatch")

    base = yaml.safe_load(args.base_dataset.read_text(encoding="utf-8"))
    train_list = Path(base["train"])
    val_list = Path(base["val"])
    train_images = load_nonempty_lines(train_list)
    val_images = load_nonempty_lines(val_list)
    rows, split_summary = select_training_background_rows(
        read_jsonl(manifest), train_images=train_images, val_images=val_images
    )
    groups, leftovers = make_source_diverse_groups(rows, group_size=4)
    if not groups:
        raise RuntimeError("no complete background mosaics can be materialized")

    args.output.mkdir(parents=True)
    records = materialize_background_mosaics(
        groups, background_root=args.background_root, output_root=args.output
    )
    augmented_train = train_images + [row["image_path"] for row in records]
    output_train = args.output / "train.txt"
    output_val = args.output / "val.txt"
    output_yaml = args.output / "dataset.yaml"
    output_train.write_text("\n".join(augmented_train) + "\n", encoding="utf-8")
    output_val.write_text("\n".join(val_images) + "\n", encoding="utf-8")
    output_config = dict(base)
    output_config["train"] = str(output_train.resolve())
    output_config["val"] = str(output_val.resolve())
    output_yaml.write_text(yaml.safe_dump(output_config, sort_keys=False), encoding="utf-8")

    materialized_manifest = args.output / "background_mosaic_manifest.jsonl"
    materialized_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    unique_sources = {
        item["source_file_name"] for row in records for item in row["inputs"]
    }
    payload = {
        "status": "ready_for_paired_selective_bce_screen",
        "protocol": "fold_background_100mp_low_intensity_v1",
        "base_dataset": str(args.base_dataset.resolve()),
        "base_dataset_sha256": sha256_file(args.base_dataset),
        "background_manifest_sha256": manifest_sha,
        "frozen_decision_sha256": sha256_file(args.frozen_decision),
        "fold_split": split_summary,
        "base_train_image_count": len(train_images),
        "validation_image_count": len(val_images),
        "selected_background_crop_count": len(rows),
        "used_background_crop_count": len(records) * 4,
        "unused_background_crop_count": len(leftovers),
        "background_source_count": len(unique_sources),
        "background_mosaic_count": len(records),
        "background_fraction_of_augmented_train": len(records) / len(augmented_train),
        "augmented_train_image_count": len(augmented_train),
        "native_crop_scale_preserved": True,
        "objects_per_background_label": 0,
        "source_leakage_to_validation": 0,
        "dataset_yaml": str(output_yaml.resolve()),
        "dataset_yaml_sha256": sha256_file(output_yaml),
        "materialized_manifest_sha256": sha256_file(materialized_manifest),
    }
    (args.output / "build_decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "ASSET_SHA256.txt").write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(args.output)}"
            for path in sorted(args.output.rglob("*"))
            if path.is_file() and path.name != "ASSET_SHA256.txt"
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
