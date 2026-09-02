#!/usr/bin/env python3
"""Materialize the frozen six-class MacroExpert-M training view.

The source dataset remains immutable.  Images are symlinked into an isolated
view and labels are remapped as follows: ship fine classes 0..3 stay 0..3,
vehicle 24 becomes 4, and all aircraft classes 4..23 become reject class 5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

NAMES = ["HM", "LQS", "QHS", "MS", "vehicle", "AIRCRAFT_REJECT"]
DEFAULT_REPEAT = {0: 12, 1: 8, 2: 2, 3: 1, 4: 8}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_path(data_root: Path, relative_image: str) -> Path:
    parts = list(Path(relative_image).parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError as error:
        raise ValueError(f"image path has no images component: {relative_image}") from error
    return (data_root / Path(*parts)).with_suffix(".txt")


def _read_and_remap(path: Path) -> tuple[list[str], list[int]]:
    rows: list[str] = []
    labels: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO fields")
        source = int(fields[0])
        if not 0 <= source < 25:
            raise ValueError(f"{path}:{line_number}: category {source} outside [0,24]")
        target = source if source <= 3 else 4 if source == 24 else 5
        rows.append(" ".join([str(target), *fields[1:]]))
        labels.append(target)
    return rows, labels


def _keep_aircraft_only(image_id: int, fraction: float, seed: int) -> bool:
    token = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    value = int.from_bytes(token[:8], "big") / 2**64
    return value < fraction


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(source.resolve(), target.parent.resolve())
    target.symlink_to(relative)


def materialize(
    *, manifest: Path, data_root: Path, output: Path, fold: int,
    aircraft_only_keep: float, seed: int, repeats: dict[int, int],
) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest must contain non-empty samples")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {output}")
    if not 0.0 <= aircraft_only_keep <= 1.0:
        raise ValueError("aircraft-only keep fraction must be in [0,1]")

    original_counts: Counter[int] = Counter()
    materialized_counts: Counter[int] = Counter()
    original_counts_by_split: dict[str, Counter[int]] = {
        "train": Counter(), "val": Counter()
    }
    materialized_counts_by_split: dict[str, Counter[int]] = {
        "train": Counter(), "val": Counter()
    }
    original_images: Counter[str] = Counter()
    materialized_images: Counter[str] = Counter()
    split_ids: dict[str, set[int]] = {"train": set(), "val": set()}
    split_groups: dict[str, set[str]] = {"train": set(), "val": set()}
    dropped_aircraft_only: list[int] = []

    for sample in samples:
        image_id = int(sample["image_id"])
        relative_image = str(sample["relative_path"])
        split = "val" if int(sample["fold"]) == fold else "train"
        image_path = data_root / relative_image
        label_path = _label_path(data_root, relative_image)
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"missing image/label for image_id={image_id}")
        rows, labels = _read_and_remap(label_path)
        original_counts.update(labels)
        original_counts_by_split[split].update(labels)
        original_images[split] += 1
        copies = 1
        if split == "train":
            target_labels = [label for label in labels if label != 5]
            if target_labels:
                copies = max(repeats[label] for label in target_labels)
            elif labels and not _keep_aircraft_only(image_id, aircraft_only_keep, seed):
                dropped_aircraft_only.append(image_id)
                continue
        split_ids[split].add(image_id)
        split_groups[split].add(str(sample.get("group_id", "")))
        for copy_index in range(copies):
            stem = f"img{image_id:06d}_r{copy_index:02d}"
            image_target = output / "images" / split / f"{stem}{image_path.suffix.lower()}"
            label_target = output / "labels" / split / f"{stem}.txt"
            _link(image_path, image_target)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            materialized_counts.update(labels)
            materialized_counts_by_split[split].update(labels)
            materialized_images[split] += 1

    if split_ids["train"] & split_ids["val"]:
        raise AssertionError("image leakage across train/val")
    group_overlap = sorted((split_groups["train"] & split_groups["val"]) - {""})
    if group_overlap:
        raise AssertionError(f"group leakage across train/val: {group_overlap[:5]}")

    dataset = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 6,
        "names": {index: name for index, name in enumerate(NAMES)},
    }
    dataset_path = output / "dataset.yaml"
    dataset_path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    audit: dict[str, Any] = {
        "schema_version": "macroexpert_fold_view_v1",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "data_root": str(data_root.resolve()),
        "fold": fold,
        "seed": seed,
        "class_mapping": {"0": 0, "1": 1, "2": 2, "3": 3, "4..23": 5, "24": 4},
        "repeat_factors": {str(key): value for key, value in sorted(repeats.items())},
        "aircraft_only_keep_fraction": aircraft_only_keep,
        "original_image_counts": dict(original_images),
        "materialized_image_counts": dict(materialized_images),
        "original_box_counts_remapped": {str(k): original_counts[k] for k in range(6)},
        "materialized_box_counts": {str(k): materialized_counts[k] for k in range(6)},
        "original_box_counts_by_split": {
            split: {str(k): counts[k] for k in range(6)}
            for split, counts in original_counts_by_split.items()
        },
        "materialized_box_counts_by_split": {
            split: {str(k): counts[k] for k in range(6)}
            for split, counts in materialized_counts_by_split.items()
        },
        "unique_train_images": len(split_ids["train"]),
        "unique_val_images": len(split_ids["val"]),
        "dropped_aircraft_only_count": len(dropped_aircraft_only),
        "dropped_aircraft_only_ids_sha256": hashlib.sha256(
            json.dumps(dropped_aircraft_only).encode()
        ).hexdigest(),
        "cross_split_image_count": 0,
        "cross_split_group_count": 0,
        "dataset_yaml": str(dataset_path.resolve()),
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--aircraft-only-keep", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = materialize(
        manifest=args.manifest,
        data_root=args.data_root,
        output=args.output,
        fold=args.fold,
        aircraft_only_keep=args.aircraft_only_keep,
        seed=args.seed,
        repeats=DEFAULT_REPEAT,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
