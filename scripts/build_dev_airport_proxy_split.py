#!/usr/bin/env python3
"""Reuse B's dev_v1 and replace only MAR20 grouping with airport-proxy K60."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from rsdet.data.airport_proxy_split import (
    AirportProxyGroup,
    solve_grouped_validation_partition,
)
from rsdet.data.xh_dataset import FINE_NAMES, coarse_name

MAR20_RE = re.compile(r"^MAR20_(\d+)$")
AIRCRAFT_CLASS_IDS = tuple(range(4, 24))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_classes(path: Path) -> list[int]:
    classes: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO columns")
        class_id = int(parts[0])
        if not 0 <= class_id < len(FINE_NAMES):
            raise ValueError(f"{path}:{line_number}: invalid class_id={class_id}")
        classes.append(class_id)
    return classes


def load_group_map(path: Path) -> tuple[dict[str, str], str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if not rows:
        raise ValueError("airport-proxy group CSV is empty")
    image_column = "competition_image_id" if "competition_image_id" in rows[0] else "image_name"
    if image_column not in rows[0] or "group_id" not in rows[0]:
        raise ValueError("group CSV requires competition_image_id/image_name and group_id")
    mapping: dict[str, str] = {}
    for row in rows:
        stem = Path(row[image_column]).stem
        group_id = row["group_id"].strip()
        if stem in mapping:
            raise ValueError(f"duplicate group assignment for {stem}")
        if not group_id:
            raise ValueError(f"empty group_id for {stem}")
        mapping[stem] = group_id
    return mapping, image_column


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--group-map", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b-mapping-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--version", default="dev_v2_airport_proxy_k60")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args()

    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    samples = base.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("base manifest has no samples")
    if len({int(sample["image_id"]) for sample in samples}) != len(samples):
        raise ValueError("base manifest contains duplicate image_id")

    proxy_by_stem, source_image_column = load_group_map(args.group_map)
    mar20_samples = [
        sample for sample in samples if MAR20_RE.fullmatch(Path(sample["relative_path"]).stem)
    ]
    manifest_stems = {Path(sample["relative_path"]).stem for sample in mar20_samples}
    if manifest_stems != set(proxy_by_stem):
        raise ValueError(
            "MAR20 coverage mismatch: "
            f"manifest_only={sorted(manifest_stems - set(proxy_by_stem))[:5]}, "
            f"map_only={sorted(set(proxy_by_stem) - manifest_stems)[:5]}"
        )
    if len(set(proxy_by_stem.values())) != 60:
        raise ValueError("expected exactly 60 MAR20 airport-proxy groups")

    group_images: dict[str, list[int]] = {}
    group_classes: dict[str, Counter[int]] = {}
    group_old_split: dict[str, Counter[str]] = {}
    classes_by_image: dict[int, list[int]] = {}
    for sample in mar20_samples:
        stem = Path(sample["relative_path"]).stem
        label_path = args.label_dir / f"{stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        classes = read_classes(label_path)
        if not classes or any(class_id not in AIRCRAFT_CLASS_IDS for class_id in classes):
            raise ValueError(f"{stem} is not a non-empty aircraft-only sample")
        image_id = int(sample["image_id"])
        group_id = proxy_by_stem[stem]
        classes_by_image[image_id] = classes
        group_images.setdefault(group_id, []).append(image_id)
        group_classes.setdefault(group_id, Counter()).update(classes)
        group_old_split.setdefault(group_id, Counter()).update([sample["split"]])

    groups = [
        AirportProxyGroup(
            group_id=group_id,
            image_ids=tuple(sorted(group_images[group_id])),
            class_box_counts=group_classes[group_id],
            old_train_images=group_old_split[group_id]["train"],
            old_val_images=group_old_split[group_id]["val"],
        )
        for group_id in sorted(group_images)
    ]
    target_val_images = round(len(mar20_samples) * args.val_fraction)
    val_groups, solver = solve_grouped_validation_partition(
        groups,
        class_ids=AIRCRAFT_CLASS_IDS,
        target_val_images=target_val_images,
        val_fraction=args.val_fraction,
    )

    output_samples: list[dict[str, object]] = []
    changed_mar20_images = 0
    for sample in samples:
        updated = dict(sample)
        stem = Path(sample["relative_path"]).stem
        if stem in proxy_by_stem:
            group_id = proxy_by_stem[stem]
            split = "val" if group_id in val_groups else "train"
            changed_mar20_images += split != sample["split"]
            updated["split"] = split
            updated["group_id"] = group_id
            updated["group_rule"] = "mar20_airport_proxy_k60"
        output_samples.append(updated)

    # Non-MAR20 records are byte-for-field compatible with B's dev_v1 entries.
    for old, new in zip(samples, output_samples, strict=True):
        stem = Path(old["relative_path"]).stem
        if stem not in proxy_by_stem and old != new:
            raise RuntimeError(f"non-MAR20 sample changed unexpectedly: {stem}")

    split_by_group: dict[str, set[str]] = {}
    split_counts: Counter[str] = Counter()
    coarse_counts: Counter[tuple[str, str]] = Counter()
    fine_boxes: Counter[tuple[int, str]] = Counter()
    fine_images: dict[tuple[int, str], set[int]] = {}
    fine_groups: dict[tuple[int, str], set[str]] = {}
    for sample in output_samples:
        split = str(sample["split"])
        group_id = str(sample["group_id"])
        image_id = int(sample["image_id"])
        stem = Path(sample["relative_path"]).stem
        label_path = args.label_dir / f"{stem}.txt"
        classes = read_classes(label_path)
        split_counts[split] += 1
        split_by_group.setdefault(group_id, set()).add(split)
        coarse = coarse_name(classes[0])
        coarse_counts[(coarse, split)] += 1
        for class_id in classes:
            fine_boxes[(class_id, split)] += 1
            fine_images.setdefault((class_id, split), set()).add(image_id)
            fine_groups.setdefault((class_id, split), set()).add(group_id)
    crossing_groups = sorted(
        group_id for group_id, split_set in split_by_group.items() if len(split_set) > 1
    )
    if crossing_groups:
        raise RuntimeError(f"groups cross train/val: {crossing_groups[:5]}")
    missing_train = [c for c in range(len(FINE_NAMES)) if fine_boxes[(c, "train")] == 0]
    missing_val = [c for c in range(len(FINE_NAMES)) if fine_boxes[(c, "val")] == 0]
    if missing_train or missing_val:
        raise RuntimeError(
            f"class coverage failure: missing_train={missing_train}, missing_val={missing_val}"
        )

    manifest = {
        "version": args.version,
        "data_version": base["data_version"],
        "seed": base.get("seed", 42),
        "val_fraction": args.val_fraction,
        "base_manifest": str(args.base_manifest),
        "base_manifest_sha256": sha256(args.base_manifest),
        "group_map": str(args.group_map),
        "group_map_sha256": sha256(args.group_map),
        "group_semantics": "mar20_airport_proxy_visual_cluster_k60_not_ground_truth",
        "non_mar20_policy": "copied_exactly_from_base_manifest",
        "samples": output_samples,
    }
    write_json(args.output, manifest)

    args.b_mapping_output.parent.mkdir(parents=True, exist_ok=True)
    with args.b_mapping_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "group_id"])
        writer.writeheader()
        for sample in sorted(mar20_samples, key=lambda item: int(item["image_id"])):
            image_name = Path(sample["relative_path"]).name
            writer.writerow(
                {"image_name": image_name, "group_id": proxy_by_stem[Path(image_name).stem]}
            )

    class_rows = []
    for class_id, class_name in enumerate(FINE_NAMES):
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "train_boxes": fine_boxes[(class_id, "train")],
                "val_boxes": fine_boxes[(class_id, "val")],
                "train_images": len(fine_images.get((class_id, "train"), set())),
                "val_images": len(fine_images.get((class_id, "val"), set())),
                "train_groups": len(fine_groups.get((class_id, "train"), set())),
                "val_groups": len(fine_groups.get((class_id, "val"), set())),
            }
        )
    summary = {
        "status": "dev_v2_airport_proxy_k60_ready",
        "version": args.version,
        "base_manifest_sha256": sha256(args.base_manifest),
        "group_map_sha256": sha256(args.group_map),
        "source_group_image_column": source_image_column,
        "output_manifest": str(args.output),
        "output_manifest_sha256": sha256(args.output),
        "b_mapping_output": str(args.b_mapping_output),
        "b_mapping_sha256": sha256(args.b_mapping_output),
        "sample_count": len(output_samples),
        "split_counts": dict(sorted(split_counts.items())),
        "coarse_image_counts": {
            f"{coarse}_{split}": coarse_counts[(coarse, split)]
            for coarse in ("ship", "aircraft", "vehicle")
            for split in ("train", "val")
        },
        "mar20_image_count": len(mar20_samples),
        "mar20_group_count": len(groups),
        "mar20_train_groups": len(groups) - len(val_groups),
        "mar20_val_groups": len(val_groups),
        "mar20_target_val_images": target_val_images,
        "changed_mar20_images_vs_dev_v1": changed_mar20_images,
        "non_mar20_changed_images": 0,
        "cross_split_group_count": 0,
        "missing_train_classes": missing_train,
        "missing_val_classes": missing_val,
        "selected_mar20_val_groups": sorted(val_groups),
        "solver": solver,
        "per_class": class_rows,
    }
    write_json(args.summary_output, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
