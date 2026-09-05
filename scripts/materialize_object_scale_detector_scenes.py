#!/usr/bin/env python3
"""Build a leakage-safe MPSR-style scene-crop training supplement.

Each source image contributes at most one context-preserving crop centred on a
small Ship/Vehicle GT. All sufficiently visible annotations are retained and a
crop is rejected if it cuts through any annotation. The validation list is
copied unchanged and never supplies a crop. Full-data fitting must opt in
explicitly; in that mode train and val must be the same complete list and the
downstream trainer must disable validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rsdet.augmentation.jitter_hard_negative import Box  # noqa: E402
from rsdet.augmentation.object_scale_refinement import (  # noqa: E402
    ScaleCropPolicy,
    build_scale_crop,
)
from rsdet.augmentation.scene_scale_materialization import (  # noqa: E402
    paired_label_path,
    reflect_crop,
    yolo_rows_for_crop,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_paths(path: Path) -> list[Path]:
    rows = [Path(row.strip()).resolve() for row in path.read_text().splitlines() if row.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"path list must be non-empty and unique: {path}")
    return rows


def annotations(image_path: Path) -> tuple[list[int], list[Box], int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    label_path = paired_label_path(image_path)
    category_ids: list[int] = []
    boxes: list[Box] = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected five fields")
        category_id = int(fields[0])
        cx, cy, box_width, box_height = map(float, fields[1:])
        category_ids.append(category_id)
        boxes.append(
            Box(
                (cx - box_width / 2.0) * width,
                (cy - box_height / 2.0) * height,
                (cx + box_width / 2.0) * width,
                (cy + box_height / 2.0) * height,
            )
        )
    return category_ids, boxes, width, height


def stable_rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network-size", type=int, default=1280)
    parser.add_argument("--target-network-side", type=int, default=48)
    parser.add_argument("--ship-target-network-side", type=int)
    parser.add_argument("--vehicle-target-network-side", type=int)
    parser.add_argument("--output-pixels", type=int, default=800)
    parser.add_argument("--max-extra-images", type=int, default=744)
    parser.add_argument("--max-extra-fraction", type=float, default=0.25)
    parser.add_argument("--target-classes", type=int, nargs="+", default=[0, 1, 2, 3, 24])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--full-training",
        action="store_true",
        help="allow identical complete train/val lists for a downstream val=False full fit",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    class_target_sides = {
        "ship": args.ship_target_network_side or args.target_network_side,
        "vehicle": args.vehicle_target_network_side or args.target_network_side,
    }
    if min(args.network_size, args.target_network_side, args.output_pixels, *class_target_sides.values()) <= 0:
        raise ValueError("network and output sizes must be positive")
    if args.max_extra_images <= 0 or not 0.0 < args.max_extra_fraction <= 1.0:
        raise ValueError("invalid supplement cap")

    train_paths, val_paths = read_paths(args.train_list), read_paths(args.val_list)
    overlap = set(train_paths) & set(val_paths)
    if overlap and not args.full_training:
        raise ValueError(f"train/validation leakage: {len(overlap)} paths overlap")
    if args.full_training and set(train_paths) != set(val_paths):
        raise ValueError("full-training requires identical complete train and val path sets")
    target_classes = set(args.target_classes)
    policies = {
        coarse: ScaleCropPolicy(
            network_size=args.network_size,
            target_network_side=target_side,
            target_visibility=0.95,
            keep_visibility=0.70,
            reject_partial_visibility=0.05,
            center_jitter_fraction=0.06,
            context_side_min=96.0,
            seed=args.seed,
        )
        for coarse, target_side in class_target_sides.items()
    }

    candidates: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for image_path in train_paths:
        category_ids, boxes, width, height = annotations(image_path)
        eligible = []
        scale = args.network_size / max(width, height)
        for index, (category_id, box) in enumerate(zip(category_ids, boxes, strict=True)):
            network_side = max(box.width, box.height) * scale
            coarse = "vehicle" if category_id == 24 else "ship"
            if category_id in target_classes and network_side < class_target_sides[coarse]:
                relative_side = network_side / class_target_sides[coarse]
                eligible.append((relative_side, network_side, index, category_id, coarse))
        if not eligible:
            continue
        # The smallest target is the most under-resolved. One crop per source
        # image prevents crowded scenes from dominating the supplement.
        _, network_side, target_index, category_id, coarse = min(eligible)
        result = build_scale_crop(
            boxes[target_index],
            image_width=width,
            image_height=height,
            all_boxes=boxes,
            policy=policies[coarse],
            stable_key=f"{image_path.name}|{target_index}|{category_id}",
        )
        if result is None:
            counts[f"rejected_partial_or_visibility:{category_id}"] += 1
            continue
        crop, kept_indices = result
        candidates.append(
            {
                "image_path": image_path,
                "category_ids": category_ids,
                "boxes": boxes,
                "crop": crop,
                "kept_indices": kept_indices,
                "target_index": target_index,
                "target_category_id": category_id,
                "source_network_side": network_side,
                "target_network_side": class_target_sides[coarse],
                "rank": stable_rank(args.seed, f"{category_id}|{image_path.name}|{target_index}"),
            }
        )
        counts[f"eligible:{category_id}"] += 1

    cap = min(args.max_extra_images, int(len(train_paths) * args.max_extra_fraction))
    # Deterministic class round-robin retains scarce Vehicle/class-safe Ship
    # scenes instead of allowing frequent Ship classes to consume the cap.
    by_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_class[int(row["target_category_id"])].append(row)
    for rows in by_class.values():
        rows.sort(key=lambda row: str(row["rank"]))
    selected: list[dict[str, object]] = []
    class_order = sorted(by_class)
    while len(selected) < cap and class_order:
        next_order = []
        for category_id in class_order:
            if len(selected) >= cap:
                break
            rows = by_class[category_id]
            if rows:
                selected.append(rows.pop())
            if rows:
                next_order.append(category_id)
        class_order = next_order

    manifest = []
    for sequence, row in enumerate(selected):
        image_path = Path(row["image_path"])
        crop = row["crop"]
        assert isinstance(crop, Box)
        name = f"scale_{sequence:04d}_{image_path.stem}.jpg"
        manifest.append(
            {
                "name": name,
                "source_image": str(image_path),
                "source_image_sha256": sha256(image_path),
                "target_index": int(row["target_index"]),
                "target_category_id": int(row["target_category_id"]),
                "source_network_side": float(row["source_network_side"]),
                "target_network_side": int(row["target_network_side"]),
                "crop_xyxy": crop.as_list(),
                "retained_annotation_indices": list(row["kept_indices"]),
            }
        )

    summary = {
        "status": "dry_run" if args.dry_run else "complete",
        "schema_version": "object_scale_detector_scenes_v1",
        "train_source_count": len(train_paths),
        "validation_source_count": len(val_paths),
        "train_validation_overlap": len(overlap),
        "training_mode": "full_no_validation" if args.full_training else "held_out_validation",
        "validation_used_for_training_or_selection": False,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "cap": cap,
        "selected_by_class": dict(sorted(Counter(int(r["target_category_id"]) for r in selected).items())),
        "counts": dict(sorted(counts.items())),
        "policies": {coarse: asdict(policy) for coarse, policy in policies.items()},
        "target_classes": sorted(target_classes),
        "selection": "one_smallest_target_per_source_then_deterministic_class_round_robin",
        "partial_annotation_policy": "reject_0.05_to_0.70_visibility_keep_ge_0.70",
        "isolated_object_stretching": False,
        "validation_unchanged": True,
        "train_list_sha256": sha256(args.train_list),
        "val_list_sha256": sha256(args.val_list),
    }
    if args.dry_run:
        args.output.mkdir(parents=True)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 0

    image_dir = args.output / "dataset" / "images" / "train"
    label_dir = args.output / "dataset" / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    generated_paths: list[Path] = []
    for row, record in zip(selected, manifest, strict=True):
        image_path = Path(row["image_path"])
        crop = row["crop"]
        assert isinstance(crop, Box)
        with Image.open(image_path) as source:
            rendered = reflect_crop(source, crop, args.output_pixels)
        output_image = image_dir / str(record["name"])
        rendered.save(output_image, quality=95, subsampling=0)
        output_label = label_dir / output_image.with_suffix(".txt").name
        output_label.write_text(
            "\n".join(
                yolo_rows_for_crop(
                    row["boxes"],
                    row["category_ids"],
                    row["kept_indices"],
                    crop,
                    args.output_pixels,
                )
            )
            + "\n"
        )
        generated_paths.append(output_image.resolve())

    train_output = args.output / "train.txt"
    val_output = args.output / "val.txt"
    train_output.write_text("\n".join(map(str, [*train_paths, *generated_paths])) + "\n")
    val_output.write_text("\n".join(map(str, val_paths)) + "\n")
    dataset = args.output / "dataset.yaml"
    names = ", ".join(repr(str(index)) for index in range(25))
    dataset.write_text(
        f"path: /\ntrain: {train_output.resolve()}\nval: {val_output.resolve()}\n"
        f"nc: 25\nnames: [{names}]\n"
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary.update(
        {
            "dataset_yaml": str(dataset.resolve()),
            "dataset_yaml_sha256": sha256(dataset),
            "output_train_count": len(train_paths) + len(generated_paths),
            "output_val_count": len(val_paths),
        }
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
