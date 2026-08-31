#!/usr/bin/env python3
"""Materialize a paired YOLO dataset from reviewed missing-label decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from rsdet.data.xh_dataset import FINE_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_label(image: Path) -> Path:
    parts = list(image.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def materialize(
    manifest: Path,
    data_root: Path,
    confirmed_path: Path,
    ignored_path: Path,
    output_dir: Path,
    *,
    held_out_fold: int | None,
    add_confirmed: bool,
    ambiguous_policy: str,
) -> dict:
    if ambiguous_policy not in {"exclude_image", "keep_original"}:
        raise ValueError(f"unsupported ambiguous policy: {ambiguous_policy}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest has no samples")
    confirmed = json.loads(confirmed_path.read_text(encoding="utf-8"))
    ignored = json.loads(ignored_path.read_text(encoding="utf-8"))
    confirmed_by_file: dict[str, list[dict]] = defaultdict(list)
    for row in confirmed:
        confirmed_by_file[str(row["file_name"])].append(row)
    ignored_files = {str(row["file_name"]) for row in ignored}

    image_dir = output_dir / "images" / "train"
    label_dir = output_dir / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    train_images: list[Path] = []
    added_by_fold: Counter[int] = Counter()
    excluded_files: list[str] = []
    skipped_confirmed = 0
    selected_files: set[str] = set()

    for sample in sorted(samples, key=lambda row: int(row["image_id"])):
        fold = int(sample["fold"])
        if held_out_fold is not None and fold == held_out_fold:
            continue
        relative = str(sample["relative_path"])
        if relative in ignored_files and ambiguous_policy == "exclude_image":
            excluded_files.append(relative)
            skipped_confirmed += len(confirmed_by_file.get(relative, []))
            continue
        source_image = (data_root / relative).resolve()
        source_label = _source_label(source_image)
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(source_image if not source_image.is_file() else source_label)
        selected_files.add(relative)
        stem = f"image_{int(sample['image_id']):05d}"
        target_image = image_dir / f"{stem}{source_image.suffix.lower()}"
        if not target_image.exists() and not target_image.is_symlink():
            os.symlink(source_image, target_image)
        elif target_image.resolve() != source_image:
            raise FileExistsError(target_image)
        lines = source_label.read_text(encoding="utf-8").splitlines()
        if add_confirmed:
            with Image.open(source_image) as image:
                width, height = image.size
            for row in confirmed_by_file.get(relative, []):
                x1, y1, x2, y2 = (float(value) for value in row["bbox_xyxy"])
                x1 = max(0.0, min(float(width), x1))
                x2 = max(0.0, min(float(width), x2))
                y1 = max(0.0, min(float(height), y1))
                y2 = max(0.0, min(float(height), y2))
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"invalid reviewed bbox: {row['candidate_id']}")
                lines.append(
                    "24 "
                    f"{(x1 + x2) / 2.0 / width:.8f} "
                    f"{(y1 + y2) / 2.0 / height:.8f} "
                    f"{(x2 - x1) / width:.8f} "
                    f"{(y2 - y1) / height:.8f}"
                )
                added_by_fold[fold] += 1
        (label_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        train_images.append(target_image.absolute())

    unresolved_confirmed = sorted(set(confirmed_by_file) - selected_files - set(excluded_files))
    if held_out_fold is None and unresolved_confirmed:
        raise ValueError(f"reviewed file absent from manifest: {unresolved_confirmed}")
    list_path = output_dir / "train_images.txt"
    list_path.write_text("\n".join(map(str, train_images)) + "\n", encoding="utf-8")
    dataset = output_dir / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "path": str(output_dir.resolve()),
                "train": str(list_path.resolve()),
                "val": str(list_path.resolve()),
                "nc": len(FINE_NAMES),
                "names": list(FINE_NAMES),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    audit = {
        "status": "partial_label_safe_dataset_ready",
        "protocol": "paired_reviewed_missing_label_dataset_v1",
        "held_out_fold": held_out_fold,
        "add_confirmed": add_confirmed,
        "ambiguous_policy": ambiguous_policy,
        "image_count": len(train_images),
        "excluded_ambiguous_image_count": len(set(excluded_files)),
        "excluded_ambiguous_files": sorted(set(excluded_files)),
        "confirmed_box_count_in_source": len(confirmed),
        "confirmed_boxes_added": sum(added_by_fold.values()),
        "confirmed_boxes_skipped_with_excluded_images": skipped_confirmed,
        "confirmed_boxes_added_by_fold": dict(sorted(added_by_fold.items())),
        "manifest_sha256": _sha256(manifest),
        "confirmed_sha256": _sha256(confirmed_path),
        "ignored_sha256": _sha256(ignored_path),
        "train_list_sha256": _sha256(list_path),
        "dataset_yaml_sha256": _sha256(dataset),
        "pairing_contract": (
            "control and patch arms must use the same held_out_fold and ambiguous_policy"
        ),
    }
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--confirmed", type=Path, required=True)
    parser.add_argument("--ignored", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-out-fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--confirmed-policy", choices=("add", "omit"), default="add")
    parser.add_argument(
        "--ambiguous-policy",
        choices=("exclude_image", "keep_original"),
        default="exclude_image",
    )
    args = parser.parse_args()
    audit = materialize(
        args.manifest,
        args.data_root,
        args.confirmed,
        args.ignored,
        args.output_dir,
        held_out_fold=args.held_out_fold,
        add_confirmed=args.confirmed_policy == "add",
        ambiguous_policy=args.ambiguous_policy,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
