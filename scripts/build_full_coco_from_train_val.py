#!/usr/bin/env python3
"""Merge disjoint COCO train/validation ledgers into one full-fit ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"COCO ledger must be an object: {path}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"COCO ledger misses list key={key}: {path}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=4481)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    train = _load(args.train)
    val = _load(args.val)
    if train["categories"] != val["categories"]:
        raise ValueError("train and validation category ledgers differ")
    train_images = {int(item["id"]) for item in train["images"]}
    val_images = {int(item["id"]) for item in val["images"]}
    if len(train_images) != len(train["images"]) or len(val_images) != len(val["images"]):
        raise ValueError("duplicate image IDs inside input COCO ledger")
    if train_images & val_images:
        raise ValueError("train and validation image IDs overlap")
    all_images = train_images | val_images
    if len(all_images) != args.expected_images:
        raise ValueError(
            f"unexpected full image count={len(all_images)} expected={args.expected_images}"
        )
    train_annotations = {int(item["id"]) for item in train["annotations"]}
    val_annotations = {int(item["id"]) for item in val["annotations"]}
    if len(train_annotations) != len(train["annotations"]) or len(val_annotations) != len(
        val["annotations"]
    ):
        raise ValueError("duplicate annotation IDs inside input COCO ledger")
    if train_annotations & val_annotations:
        raise ValueError("train and validation annotation IDs overlap")
    annotations = [*train["annotations"], *val["annotations"]]
    referenced = {int(item["image_id"]) for item in annotations}
    if not referenced <= all_images:
        raise ValueError("annotation references unknown image IDs")

    merged: dict[str, Any] = {
        "images": sorted([*train["images"], *val["images"]], key=lambda item: int(item["id"])),
        "annotations": sorted(annotations, key=lambda item: int(item["id"])),
        "categories": train["categories"],
    }
    for key in ("info", "licenses"):
        if key in train:
            merged[key] = train[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False) + "\n", encoding="utf-8")
    audit = {
        "status": "pass",
        "protocol": "disjoint_coco_train_val_full_fit_merge_v1",
        "train_sha256": _sha256(args.train),
        "val_sha256": _sha256(args.val),
        "output_sha256": _sha256(args.output),
        "train_images": len(train_images),
        "val_images": len(val_images),
        "full_images": len(all_images),
        "annotations": len(annotations),
        "categories": len(train["categories"]),
        "cross_split_image_overlap": 0,
        "cross_split_annotation_overlap": 0,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
