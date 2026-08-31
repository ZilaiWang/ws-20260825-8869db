#!/usr/bin/env python3
"""Deterministically discover the extracted official DIOR trainval layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stems(directory: Path, suffixes: set[str]) -> set[str]:
    return {
        path.stem
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }


def _split_ids(path: Path) -> list[str]:
    rows = [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != len(set(rows)):
        raise ValueError(f"duplicate IDs in DIOR split: {path}")
    return rows


def discover_dior_layout(root: Path) -> dict:
    root = root.resolve()
    image_dirs = []
    annotation_dirs = []
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        image_count = sum(
            1
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        xml_count = sum(
            1
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".xml"
        )
        if image_count:
            image_dirs.append((directory, image_count))
        if xml_count:
            annotation_dirs.append((directory, xml_count))
    split_files = sorted(
        path
        for path in root.rglob("*.txt")
        if path.name.lower() in {"trainval.txt", "train.txt"}
    )
    if not image_dirs or not annotation_dirs or not split_files:
        raise FileNotFoundError(
            "extracted DIOR layout needs an image directory, XML directory, and trainval/train split"
        )

    split_priority = {"trainval.txt": 2, "train.txt": 1}
    candidates = []
    for image_dir, image_count in image_dirs:
        image_stems = _stems(image_dir, IMAGE_SUFFIXES)
        for annotation_dir, xml_count in annotation_dirs:
            xml_stems = _stems(annotation_dir, {".xml"})
            for split_file in split_files:
                selected = _split_ids(split_file)
                selected_set = set(selected)
                covered = len(selected_set & image_stems & xml_stems)
                missing = len(selected_set) - covered
                candidates.append(
                    (
                        missing == 0,
                        split_priority[split_file.name.lower()],
                        covered,
                        min(image_count, xml_count),
                        str(image_dir),
                        str(annotation_dir),
                        str(split_file),
                        image_dir,
                        annotation_dir,
                        split_file,
                        image_count,
                        xml_count,
                        selected,
                        missing,
                    )
                )
    candidates.sort(reverse=True)
    best = candidates[0]
    if not best[0] or best[2] == 0:
        raise ValueError(
            f"no complete DIOR split/layout match; best covered={best[2]} missing={best[13]}"
        )
    image_dir, annotation_dir, split_file = best[7], best[8], best[9]
    selected = best[12]
    result = {
        "status": "complete",
        "protocol": "official_dior_extracted_layout_discovery_v1",
        "root": str(root),
        "image_root": str(image_dir.resolve()),
        "annotation_root": str(annotation_dir.resolve()),
        "split_file": str(split_file.resolve()),
        "split_file_sha256": _sha256(split_file),
        "split_name": split_file.name,
        "selected_image_count": len(selected),
        "image_file_count_in_directory": best[10],
        "xml_file_count_in_directory": best[11],
        "candidate_layout_count": len(candidates),
        "selection_policy": (
            "complete split coverage, prefer trainval over train, then maximum coverage; "
            "paths are deterministic tie breakers"
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = discover_dior_layout(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
