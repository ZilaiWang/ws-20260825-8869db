#!/usr/bin/env python3
"""Build a deterministic, class-covering YOLO list for end-to-end smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    indices = [index for index, value in enumerate(parts) if value == "images"]
    if not indices:
        raise ValueError(f"image path lacks an images directory component: {image_path}")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def build_smoke_list(lines: list[str], maximum_images: int) -> tuple[list[str], dict]:
    if maximum_images <= 0:
        raise ValueError("maximum_images must be positive")
    paths = [Path(line.strip()).resolve() for line in lines if line.strip()]
    if not paths:
        raise ValueError("source train list is empty")
    by_class: dict[int, list[int]] = defaultdict(list)
    labels_by_index: dict[int, set[int]] = {}
    for index, image_path in enumerate(paths):
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        label_path = _label_path(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        labels = {
            int(line.split()[0])
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        labels_by_index[index] = labels
        for label in labels:
            by_class[label].append(index)
    selected = {indices[0] for _, indices in sorted(by_class.items()) if indices}
    for index in range(len(paths)):
        if len(selected) >= maximum_images:
            break
        selected.add(index)
    selected_indices = sorted(selected)[:maximum_images]
    selected_lines = [str(paths[index]) for index in selected_indices]
    covered = sorted({label for index in selected_indices for label in labels_by_index[index]})
    return selected_lines, {
        "status": "complete",
        "protocol": "deterministic_class_covering_yolo_smoke_list_v1",
        "source_image_count": len(paths),
        "selected_image_count": len(selected_lines),
        "covered_class_ids": covered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-images", type=int, default=32)
    args = parser.parse_args()
    payload = yaml.safe_load(args.dataset.read_text(encoding="utf-8"))
    train_path = Path(str(payload["train"]))
    if not train_path.is_absolute():
        train_path = (args.dataset.parent / train_path).resolve()
    lines, audit = build_smoke_list(
        train_path.read_text(encoding="utf-8").splitlines(), args.maximum_images
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    list_path = args.output_dir / "train.txt"
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_yaml = args.output_dir / "dataset.yaml"
    output_payload = dict(payload)
    output_payload["train"] = str(list_path.resolve())
    output_payload["val"] = str(list_path.resolve())
    output_yaml.write_text(yaml.safe_dump(output_payload, sort_keys=False), encoding="utf-8")
    audit.update(
        {
            "source_dataset_sha256": _sha256(args.dataset),
            "source_train_list_sha256": _sha256(train_path),
            "output_train_list_sha256": _sha256(list_path),
            "output_dataset_sha256": _sha256(output_yaml),
        }
    )
    audit_path = args.output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
