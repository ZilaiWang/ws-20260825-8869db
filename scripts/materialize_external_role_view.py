#!/usr/bin/env python3
"""Materialize a cache-isolated YOLO view for one external-data role.

The image payload remains shared through a directory symlink.  Label text files are
copied so every role owns independent label inodes and an independent Ultralytics
``*.cache`` file.  This prevents a later edit in one role from mutating another role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_role_yaml(path: Path) -> tuple[Path, Path, dict]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("role dataset YAML must be a mapping")
    dataset_root = Path(str(payload.get("path", path.parent))).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (path.parent / dataset_root).resolve()
    train_value = Path(str(payload["train"]))
    if not train_value.is_absolute():
        train_value = (dataset_root / train_value).resolve()
    return dataset_root, train_value, payload


def materialize_role_view(role_yaml: Path, output_dir: Path) -> dict:
    source_root, source_list, payload = _load_role_yaml(role_yaml)
    source_images = (source_root / "images" / "train").resolve()
    source_labels = (source_root / "labels" / "train").resolve()
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError("source dataset must contain images/train and labels/train")
    raw_lines = [
        line.strip()
        for line in source_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not raw_lines:
        raise ValueError("role train list is empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_parent = output_dir / "images"
    label_root = output_dir / "labels" / "train"
    image_parent.mkdir(exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    image_link = image_parent / "train"
    if image_link.exists() or image_link.is_symlink():
        if not image_link.is_symlink() or image_link.resolve() != source_images:
            raise FileExistsError(f"incompatible existing image view: {image_link}")
    else:
        image_link.symlink_to(source_images, target_is_directory=True)

    copied = existing = 0
    for source_label in sorted(source_labels.rglob("*.txt")):
        target = label_root / source_label.relative_to(source_labels)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if _sha256(target) != _sha256(source_label):
                raise ValueError(f"existing role label differs from source: {target}")
            existing += 1
            continue
        shutil.copy2(source_label, target)
        copied += 1

    rewritten: list[str] = []
    missing = []
    for raw in raw_lines:
        source_image = Path(raw).expanduser().resolve()
        try:
            relative = source_image.relative_to(source_images)
        except ValueError as exc:
            raise ValueError(f"role image is outside source images/train: {raw}") from exc
        if not source_image.is_file():
            missing.append(raw)
        rewritten.append(str((image_link / relative).absolute()))
    if missing:
        raise FileNotFoundError(f"{len(missing)} role images are missing; first={missing[0]}")

    train_list = output_dir / "train-role.txt"
    train_list.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    output_yaml = output_dir / "dataset.yaml"
    isolated = dict(payload)
    isolated["path"] = str(output_dir.resolve())
    isolated["train"] = str(train_list.resolve())
    isolated["val"] = str(train_list.resolve())
    output_yaml.write_text(yaml.safe_dump(isolated, sort_keys=False), encoding="utf-8")

    audit = {
        "status": "complete",
        "protocol": "external_role_cache_isolation_v1",
        "source_role_yaml": str(role_yaml.resolve()),
        "source_role_yaml_sha256": _sha256(role_yaml),
        "source_train_list_sha256": _sha256(source_list),
        "output_dataset_yaml": str(output_yaml.resolve()),
        "output_dataset_yaml_sha256": _sha256(output_yaml),
        "output_train_list_sha256": _sha256(train_list),
        "sampled_row_count": len(rewritten),
        "unique_image_count": len(set(rewritten)),
        "label_file_count": len(list(label_root.rglob("*.txt"))),
        "label_materialization": {
            "hard_linked": 0,
            "copied": copied,
            "already_verified": existing,
        },
        "cache_contract": (
            "each role owns independent label inodes and labels/train.cache; "
            "image bytes are shared read-only"
        ),
    }
    audit_path = output_dir / "role_view_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = materialize_role_view(args.role_yaml, args.output_dir)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
