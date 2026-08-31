#!/usr/bin/env python3
"""Build frozen role-specific repeated image lists for external pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROLE_MULTIPLIERS = {
    "EXT-G": {0: 2, 1: 1, 2: 1, 3: 4},
    "EXT-V": {0: 1, 1: 1, 2: 2, 3: 4},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_role_list(coco: dict, image_root: Path, role: str) -> tuple[list[str], dict]:
    if role not in ROLE_MULTIPLIERS:
        raise ValueError(f"unsupported external role {role}")
    by_image: dict[int, list[int]] = defaultdict(list)
    for row in coco["annotations"]:
        by_image[int(row["image_id"])].append(int(row["category_id"]))
    repeats: Counter[int] = Counter()
    lines: list[str] = []
    for image in sorted(coco["images"], key=lambda row: int(row["id"])):
        image_path = (image_root / image["file_name"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        categories = set(by_image[int(image["id"])])
        repeat = max(
            (ROLE_MULTIPLIERS[role][category] for category in categories),
            default=1,
        )
        repeats[repeat] += 1
        lines.extend([str(image_path)] * repeat)
    audit = {
        "status": "complete",
        "protocol": "frozen_external_role_repetition_sampler_v1",
        "role": role,
        "category_multipliers": ROLE_MULTIPLIERS[role],
        "unique_image_count": len(coco["images"]),
        "sampled_row_count": len(lines),
        "repeat_histogram": dict(sorted(repeats.items())),
        "contract": "only the frozen role differs between EXT-G and EXT-V",
    }
    return lines, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--role", choices=tuple(ROLE_MULTIPLIERS), required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    lines, audit = build_role_list(coco, args.dataset_root / "images/train", args.role)
    list_path = args.dataset_root / f"train-{args.role.lower()}.txt"
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    base_yaml = yaml.safe_load((args.dataset_root / "dataset.yaml").read_text(encoding="utf-8"))
    base_yaml["train"] = str(list_path.resolve())
    base_yaml["val"] = str(list_path.resolve())
    role_yaml = args.dataset_root / f"dataset-{args.role.lower()}.yaml"
    role_yaml.write_text(yaml.safe_dump(base_yaml, sort_keys=False), encoding="utf-8")
    audit.update(
        {
            "coco_sha256": _sha256(args.coco),
            "train_list": str(list_path.resolve()),
            "train_list_sha256": _sha256(list_path),
            "dataset_yaml": str(role_yaml.resolve()),
            "dataset_yaml_sha256": _sha256(role_yaml),
        }
    )
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
