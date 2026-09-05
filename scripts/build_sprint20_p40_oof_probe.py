#!/usr/bin/env python3
"""Materialize immutable inputs for the existing P40 short-CV3 head probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--cv3-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    coco = _read(args.coco)
    config = _read(args.base_config)
    image_fold = {int(row["id"]): int(row["fold"]) for row in coco["images"]}
    if set(image_fold.values()) != {0, 1, 2}:
        raise ValueError("Expected aggregate COCO images with folds 0,1,2")
    if len(image_fold) != len(coco["images"]):
        raise ValueError("Duplicate image IDs")

    manifest = {
        "schema": "sprint20_p40_short_cv3_probe_inputs_v1",
        "evidence_role": "outer_oof_short",
        "warning": (
            "Existing P40 CV3 uses S1024/40e -> P40/40e; it is source-disjoint OOF but "
            "does not match the full model's S1024/160e -> P40/40e maturity."
        ),
        "aggregate_coco_sha256": _sha256(args.coco),
        "base_config_sha256": _sha256(args.base_config),
        "folds": [],
    }
    for fold in (0, 1, 2):
        ids = {image_id for image_id, value in image_fold.items() if value == fold}
        fold_coco = {
            **{key: value for key, value in coco.items() if key not in {"images", "annotations"}},
            "images": [row for row in coco["images"] if int(row["id"]) in ids],
            "annotations": [row for row in coco["annotations"] if int(row["image_id"]) in ids],
        }
        weight = (
            args.cv3_root / f"fold_{fold}/adaptation/runs/resolution_adaptation/weights/last.pt"
        ).resolve()
        if not weight.is_file():
            raise FileNotFoundError(weight)
        fold_config = copy.deepcopy(config)
        fold_config["model"]["weight_path"] = str(weight)
        fold_config["model"]["expected_sha256"] = _sha256(weight)
        fold_config["deployment_role"] = "EXPERIMENTAL_SHORT_OOF_HEAD_PROBE"
        coco_path = args.output_dir / f"fold_{fold}/ground_truth.json"
        config_path = args.output_dir / f"fold_{fold}/config.json"
        _write(coco_path, fold_coco)
        _write(config_path, fold_config)
        manifest["folds"].append(
            {
                "fold": fold,
                "images": len(fold_coco["images"]),
                "annotations": len(fold_coco["annotations"]),
                "weight_path": str(weight),
                "weight_sha256": _sha256(weight),
                "coco": str(coco_path),
                "coco_sha256": _sha256(coco_path),
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
            }
        )
    _write(args.output_dir / "input_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
