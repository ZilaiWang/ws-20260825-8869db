#!/usr/bin/env python3
"""Materialize formal CV3 images in the pseudo-evaluation directory contract.

The output uses symlinks, keeps the original image bytes untouched, and rewrites
only COCO ``file_name`` values to unique basenames expected by the Docker-style
runner.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.gt.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    basenames = [Path(str(item["file_name"])).name for item in images]
    if len(basenames) != len(set(basenames)):
        raise ValueError("formal CV3 file basenames are not unique")
    folds = {int(item["fold"]) for item in images}
    if folds != {0, 1, 2}:
        raise ValueError(f"expected folds 0,1,2, got {sorted(folds)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    counts = {0: 0, 1: 0, 2: 0}
    rewritten = json.loads(json.dumps(payload))
    for source_item, target_item in zip(images, rewritten["images"], strict=True):
        fold = int(source_item["fold"])
        relative = Path(str(source_item["file_name"]))
        source = (args.data_root / relative).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        image_dir = args.output_root / f"fold_{fold}" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        target = image_dir / relative.name
        if target.is_symlink():
            if target.resolve() != source:
                raise ValueError(f"existing symlink points elsewhere: {target}")
        elif target.exists():
            raise FileExistsError(target)
        else:
            os.symlink(source, target)
        target_item["file_name"] = relative.name
        counts[fold] += 1

    (args.output_root / "ground_truth.json").write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "status": "complete",
        "protocol": "formal_cv3_symlink_pseudo_root_v1",
        "images": len(images),
        "fold_counts": {str(key): value for key, value in counts.items()},
        "source_gt": str(args.gt.resolve()),
        "data_root": str(args.data_root.resolve()),
    }
    (args.output_root / "materialization_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
