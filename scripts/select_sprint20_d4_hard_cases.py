#!/usr/bin/env python3
"""Select real images where the existing full D4 policy changed a class."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.max_images < 1 or not 0 <= args.threshold <= 1:
        raise ValueError("Invalid selection arguments")

    cache = _read(args.cache)
    rows = _read(args.probabilities)
    changed: Counter[int] = Counter()
    for row in rows:
        probabilities = list(map(float, row["probabilities"]))
        if len(probabilities) != 20:
            raise ValueError("Expected 20 aircraft probabilities")
        best = max(range(20), key=probabilities.__getitem__)
        old = int(row["old_category"])
        new = best + 4 if probabilities[best] >= args.threshold else old
        if new != old:
            changed[int(row["image_id"])] += 1
    available = {int(row["image_id"]): row for row in cache["images"]}
    selected = [
        image_id
        for image_id, _ in sorted(changed.items(), key=lambda item: (-item[1], item[0]))
        if image_id in available
    ][: args.max_images]
    if not selected:
        raise ValueError("No previously changed D4 images overlap the cache")
    output = {
        **{key: value for key, value in cache.items() if key != "images"},
        "role": "engineering_only",
        "selection": {
            "method": "existing_full_d4_changed_class_count_descending",
            "probabilities_sha256": _sha256(args.probabilities),
            "threshold": args.threshold,
            "selected_images": selected,
            "historical_changed_objects": {
                str(image_id): changed[image_id] for image_id in selected
            },
            "accuracy_admission": False,
        },
        "images": [available[image_id] for image_id in selected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["selection"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
