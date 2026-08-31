#!/usr/bin/env python3
"""Create audited scale-preserving tiles from an external coarse COCO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rsdet.external.slicing import slice_coco


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-coco", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-coco", type=Path, required=True)
    parser.add_argument("--output-image-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--min-visibility", type=float, default=0.7)
    parser.add_argument("--empty-tiles-per-image", type=int, default=2)
    args = parser.parse_args()
    payload = json.loads(args.input_coco.read_text(encoding="utf-8"))
    output, audit = slice_coco(
        payload,
        args.image_root,
        args.output_image_root,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_visibility=args.min_visibility,
        empty_tiles_per_image=args.empty_tiles_per_image,
    )
    args.output_coco.parent.mkdir(parents=True, exist_ok=True)
    args.output_coco.write_text(
        json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    audit["input_coco"] = str(args.input_coco.resolve())
    audit["input_coco_sha256"] = _sha256(args.input_coco)
    audit["output_coco"] = str(args.output_coco.resolve())
    audit["output_coco_sha256"] = _sha256(args.output_coco)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
