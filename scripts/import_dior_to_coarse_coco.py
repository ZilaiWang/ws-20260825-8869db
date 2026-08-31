#!/usr/bin/env python3
"""Import official DIOR VOC XML into the frozen four-coarse COCO contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rsdet.external.dior import import_dior


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--keep-difficult", action="store_true")
    args = parser.parse_args()
    payload, audit = import_dior(
        args.image_root,
        args.annotation_root,
        split_file=args.split_file,
        keep_difficult=args.keep_difficult,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    audit.update(
        {
            "output_coco": str(args.output.resolve()),
            "output_coco_sha256": _sha256(args.output),
            "source_license": "DIOR official CC BY-NC 4.0",
        }
    )
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
