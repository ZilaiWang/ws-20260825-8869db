#!/usr/bin/env python3
"""Import DOTA labelTxt OBB annotations as an audited four-class COCO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rsdet.external.dota import import_dota


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--keep-difficult", action="store_true")
    parser.add_argument(
        "--difficult-policy",
        choices=("drop", "keep_primary", "keep_all_mapped"),
        default="drop",
        help=(
            "drop all difficult objects, keep only difficult compact primary objects, "
            "or keep every difficult object that maps to a foreground coarse class"
        ),
    )
    parser.add_argument(
        "--allow-label-superset",
        action="store_true",
        help="Allow a full label directory with only a downloaded image subset.",
    )
    args = parser.parse_args()
    payload, audit = import_dota(
        args.image_root,
        args.label_root,
        keep_difficult=args.keep_difficult,
        difficult_policy=args.difficult_policy,
        require_exact_stem_set=not args.allow_label_superset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    audit["output_coco"] = str(args.output.resolve())
    audit["output_coco_sha256"] = _sha256(args.output)
    audit["source_license"] = "DOTA-v1.0 official academic-use license; verify intended use"
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
