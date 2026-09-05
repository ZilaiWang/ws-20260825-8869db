#!/usr/bin/env python3
"""Apply frozen H3 strict Ship cross-fine dedup to COCO predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.postprocess.strict_ship_cross_fine import suppress_strict_ship_cross_fine


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    by_image = defaultdict(list)
    for row in rows:
        by_image[int(row["image_id"])].append(row)
    output = []
    audits = {}
    for image_id in sorted(by_image):
        kept, audit = suppress_strict_ship_cross_fine(by_image[image_id])
        output.extend(kept)
        if audit["suppressed_count"]:
            audits[str(image_id)] = audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8")
    payload = {
        "status": "complete",
        "role": "h3_strict_ship_cross_fine_dedup",
        "frozen_parameters": {
            "ship_category_ids": [0, 1, 2, 3],
            "iou_threshold": 0.75,
            "ios_threshold": 0.90,
            "normalized_center_distance_threshold": 0.20,
            "selection": "highest_score_real_proposal",
        },
        "input_sha256": _sha256(args.input),
        "output_sha256": _sha256(args.output),
        "input_count": len(rows),
        "output_count": len(output),
        "suppressed_count": len(rows) - len(output),
        "changed_images": audits,
    }
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
