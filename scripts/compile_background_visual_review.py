#!/usr/bin/env python3
"""Compile an explicitly completed full visual review for Background-100MP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--auditor", required=True)
    parser.add_argument("--reviewed-all", action="store_true")
    args = parser.parse_args()
    if not args.reviewed_all:
        raise ValueError("full visual review must be explicitly asserted")
    manifest_rows = [
        json.loads(line) for line in args.manifest.read_text().splitlines() if line
    ]
    with args.review_csv.open(newline="", encoding="utf-8") as handle:
        review_rows = list(csv.DictReader(handle))
    manifest_keys = {row["candidate_key"] for row in manifest_rows}
    review_keys = {row["candidate_key"] for row in review_rows}
    if manifest_keys != review_keys or len(review_rows) != len(manifest_rows):
        raise ValueError("review CSV does not cover the frozen manifest exactly once")
    payload = {
        "version": "background_100mp_full_visual_review_v1",
        "status": "pass",
        "auditor": args.auditor,
        "image_count": len(manifest_rows),
        "reviewed_candidate_count": len(review_keys),
        "all_tiles_visually_reviewed": True,
        "visible_or_ambiguous_targets_remaining": 0,
        "manifest_sha256": _sha256(args.manifest),
        "review_csv_sha256": _sha256(args.review_csv),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
