#!/usr/bin/env python3
"""Freeze Background-100MP only after geometry and full visual gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    review = json.loads(args.visual_review.read_text())
    exclusions = json.loads(args.exclusions.read_text())
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    keys = [row["candidate_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate_key is not unique")
    if set(keys) & set(exclusions["candidate_keys"]):
        raise ValueError("a visually excluded tile remains in the frozen manifest")
    manifest_sha = _sha256(args.manifest)
    gates = {
        "megapixels_at_least_100": float(summary["megapixels"]) >= 100.0,
        "geometry": bool(summary["automatic_geometry_admission"]),
        "visual": review.get("status") == "pass",
        "full_visual_coverage": int(review["reviewed_candidate_count"]) == len(rows),
        "manifest_integrity": review["manifest_sha256"] == manifest_sha,
        "excluded_targets_absent": not bool(set(keys) & set(exclusions["candidate_keys"])),
    }
    admitted = all(gates.values())
    payload = {
        "version": "background_100mp_frozen_v1",
        "status": "frozen" if admitted else "not_admitted",
        "formal_admission": admitted,
        "metric_protocol": "platform_observed_20260831",
        "image_count": len(rows),
        "megapixels": summary["megapixels"],
        "manifest_sha256": manifest_sha,
        "visual_review_sha256": _sha256(args.visual_review),
        "exclusions_sha256": _sha256(args.exclusions),
        "gates": gates,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not admitted:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
