#!/usr/bin/env python3
"""Freeze a source/group-disjoint Sentinel-B before model evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--forbidden-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-images", type=int, default=600)
    parser.add_argument("--seed", default="sentinel-b-20260901")
    args = parser.parse_args()
    with args.registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with args.forbidden_registry.open(encoding="utf-8", newline="") as handle:
        forbidden = list(csv.DictReader(handle))
    required = {"image_id", "group_id", "source_id"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"registry must contain {sorted(required)}")
    forbidden_sources = {row["source_id"] for row in forbidden}
    forbidden_groups = {row["group_id"] for row in forbidden}
    overlap_sources = {row["source_id"] for row in rows} & forbidden_sources
    overlap_groups = {row["group_id"] for row in rows} & forbidden_groups
    if overlap_sources or overlap_groups:
        raise ValueError(
            f"Sentinel-B leakage: sources={sorted(overlap_sources)[:5]}, "
            f"groups={sorted(overlap_groups)[:5]}"
        )
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    group_order = sorted(
        by_group,
        key=lambda group: hashlib.sha256(f"{args.seed}|{group}".encode()).hexdigest(),
    )
    selected: list[dict[str, str]] = []
    selected_groups: list[str] = []
    for group in group_order:
        if len(selected) >= args.target_images:
            break
        selected_groups.append(group)
        selected.extend(sorted(by_group[group], key=lambda row: int(row["image_id"])))
    if len(selected) < args.target_images:
        raise RuntimeError("not enough disjoint images to freeze Sentinel-B")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "sentinel_b_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    seal = {
        "version": "sentinel_b_v1",
        "metric_protocol": "platform_observed_20260831",
        "selection_precedes_evaluation": True,
        "source_disjoint": True,
        "group_disjoint": True,
        "image_count": len(selected),
        "group_count": len(selected_groups),
        "manifest_sha256": digest,
        "seed": args.seed,
        "predictions_must_not_be_read_by_this_script": True,
        "formal_admission": True,
    }
    (args.output / "sentinel_b_seal.json").write_text(
        json.dumps(seal, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "SHA256SUMS.txt").write_text(
        f"{digest}  sentinel_b_manifest.csv\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
