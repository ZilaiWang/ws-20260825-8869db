#!/usr/bin/env python3
"""Merge per-fold PAV logits into one deterministic, fully audited OOF file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    if len(args.inputs) != 3:
        raise ValueError("exactly three formal fold inputs are required")
    arrays = [np.load(path, allow_pickle=False) for path in args.inputs]
    keys = tuple(arrays[0].files)
    if "candidate_id" not in keys or any(tuple(item.files) != keys for item in arrays):
        raise ValueError("PAV fold arrays have inconsistent keys")
    merged = {key: np.concatenate([item[key] for item in arrays], axis=0) for key in keys}
    candidate_ids = merged["candidate_id"].astype(np.int64)
    if len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError("PAV fold arrays overlap in candidate IDs")

    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        expected = {int(row["candidate_id"]) for row in csv.DictReader(handle)}
    if set(candidate_ids.tolist()) != expected:
        missing = sorted(expected - set(candidate_ids.tolist()))[:10]
        extra = sorted(set(candidate_ids.tolist()) - expected)[:10]
        raise ValueError(f"merged OOF coverage mismatch: missing={missing}, extra={extra}")

    order = np.argsort(candidate_ids, kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    if not np.array_equal(merged["candidate_id"], np.arange(len(expected))):
        raise ValueError("candidate IDs must form the frozen contiguous manifest ledger")
    if any(not np.isfinite(value).all() for key, value in merged.items() if key != "candidate_id"):
        raise ValueError("merged PAV logits contain NaN/Inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)
    summary = {
        "status": "complete",
        "n_inputs": len(args.inputs),
        "n_candidates": len(candidate_ids),
        "keys": list(keys),
        "input_sha256": {str(path): _sha256(path) for path in args.inputs},
        "manifest_sha256": _sha256(args.manifest),
        "output_sha256": _sha256(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
