#!/usr/bin/env python3
"""Append the existing cross-fit F1 foreground logit to a crop-only OMQ cache."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--proposal-manifest", type=Path, required=True)
    parser.add_argument("--foreground-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    rows = list(csv.DictReader(args.proposal_manifest.open(encoding="utf-8-sig", newline="")))
    logits = json.loads(args.foreground_logits.read_text(encoding="utf-8"))
    n = int(arrays["features"].shape[0])
    if len(rows) != n or [int(row["candidate_index"]) for row in rows] != list(range(n)):
        raise ValueError("proposal manifest is not aligned with OMQ cache")
    proposal_uids = [row["proposal_uid"] for row in rows]
    if len(logits) != n or set(logits) != set(proposal_uids):
        raise ValueError("foreground logit ledger does not cover the same proposals")
    values = np.asarray([float(logits[uid]) for uid in proposal_uids], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("foreground logits contain NaN/Inf")
    base_dimension = int(arrays["features"].shape[1])
    arrays["features"] = np.concatenate(
        (arrays["features"].astype(np.float16), values[:, None].astype(np.float16)), axis=1
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    summary = {
        "status": "complete",
        "protocol": "v5_crop_omq_plus_existing_crossfit_f1_foreground_logit_v1",
        "rows": n,
        "base_dimension": base_dimension,
        "output_dimension": int(arrays["features"].shape[1]),
        "logit_mean": float(values.mean()),
        "logit_std": float(values.std()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
