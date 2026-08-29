#!/usr/bin/env python3
"""Fuse pre-registered proposal evidence with a fixed log-geometric mean."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometric_score(
    item: Mapping[str, Any], fields: Sequence[str], weights: Sequence[float]
) -> float:
    if len(fields) != len(weights) or not fields:
        raise ValueError("fields and weights must be non-empty and aligned")
    total = float(sum(weights))
    if total <= 0.0 or any(weight < 0.0 for weight in weights):
        raise ValueError("weights must be non-negative with positive sum")
    epsilon = 1e-10
    value = 0.0
    for field, weight in zip(fields, weights, strict=True):
        evidence = float(item[field])
        if not math.isfinite(evidence) or evidence < 0.0:
            raise ValueError(f"invalid evidence {field}={evidence}")
        value += weight / total * math.log(max(evidence, epsilon))
    return math.exp(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--field", action="append", required=True)
    parser.add_argument("--weight", action="append", type=float, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    output = []
    for item in rows:
        row = dict(item)
        row["score"] = geometric_score(item, args.field, args.weight)
        output.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "protocol": "fixed_log_geometric_evidence_fusion_v1",
        "fields": args.field,
        "weights": args.weight,
        "candidate_count": len(output),
        "input_sha256": _sha256(args.input),
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
