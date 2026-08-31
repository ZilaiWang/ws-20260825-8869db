#!/usr/bin/env python3
"""Replace exactly one source-fold in a frozen baseline prediction ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_fold(base: list[dict], candidate: list[dict], fold: int) -> list[dict]:
    if fold not in (0, 1, 2):
        raise ValueError("fold must be 0, 1, or 2")
    if any(int(row.get("source_fold", -1)) != fold for row in candidate):
        raise ValueError("candidate ledger contains a row from a non-candidate fold")
    kept = [row for row in base if int(row.get("source_fold", -1)) != fold]
    base_other_images = {
        int(row["image_id"]) for row in kept
    }
    candidate_images = {int(row["image_id"]) for row in candidate}
    if base_other_images & candidate_images:
        raise ValueError("candidate image IDs overlap a non-replaced base fold")
    output = kept + candidate
    output.sort(
        key=lambda row: (
            int(row["image_id"]),
            -float(row["score"]),
            int(row["category_id"]),
            tuple(float(value) for value in row["bbox"]),
        )
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate-fold", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_fold.read_text(encoding="utf-8"))
    if not isinstance(base, list) or not isinstance(candidate, list):
        raise ValueError("prediction ledgers must be COCO result lists")
    output = replace_fold(base, candidate, args.fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False) + "\n")
    audit = {
        "status": "complete",
        "protocol": "replace_exactly_one_candidate_prediction_fold_v1",
        "fold": args.fold,
        "base_rows": len(base),
        "candidate_fold_rows": len(candidate),
        "output_rows": len(output),
        "removed_base_fold_rows": sum(
            int(row.get("source_fold", -1)) == args.fold for row in base
        ),
        "base_sha256": _sha256(args.base),
        "candidate_fold_sha256": _sha256(args.candidate_fold),
        "output_sha256": _sha256(args.output),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
