#!/usr/bin/env python3
"""Create a deterministic Ship FP_BG/FN_MISS/FP_CLS stratified review queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _rank(row: dict[str, str], seed: str) -> str:
    identity = "|".join(
        (seed, row["reason"], row["category_id"], row["image_id"], row["item_uid"])
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=12)
    parser.add_argument("--seed", default="ship-review-20260901")
    args = parser.parse_args()
    if args.per_stratum <= 0:
        raise ValueError("per-stratum must be positive")
    with args.cases.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [
        row
        for row in rows
        if row.get("class_name") == "ship"
        and row.get("reason") in {"FP_BG", "FN_MISS", "FP_CLS", "FN_CLS"}
    ]
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        strata[(row["reason"], row["category_id"])].append(row)
    selected: list[dict[str, str]] = []
    for key in sorted(strata):
        ordered = sorted(strata[key], key=lambda row: _rank(row, args.seed))
        selected.extend(ordered[: args.per_stratum])
    selected.sort(key=lambda row: (row["reason"], int(row["category_id"]), row["image_id"]))
    output_rows = [
        {
            **row,
            "review_label": "",
            "review_valid": "",
            "review_notes": "",
        }
        for row in selected
    ]
    if not output_rows:
        raise RuntimeError("no eligible Ship error cases were available for review")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "version": "ship_error_review_v1",
        "source": str(args.cases),
        "eligible": len(eligible),
        "selected": len(selected),
        "strata": {f"{reason}:{category}": len(values) for (reason, category), values in strata.items()},
        "allowed_review_labels": ["background", "wrong_fine", "missed_object", "localization", "duplicate", "ambiguous"],
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
