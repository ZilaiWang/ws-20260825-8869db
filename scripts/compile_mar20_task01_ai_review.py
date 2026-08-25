#!/usr/bin/env python3
"""Compile a frozen blind-review TSV into the server review contract CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDS = (
    "card_id",
    "label",
    "confidence",
    "supporting_evidence",
    "counter_evidence",
    "fixed_structure_types",
    "style_only",
    "notes",
)
ALLOWED_LABELS = {
    "same_frame",
    "geometric_overlap",
    "same_local_site",
    "likely_same_airport",
    "not_same_local_site",
    "different_airport",
    "uncertain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--review-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def main() -> None:
    args = parse_args()
    template = read_rows(args.template, ",")
    reviews = read_rows(args.review_tsv, "\t")
    expected_ids = [row["card_id"] for row in template]
    actual_ids = [row["card_id"] for row in reviews]
    if actual_ids != expected_ids:
        raise ValueError("review card IDs/order do not exactly match the blind template")
    if len(actual_ids) != 376 or len(set(actual_ids)) != 376:
        raise ValueError("expected exactly 376 unique blind cards")
    for row in reviews:
        if set(row) != set(FIELDS):
            raise ValueError(f"unexpected columns for {row.get('card_id')}: {sorted(row)}")
        if row["label"] not in ALLOWED_LABELS:
            raise ValueError(f"invalid label for {row['card_id']}: {row['label']}")
        confidence = float(row["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"invalid confidence for {row['card_id']}: {confidence}")
        if not row["supporting_evidence"].strip():
            raise ValueError(f"missing evidence for {row['card_id']}")
        if row["style_only"] not in {"0", "1"}:
            raise ValueError(f"invalid style_only for {row['card_id']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(reviews)
    print(f"BLIND_REVIEW_COMPILE_PASS rows={len(reviews)} output={args.output}")


if __name__ == "__main__":
    main()
