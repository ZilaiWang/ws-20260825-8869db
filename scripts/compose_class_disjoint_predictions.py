#!/usr/bin/env python3
"""Compose COCO predictions from disjoint label owners without refitting scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

OFFICIAL_LABELS = frozenset(range(25))


def _labels(text: str) -> frozenset[int]:
    values: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = map(int, token.split("-", 1))
            if last < first:
                raise ValueError(f"descending label range: {token}")
            values.update(range(first, last + 1))
        else:
            values.add(int(token))
    labels = frozenset(values)
    if not labels or labels - OFFICIAL_LABELS:
        raise ValueError(f"invalid label ownership: {sorted(labels)}")
    return labels


def _read(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"predictions must be a COCO list: {path}")
    return payload


def compose(
    primary: list[dict[str, Any]],
    expert: list[dict[str, Any]],
    *,
    primary_labels: frozenset[int],
    expert_labels: frozenset[int],
    primary_threshold: float = 0.0,
    expert_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    if primary_labels & expert_labels:
        raise ValueError("label ownership must be disjoint")
    if primary_labels | expert_labels != OFFICIAL_LABELS:
        raise ValueError("label ownership must cover all official labels")
    for name, threshold in (
        ("primary_threshold", primary_threshold),
        ("expert_threshold", expert_threshold),
    ):
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    rows = [
        dict(row)
        for source, labels, threshold in (
            (primary, primary_labels, float(primary_threshold)),
            (expert, expert_labels, float(expert_threshold)),
        )
        for row in source
        if int(row["category_id"]) in labels and float(row["score"]) >= threshold
    ]
    rows.sort(
        key=lambda row: (
            int(row["image_id"]),
            -float(row["score"]),
            int(row["category_id"]),
            tuple(float(value) for value in row["bbox"]),
        )
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--primary-labels", default="0-23")
    parser.add_argument("--expert-labels", default="24")
    parser.add_argument("--primary-threshold", type=float, default=0.0)
    parser.add_argument("--expert-threshold", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = compose(
        _read(args.primary),
        _read(args.expert),
        primary_labels=_labels(args.primary_labels),
        expert_labels=_labels(args.expert_labels),
        primary_threshold=args.primary_threshold,
        expert_threshold=args.expert_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, separators=(",", ":")) + "\n")
    print(json.dumps({"status": "complete", "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
