#!/usr/bin/env python3
"""B0 CPU audit for M1 formal OOF candidates.

The script is intentionally read-only with respect to source artifacts. It
joins the aggregate proposal ledger with the formal crop manifest and writes
compact audit tables for fold/source/size/score analysis. It does not select
deployment thresholds or change predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCORE_BINS = (
    (0.001, 0.01, "0.001-0.01"),
    (0.01, 0.03, "0.01-0.03"),
    (0.03, 0.051, "0.03-0.051"),
    (0.051, 0.10, "0.051-0.10"),
    (0.10, float("inf"), ">=0.10"),
)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1 OOF B0 source/score audit")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _score_bin(score: float) -> str:
    for low, high, label in SCORE_BINS:
        if low <= score < high:
            return label
    raise ValueError(f"score out of audit range: {score}")


def _size_bin(short_edge: float) -> str:
    if short_edge < 32:
        return "<32"
    if short_edge < 64:
        return "32-64"
    if short_edge < 128:
        return "64-128"
    if short_edge < 256:
        return "128-256"
    return ">=256"


def _iter_unique_gt(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        uid = row["annotation_uid"]
        if uid in result:
            continue
        result[uid] = row
    return result


def _summary(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    counters: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        counters[key]["proposals"] += 1
        totals[key] += 1
        if row["score"] >= 0.051:
            counters[key]["kept_at_0.051"] += 1
        if row["score"] < 0.051:
            counters[key]["below_0.051"] += 1
    result: list[dict[str, Any]] = []
    for key in sorted(counters):
        item = {name: value for name, value in zip(keys, key)}
        item.update(counters[key])
        result.append(item)
    return result


def main() -> int:
    args = _args()
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    aggregate = root / "M1-CV3-OOF-aggregate"
    proposals = _read_csv(aggregate / "oof_proposals.csv")
    images = {row["image_id"]: row for row in _read_csv(aggregate / "oof_images.csv")}
    gt = _iter_unique_gt(_read_csv(root / "formal_crop_manifest.csv"))

    if len(proposals) != 55548:
        raise SystemExit(f"unexpected proposal count: {len(proposals)}")
    if len(images) != 4481:
        raise SystemExit(f"unexpected image count: {len(images)}")
    if len(gt) != 20933:
        raise SystemExit(f"unexpected unique GT count: {len(gt)}")

    audit_rows: list[dict[str, Any]] = []
    for row in proposals:
        image = images[row["image_id"]]
        score = _float(row, "score")
        width = _float(row, "width")
        height = _float(row, "height")
        short_edge = min(width, height)
        audit_rows.append(
            {
                "proposal_uid": row["proposal_uid"],
                "image_id": row["image_id"],
                "fold": image["fold"],
                "group_id": image["group_id"],
                "relative_path": image["relative_path"],
                "category_id": row["category_id"],
                "score": score,
                "x": _float(row, "x"),
                "y": _float(row, "y"),
                "width": width,
                "height": height,
                "area": width * height,
                "short_edge": short_edge,
                "size_bin": _size_bin(short_edge),
                "score_bin": _score_bin(score),
            }
        )

    fields = list(audit_rows[0])
    with (output / "proposal_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit_rows)

    summaries = {
        "overall": _summary(audit_rows, ()),
        "fold": _summary(audit_rows, ("fold",)),
        "source_group": _summary(audit_rows, ("group_id",)),
        "category": _summary(audit_rows, ("category_id",)),
        "size": _summary(audit_rows, ("size_bin",)),
        "score": _summary(audit_rows, ("score_bin",)),
        "source_size": _summary(audit_rows, ("group_id", "size_bin")),
    }
    metadata = {
        "status": "complete_b0_proposal_audit",
        "input": {
            "root": str(root),
            "aggregate": str(aggregate),
            "formal_crop_manifest": str(root / "formal_crop_manifest.csv"),
        },
        "counts": {
            "images": len(images),
            "proposals": len(proposals),
            "unique_ground_truth_annotations": len(gt),
        },
        "score_threshold_reference": 0.051,
        "scientific_scope": {
            "diagnostic_only": True,
            "does_not_select_deployment_threshold": True,
            "does_not_change_predictions": True,
        },
        "summary_files": {name: f"{name}_summary.json" for name in summaries},
    }
    (output / "b0_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, payload in summaries.items():
        (output / f"{name}_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"B0_AUDIT_PASS images={len(images)} proposals={len(proposals)} gt={len(gt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
