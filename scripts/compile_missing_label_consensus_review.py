#!/usr/bin/env python3
"""Compile a completed missing-label review into immutable patch candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ALLOWED = {"confirmed_missing", "ambiguous_ignore", "rejected"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(row: dict[str, str]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "fold": int(row["fold"]),
        "file_name": row["file_name"],
        "category_id": int(row["category_id"]),
        "bbox_xyxy": [
            float(row["bbox_x1"]),
            float(row["bbox_y1"]),
            float(row["bbox_x2"]),
            float(row["bbox_y2"]),
        ],
        "evidence": {
            "primary_score": float(row["primary_score"]),
            "support_score": float(row["support_score"]),
            "support_iou": float(row["support_iou"]),
            "agreement_product": float(row["agreement_product"]),
            "maximum_gt_iou": float(row["maximum_gt_iou"]),
        },
        "review_note": row.get("review_note", "").strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument(
        "--decision-csv",
        type=Path,
        help="Optional compact candidate_id/human_decision/review_note overlay.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.review_csv.open(encoding="utf-8")))
    if not rows:
        raise ValueError("review CSV is empty")
    candidate_ids = [row["candidate_id"] for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_id contains duplicates")

    decision_csv_sha256 = None
    if args.decision_csv is not None:
        decisions = list(csv.DictReader(args.decision_csv.open(encoding="utf-8")))
        decision_ids = [row["candidate_id"] for row in decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("decision overlay candidate_id contains duplicates")
        missing = sorted(set(candidate_ids) - set(decision_ids))
        unexpected = sorted(set(decision_ids) - set(candidate_ids))
        if missing or unexpected:
            raise ValueError(
                f"decision overlay does not exactly cover review candidates: "
                f"missing={missing}, unexpected={unexpected}"
            )
        by_id = {row["candidate_id"]: row for row in decisions}
        for row in rows:
            overlay = by_id[row["candidate_id"]]
            row["human_decision"] = overlay.get("human_decision", "").strip()
            row["review_note"] = overlay.get("review_note", "").strip()
        decision_csv_sha256 = _sha256(args.decision_csv)

    incomplete = [row["candidate_id"] for row in rows if not row["human_decision"].strip()]
    invalid = {
        row["candidate_id"]: row["human_decision"].strip()
        for row in rows
        if row["human_decision"].strip()
        and row["human_decision"].strip() not in ALLOWED
    }
    if invalid:
        raise ValueError(f"invalid human decisions: {invalid}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if incomplete:
        decision = {
            "status": "waiting_for_manual_missing_label_review",
            "annotation_patch_admission": False,
            "ignore_patch_admission": False,
            "row_count": len(rows),
            "incomplete_count": len(incomplete),
            "incomplete_candidate_ids": incomplete,
            "review_csv_sha256": _sha256(args.review_csv),
            "decision_csv_sha256": decision_csv_sha256,
        }
        (output_dir / "decision.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 2

    buckets: dict[str, list[dict[str, Any]]] = {decision: [] for decision in ALLOWED}
    for row in rows:
        buckets[row["human_decision"].strip()].append(_record(row))
    file_for_decision = {
        "confirmed_missing": "confirmed_missing_v1.json",
        "ambiguous_ignore": "ignored_ambiguous_v1.json",
        "rejected": "rejected_candidates_v1.json",
    }
    output_sha: dict[str, str] = {}
    for name, records in buckets.items():
        path = output_dir / file_for_decision[name]
        path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output_sha[path.name] = _sha256(path)

    admitted = bool(buckets["confirmed_missing"] or buckets["ambiguous_ignore"])
    status = "ready_for_annotation_patch_experiment" if admitted else "complete_no_patch"
    decision = {
        "status": status,
        "protocol": "human_compiled_vehicle_missing_label_patch_v1",
        "annotation_patch_admission": bool(buckets["confirmed_missing"]),
        "ignore_patch_admission": bool(buckets["ambiguous_ignore"]),
        "counts": {name: len(records) for name, records in buckets.items()},
        "review_csv": str(args.review_csv.resolve()),
        "review_csv_sha256": _sha256(args.review_csv),
        "decision_csv_sha256": decision_csv_sha256,
        "output_sha256": output_sha,
        "contract": {
            "confirmed_missing": "may be added as category 24 only in a paired experiment",
            "ambiguous_ignore": "may suppress negative loss only in a paired experiment",
            "rejected": "must remain ordinary background supervision",
        },
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(
            {
                "review_csv_sha256": decision["review_csv_sha256"],
                "decision_csv_sha256": decision["decision_csv_sha256"],
                "outputs": output_sha,
                "automatic_annotation_admission": False,
                "automatic_ignore_admission": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
