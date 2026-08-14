#!/usr/bin/env python3
"""Compile a completed blinded FP_BG review into an audited whitelist."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from rsdet.analysis.fp_bg_audit import (
    HUMAN_LABELS,
    LABEL_CLEAR_BACKGROUND,
    compute_audit_summary,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _unique_by(rows: list[dict[str, str]], key: str, source: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "").strip()
        if not value:
            raise ValueError(f"{source} 存在空 {key}")
        if value in result:
            raise ValueError(f"{source} 存在重复 {key}: {value}")
        result[value] = row
    return result


def compile_review(
    *,
    decisions_path: Path,
    mapping_path: Path,
    audit_path: Path,
    minimum_repeat_consistency: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Join the blinded decisions after review and enforce all science gates."""

    decisions = _read_csv(decisions_path)
    mappings = _read_csv(mapping_path)
    audit_rows = _read_csv(audit_path)
    decision_by_card = _unique_by(decisions, "card_id", "decisions")
    mapping_by_card = _unique_by(mappings, "card_id", "mapping")
    audit_by_uid = _unique_by(audit_rows, "audit_uid", "audit")
    if set(decision_by_card) != set(mapping_by_card):
        missing = sorted(set(mapping_by_card) - set(decision_by_card))
        extra = sorted(set(decision_by_card) - set(mapping_by_card))
        raise ValueError(f"card_id 集合不一致: missing={missing[:5]} extra={extra[:5]}")

    joined: list[dict[str, Any]] = []
    for card_id in sorted(mapping_by_card):
        mapping = mapping_by_card[card_id]
        decision = decision_by_card[card_id]
        audit_uid = mapping.get("audit_uid", "").strip()
        if audit_uid not in audit_by_uid:
            raise ValueError(f"mapping audit_uid 不在冻结 audit 中: {audit_uid}")
        audit = audit_by_uid[audit_uid]
        if mapping.get("proposal_uid", "").strip() != audit.get("proposal_uid", "").strip():
            raise ValueError(f"proposal_uid 不一致: {card_id}")
        label = decision.get("label", "").strip()
        if label not in HUMAN_LABELS:
            raise ValueError(f"{card_id} 标签未填或非法: {label!r}")
        joined.append(
            {
                "card_id": card_id,
                "audit_uid": audit_uid,
                "proposal_uid": audit["proposal_uid"].strip(),
                "image_id": int(audit["image_id"]),
                "fold": int(audit["fold"]),
                "category_id": int(audit["category_id"]),
                "class_name": audit["class_name"].strip(),
                "score": float(audit["score"]),
                "bbox_xyxy": audit["bbox_xyxy"],
                "is_repeat_control": audit.get("is_repeat_control", "").lower() == "true",
                "repeat_of": audit.get("repeat_of", "").strip(),
                "label": label,
                "labeler": decision.get("labeler", "").strip(),
                "notes": decision.get("notes", "").strip(),
            }
        )

    summary = compute_audit_summary(joined)
    repeat_rate = summary["repeat_consistency_rate"]
    repeat_gate = bool(
        summary["repeat_pairs_total"] > 0
        and summary["repeat_pairs_incomplete"] == 0
        and repeat_rate is not None
        and repeat_rate >= minimum_repeat_consistency
    )

    by_proposal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_proposal[row["proposal_uid"]].append(row)
    conflicting = {
        proposal_uid: sorted({row["label"] for row in rows})
        for proposal_uid, rows in by_proposal.items()
        if len({row["label"] for row in rows}) != 1
    }
    whitelist: list[dict[str, Any]] = []
    if repeat_gate and not conflicting:
        for proposal_uid, rows in sorted(by_proposal.items()):
            canonical = next((row for row in rows if not row["is_repeat_control"]), rows[0])
            if canonical["label"] == LABEL_CLEAR_BACKGROUND:
                whitelist.append(
                    {
                        key: canonical[key]
                        for key in (
                            "proposal_uid",
                            "image_id",
                            "fold",
                            "category_id",
                            "class_name",
                            "score",
                            "bbox_xyxy",
                        )
                    }
                )

    decision = {
        "status": "review_compiled" if repeat_gate and not conflicting else "review_gate_failed",
        "manual_review_complete": summary["unlabeled"] == 0,
        "repeat_consistency_gate_passed": repeat_gate,
        "minimum_repeat_consistency": minimum_repeat_consistency,
        "conflicting_proposal_count": len(conflicting),
        "background_whitelist_admission": bool(repeat_gate and not conflicting),
        "background_whitelist_count": len(whitelist),
        "unique_reviewed_proposals": len(by_proposal),
        "card_label_counts": dict(sorted(Counter(row["label"] for row in joined).items())),
        **summary,
    }
    return joined, whitelist, decision


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--decisions", type=Path, required=True)
    value.add_argument("--sealed-mapping", type=Path, required=True)
    value.add_argument("--audit-csv", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--minimum-repeat-consistency", type=float, default=0.85)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 0.0 <= args.minimum_repeat_consistency <= 1.0:
        raise ValueError("minimum-repeat-consistency 必须位于 [0, 1]")
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    joined, whitelist, decision = compile_review(
        decisions_path=args.decisions,
        mapping_path=args.sealed_mapping,
        audit_path=args.audit_csv,
        minimum_repeat_consistency=args.minimum_repeat_consistency,
    )
    _write_csv(destination / "compiled_review.csv", joined, list(joined[0]))
    whitelist_fields = [
        "proposal_uid",
        "image_id",
        "fold",
        "category_id",
        "class_name",
        "score",
        "bbox_xyxy",
    ]
    _write_csv(destination / "clear_background_whitelist.csv", whitelist, whitelist_fields)
    decision["input_sha256"] = {
        "decisions": _sha256(args.decisions),
        "sealed_mapping": _sha256(args.sealed_mapping),
        "audit_csv": _sha256(args.audit_csv),
    }
    (destination / "review_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["background_whitelist_admission"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
