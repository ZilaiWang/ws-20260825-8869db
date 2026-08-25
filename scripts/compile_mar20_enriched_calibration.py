#!/usr/bin/env python3
"""Compile 00B enriched review and create component-isolated calibration splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from rsdet.grouping.contracts import (
    MASKED_PATCH_PROTOCOL_VERSION,
    EvidenceLabel,
    atomic_write_json,
    parse_node_uid,
    sha256_file,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile enriched MAR20 calibration")
    parser.add_argument("--prior-pairs", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-positive-pairs", type=int, default=30)
    parser.add_argument("--recommended-positive-pairs", type=int, default=60)
    parser.add_argument("--minimum-heldout-positive-pairs", type=int, default=5)
    parser.add_argument("--recommended-heldout-positive-pairs", type=int, default=15)
    parser.add_argument("--minimum-calibration-positive-pairs", type=int, default=5)
    parser.add_argument("--minimum-repeat-agreement", type=float, default=0.90)
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    return parser.parse_args(argv)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if parse_node_uid(root_left) > parse_node_uid(root_right):
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left


def _stable_fraction(value: str) -> float:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) / float(16**16)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prior_path = Path(args.prior_pairs).expanduser().resolve()
    mapping_path = Path(args.mapping).expanduser().resolve()
    decisions_path = Path(args.decisions).expanduser().resolve()
    prior = _read(prior_path)
    mapping = _read(mapping_path)
    decisions = _read(decisions_path)
    by_card = {row["card_id"]: row for row in mapping}
    decision_by_card = {row["card_id"]: row for row in decisions}
    if (
        len(by_card) != len(mapping)
        or len(decision_by_card) != len(decisions)
        or set(by_card) != set(decision_by_card)
    ):
        raise ValueError("mapping/decision card contract mismatch")
    allowed = {label.value for label in EvidenceLabel}
    combined_cards = []
    for card_id in sorted(by_card):
        decision = decision_by_card[card_id]
        label = decision["label"].strip()
        if label not in allowed:
            raise ValueError(f"{card_id}: missing or invalid label")
        try:
            confidence = float(decision["confidence"])
        except ValueError as error:
            raise ValueError(f"{card_id}: invalid confidence") from error
        if not 0 <= confidence <= 1:
            raise ValueError(f"{card_id}: confidence outside [0,1]")
        combined_cards.append({**by_card[card_id], **decision, "confidence": confidence})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combined_cards:
        grouped[row["pair_uid"]].append(row)
    repeat_total = repeat_agree = 0
    repeat_conflicts = []
    enriched_rows = []
    strict_labels = {
        EvidenceLabel.SAME_FRAME.value,
        EvidenceLabel.GEOMETRIC_OVERLAP.value,
        EvidenceLabel.SAME_LOCAL_SITE.value,
    }
    negative_labels = {
        EvidenceLabel.NOT_SAME_LOCAL_SITE.value,
        EvidenceLabel.DIFFERENT_AIRPORT.value,
    }
    for pair_uid, values in sorted(grouped.items()):
        labels = [row["label"].strip() for row in values]
        if len(values) > 1:
            repeat_total += len(values) - 1
            repeat_agree += sum(label == labels[0] for label in labels[1:])
            if len(set(labels)) > 1:
                repeat_conflicts.append({"pair_uid": pair_uid, "labels": labels})
        selected = sorted(values, key=lambda row: row["card_id"])[0]
        role = (
            "positive"
            if selected["label"] in strict_labels
            else ("negative" if selected["label"] in negative_labels else "excluded_uncertain")
        )
        enriched_rows.append(
            {
                "pair_uid": pair_uid,
                "node_u": selected["node_u"],
                "node_v": selected["node_v"],
                "label": selected["label"],
                "binary_role": role,
                "confidence": selected["confidence"],
                "route": selected["route"],
                "supporting_evidence": selected["supporting_evidence"],
                "counter_evidence": selected["counter_evidence"],
                "notes": selected["notes"],
                "target_relation": selected.get("target_relation", ""),
                "evidence_batch": "00b_enriched",
            }
        )
    agreement = repeat_agree / repeat_total if repeat_total else 1.0
    merged: dict[str, dict[str, Any]] = {}
    conflicts = []
    for source, source_rows in (("round_a", prior), ("00b_enriched", enriched_rows)):
        for row in source_rows:
            normalized = {
                "pair_uid": row["pair_uid"],
                "node_u": row["node_u"],
                "node_v": row["node_v"],
                "label": row["label"],
                "binary_role": row["binary_role"],
                "confidence": row.get("confidence", ""),
                "route": row.get("route", ""),
                "supporting_evidence": row.get("supporting_evidence", ""),
                "counter_evidence": row.get("counter_evidence", ""),
                "notes": row.get("notes", ""),
                "target_relation": row.get("target_relation", ""),
                "evidence_batch": source,
            }
            previous = merged.get(row["pair_uid"])
            if previous is not None:
                if previous["label"] != normalized["label"]:
                    conflicts.append(
                        {
                            "pair_uid": row["pair_uid"],
                            "labels": [previous["label"], normalized["label"]],
                        }
                    )
                continue
            merged[row["pair_uid"]] = normalized
    rows = list(merged.values())
    positives = [row for row in rows if row["binary_role"] == "positive"]
    uf = UnionFind()
    for row in positives:
        uf.union(row["node_u"], row["node_v"])
    component_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        component_pairs[uf.find(row["node_u"])].append(row)
    target_heldout = max(
        args.minimum_heldout_positive_pairs,
        round(len(positives) * args.heldout_fraction),
    )
    ordered_components = sorted(
        component_pairs,
        key=lambda root: (
            len(component_pairs[root]),
            _stable_fraction(f"component|{root}"),
            parse_node_uid(root),
        ),
    )
    heldout_components: set[str] = set()
    heldout_count = 0
    # Always leave at least one independent component for calibration.
    for root in ordered_components[:-1]:
        if heldout_count >= target_heldout:
            break
        heldout_components.add(root)
        heldout_count += len(component_pairs[root])
    component_split = {
        root: ("held_out_audit" if root in heldout_components else "calibration")
        for root in component_pairs
    }
    node_component = {node: uf.find(node) for node in uf.parent}
    for row in rows:
        if row["binary_role"] == "positive":
            row["split"] = component_split[uf.find(row["node_u"])]
            row["strict_component_id"] = f"strict:{parse_node_uid(uf.find(row['node_u']))}"
        else:
            left_component = node_component.get(row["node_u"])
            right_component = node_component.get(row["node_v"])
            if left_component is not None and left_component == right_component:
                row["split"] = component_split[left_component]
                row["strict_component_id"] = f"strict:{parse_node_uid(left_component)}"
            else:
                row["split"] = (
                    "held_out_audit"
                    if _stable_fraction(f"pair|{row['pair_uid']}") < args.heldout_fraction
                    else "calibration"
                )
                row["strict_component_id"] = ""
    positive_component_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["binary_role"] == "positive":
            positive_component_splits[row["strict_component_id"]].add(row["split"])
    cross_split_components = [
        key for key, values in positive_component_splits.items() if len(values) != 1
    ]
    heldout_positives = [row for row in positives if row["split"] == "held_out_audit"]
    calibration_positives = [row for row in positives if row["split"] == "calibration"]
    positive_by_batch: dict[str, int] = defaultdict(int)
    for row in positives:
        positive_by_batch[str(row["evidence_batch"])] += 1
    failures = []
    if agreement < args.minimum_repeat_agreement:
        failures.append("blind_repeat_agreement_below_threshold")
    if repeat_conflicts:
        failures.append("blind_repeat_conflicts")
    if conflicts:
        failures.append("prior_enriched_label_conflicts")
    if cross_split_components:
        failures.append("strict_component_split_leakage")
    minimum_evidence = bool(
        len(positives) >= args.minimum_positive_pairs
        and len(heldout_positives) >= args.minimum_heldout_positive_pairs
        and len(calibration_positives) >= args.minimum_calibration_positive_pairs
    )
    recommended = bool(
        len(positives) >= args.recommended_positive_pairs
        and len(heldout_positives) >= args.recommended_heldout_positive_pairs
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: row["pair_uid"])
    pairs_path = output / "calibration_pairs_v1p2.csv"
    with pairs_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    nodes = sorted(
        {row["node_u"] for row in rows} | {row["node_v"] for row in rows}, key=parse_node_uid
    )
    node_path = output / "calibration_node_uids_v1p2.txt"
    node_path.write_text("\n".join(nodes) + "\n", encoding="utf-8")
    if failures:
        status = "fail"
    elif recommended:
        status = "pass_recommended_evidence_target"
    elif minimum_evidence:
        status = "pass_minimum_evidence_only"
    else:
        status = "pass_with_insufficient_positive_evidence"
    summary = {
        "status": status,
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "repeat_total": repeat_total,
        "repeat_agreement": agreement,
        "repeat_conflicts": repeat_conflicts,
        "merge_conflicts": conflicts,
        "pair_count": len(rows),
        "positive_pair_count": len(positives),
        "positive_pair_count_by_batch": dict(sorted(positive_by_batch.items())),
        "heldout_positive_pair_count": len(heldout_positives),
        "calibration_positive_pair_count": len(calibration_positives),
        "heldout_positive_direction_count": 2 * len(heldout_positives),
        "strict_component_count": len(component_pairs),
        "strict_component_cross_split_count": len(cross_split_components),
        "minimum_evidence_admission": not failures and minimum_evidence,
        "recommended_evidence_target_met": not failures and recommended,
        "formal_threshold_admission": not failures and minimum_evidence,
        "thresholds": {
            "minimum_positive_pairs": args.minimum_positive_pairs,
            "recommended_positive_pairs": args.recommended_positive_pairs,
            "minimum_heldout_positive_pairs": args.minimum_heldout_positive_pairs,
            "recommended_heldout_positive_pairs": args.recommended_heldout_positive_pairs,
            "minimum_calibration_positive_pairs": args.minimum_calibration_positive_pairs,
            "minimum_repeat_agreement": args.minimum_repeat_agreement,
        },
        "failures": failures,
        "artifacts": {
            "calibration_pairs_v1p2.csv": sha256_file(pairs_path),
            "calibration_node_uids_v1p2.txt": sha256_file(node_path),
        },
    }
    atomic_write_json(output / "calibration_compile_summary_v1p2.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
