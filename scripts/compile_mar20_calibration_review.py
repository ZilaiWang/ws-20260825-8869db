#!/usr/bin/env python3
"""MG01：校验盲评一致性并编译 descriptor bake-off pair 合同。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from rsdet.grouping.contracts import PROTOCOL_VERSION, EvidenceLabel, atomic_write_json, sha256_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="编译 MAR20 calibration pair 盲评")
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-positive-pairs", type=int, default=30)
    parser.add_argument("--minimum-repeat-agreement", type=float, default=0.90)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mapping_path = Path(args.mapping).expanduser().resolve()
    decisions_path = Path(args.decisions).expanduser().resolve()
    mapping = _read_csv(mapping_path)
    decisions = _read_csv(decisions_path)
    if not mapping or len(mapping) != len(decisions):
        raise ValueError("mapping/decisions 为空或行数不一致")
    mapping_by_card = {row["card_id"]: row for row in mapping}
    decision_by_card = {row["card_id"]: row for row in decisions}
    if len(mapping_by_card) != len(mapping) or len(decision_by_card) != len(decisions):
        raise ValueError("card_id 重复")
    if set(mapping_by_card) != set(decision_by_card):
        raise ValueError("mapping/decisions card_id 集合不一致")
    allowed = {item.value for item in EvidenceLabel}
    combined = []
    missing = []
    for card_id in sorted(mapping_by_card):
        decision = decision_by_card[card_id]
        label = decision["label"].strip()
        if not label:
            missing.append(card_id)
            continue
        if label not in allowed:
            raise ValueError(f"{card_id}: 非法 label={label!r}")
        confidence_text = decision["confidence"].strip()
        try:
            confidence = float(confidence_text)
        except ValueError as error:
            raise ValueError(f"{card_id}: confidence 必须为 [0,1] 数字") from error
        if not 0 <= confidence <= 1:
            raise ValueError(f"{card_id}: confidence 不在 [0,1]")
        combined.append({**mapping_by_card[card_id], **decision, "confidence": confidence})
    if missing:
        raise ValueError(f"尚有 {len(missing)} 张卡未决策: {missing[:10]}")
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in combined:
        by_pair[row["pair_uid"]].append(row)
    repeat_total = 0
    repeat_agree = 0
    conflicts = []
    unique_rows = []
    positive_labels = {
        EvidenceLabel.SAME_FRAME.value,
        EvidenceLabel.GEOMETRIC_OVERLAP.value,
        EvidenceLabel.SAME_LOCAL_SITE.value,
    }
    negative_labels = {
        EvidenceLabel.NOT_SAME_LOCAL_SITE.value,
        EvidenceLabel.DIFFERENT_AIRPORT.value,
    }
    for pair_uid, values in sorted(by_pair.items()):
        labels = [row["label"].strip() for row in values]
        if len(values) > 1:
            repeat_total += len(values) - 1
            repeat_agree += sum(label == labels[0] for label in labels[1:])
            if len(set(labels)) > 1:
                conflicts.append({"pair_uid": pair_uid, "labels": labels})
        selected = sorted(values, key=lambda row: row["card_id"])[0]
        if selected["label"] in positive_labels:
            binary_role = "positive"
        elif selected["label"] in negative_labels:
            binary_role = "negative"
        else:
            binary_role = "excluded_uncertain"
        split_value = int(hashlib.sha256(pair_uid.encode()).hexdigest()[:8], 16) % 10
        split = "held_out_audit" if split_value < 3 else "calibration"
        unique_rows.append(
            {
                "pair_uid": pair_uid,
                "node_u": selected["node_u"],
                "node_v": selected["node_v"],
                "label": selected["label"],
                "binary_role": binary_role,
                "split": split,
                "confidence": selected["confidence"],
                "route": selected["route"],
                "supporting_evidence": selected["supporting_evidence"],
                "counter_evidence": selected["counter_evidence"],
                "notes": selected["notes"],
            }
        )
    agreement = repeat_agree / repeat_total if repeat_total else 1.0
    positives = [row for row in unique_rows if row["binary_role"] == "positive"]
    negatives = [row for row in unique_rows if row["binary_role"] == "negative"]
    failures = []
    if agreement < args.minimum_repeat_agreement:
        failures.append("blind_repeat_agreement_below_threshold")
    if conflicts:
        failures.append("blind_repeat_label_conflicts")
    evidence_sufficient = len(positives) >= args.minimum_positive_pairs
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "calibration_pairs.csv"
    with pairs_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(unique_rows[0]))
        writer.writeheader()
        writer.writerows(unique_rows)
    node_list = sorted(
        {row["node_u"] for row in unique_rows} | {row["node_v"] for row in unique_rows},
        key=lambda value: int(value.split(":")[1]),
    )
    node_path = output_dir / "calibration_node_uids.txt"
    node_path.write_text("\n".join(node_list) + "\n", encoding="utf-8")
    summary = {
        "status": "fail"
        if failures
        else ("pass" if evidence_sufficient else "pass_with_insufficient_positive_evidence"),
        "protocol_version": PROTOCOL_VERSION,
        "repeat_total": repeat_total,
        "repeat_agreement": agreement,
        "repeat_conflicts": conflicts,
        "unique_pair_count": len(unique_rows),
        "positive_pair_count": len(positives),
        "negative_pair_count": len(negatives),
        "excluded_uncertain_count": len(unique_rows) - len(positives) - len(negatives),
        "positive_evidence_sufficient": evidence_sufficient,
        "formal_threshold_admission": not failures and evidence_sufficient,
        "failures": failures,
        "artifacts": {
            "calibration_pairs.csv": sha256_file(pairs_path),
            "calibration_node_uids.txt": sha256_file(node_path),
        },
    }
    atomic_write_json(output_dir / "calibration_compile_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
