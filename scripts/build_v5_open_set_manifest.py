#!/usr/bin/env python3
"""Build the aligned three-way V5 proposal-domain open-set manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.background_gate import coarse_of_category_id
from rsdet.analysis.proposal_open_set import (
    OPEN_IGNORE,
    OPEN_LABEL_NAMES,
    proposal_open_set_label,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    with args.proposals.open(encoding="utf-8-sig", newline="") as handle:
        proposal_reader = csv.DictReader(handle)
        proposal_columns = set(proposal_reader.fieldnames or ())
        proposals = list(proposal_reader)
    with args.nodes.open(encoding="utf-8-sig", newline="") as handle:
        node_reader = csv.DictReader(handle)
        node_columns = set(node_reader.fieldnames or ())
        nodes = list(node_reader)
    required_proposals = {
        "proposal_uid", "image_id", "fold", "relative_path", "category_id",
        "x0", "y0", "x1", "y1", "detector_score", "source_prediction_index",
    }
    required_nodes = {
        "proposal_uid", "image_id", "category_id", "is_valid", "fine_correct",
        "crop_top1_class",
    }
    if required_proposals - proposal_columns or required_nodes - node_columns:
        raise ValueError("proposal/node manifest is missing required columns")
    proposal_uids = [row["proposal_uid"] for row in proposals]
    node_uids = [row["proposal_uid"] for row in nodes]
    if len(set(proposal_uids)) != len(proposal_uids) or len(set(node_uids)) != len(node_uids):
        raise ValueError("proposal_uid must be unique")
    if len(proposals) != len(nodes) or set(proposal_uids) != set(node_uids):
        raise ValueError("proposal and node ledgers must cover the same candidates")

    evidence = {row["proposal_uid"]: row for row in nodes}
    rows: list[dict[str, object]] = []
    counts = {name: 0 for name in OPEN_LABEL_NAMES}
    ignored = 0
    per_coarse = {
        coarse: {name: 0 for name in OPEN_LABEL_NAMES} | {"ignore": 0}
        for coarse in ("ship", "aircraft", "vehicle")
    }
    for proposal in proposals:
        node = evidence[str(proposal["proposal_uid"])]
        if int(node["image_id"]) != int(proposal["image_id"]) or int(
            node["category_id"]
        ) != int(
            proposal["category_id"]
        ):
            raise ValueError(f"candidate identity mismatch: {proposal['proposal_uid']}")
        label = proposal_open_set_label(
            is_valid=bool(int(node["is_valid"])),
            fine_correct=bool(int(node["fine_correct"])),
            crop_top1_class=int(node["crop_top1_class"]),
        )
        coarse = coarse_of_category_id(int(proposal["category_id"]))
        if label == OPEN_IGNORE:
            label_name = "ignore"
            ignored += 1
        else:
            label_name = OPEN_LABEL_NAMES[label]
            counts[label_name] += 1
        per_coarse[coarse][label_name] += 1
        rows.append(
            {
                "candidate_index": len(rows),
                "proposal_uid": str(proposal["proposal_uid"]),
                "image_id": int(proposal["image_id"]),
                "fold": int(proposal["fold"]),
                "relative_path": str(proposal["relative_path"]),
                "category_id": int(proposal["category_id"]),
                "coarse": coarse,
                "detector_score": float(proposal["detector_score"]),
                "source_prediction_index": int(proposal["source_prediction_index"]),
                "x0": float(proposal["x0"]),
                "y0": float(proposal["y0"]),
                "x1": float(proposal["x1"]),
                "y1": float(proposal["y1"]),
                "open_set_label": int(label),
                "open_set_label_name": label_name,
                "crop_top1_class": int(node["crop_top1_class"]),
            }
        )
    if {int(row["fold"]) for row in proposals} != {0, 1, 2}:
        raise ValueError("manifest must cover all three formal folds")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "complete",
        "protocol": "v5_three_way_proposal_open_set_manifest_v1",
        "rows": len(rows),
        "labeled_rows": len(rows) - ignored,
        "ignored_matchable_nonwinners": ignored,
        "label_counts": counts,
        "per_coarse": per_coarse,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
