#!/usr/bin/env python3
"""E1: remove cross-coarse foreground pollution from a proposal manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--policy", choices=("hard_negative", "exclude"), default="hard_negative")
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    if not source:
        raise ValueError("input manifest is empty")

    output_rows = []
    counts: Counter[str] = Counter()
    for row in source:
        result = dict(row)
        predicted_coarse = str(row["coarse"])
        is_foreground = str(row["is_foreground"]).strip() in {"1", "true", "True"}
        support_value = str(row.get("support_gt_category_id", "")).strip()
        support_coarse = (
            "" if not support_value else protocol.category_mapping[int(support_value)]
        )
        mismatch = bool(is_foreground and support_coarse != predicted_coarse)
        result["legacy_is_foreground"] = int(is_foreground)
        result["support_gt_coarse"] = support_coarse
        result["cross_coarse_foreground"] = int(mismatch)
        if mismatch and args.policy == "exclude":
            counts["excluded_cross_coarse"] += 1
            continue
        if mismatch:
            result["is_foreground"] = 0
            counts["converted_to_hard_negative"] += 1
        counts[f"{predicted_coarse}/foreground{int(result['is_foreground'])}"] += 1
        output_rows.append(result)
    for held_out in (0, 1, 2):
        training = [row for row in output_rows if int(row["fold"]) != held_out]
        for coarse in ("ship", "aircraft", "vehicle"):
            for target in (0, 1):
                if not any(
                    row["coarse"] == coarse and int(row["is_foreground"]) == target
                    for row in training
                ):
                    raise ValueError(
                        f"clean manifest lacks held_out={held_out}/{coarse}/target={target}"
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    payload = {
        "status": "clean_coarse_manifest_complete",
        "policy": args.policy,
        "input_rows": len(source),
        "output_rows": len(output_rows),
        "counts": dict(sorted(counts.items())),
        "input_sha256": _sha256(args.input),
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
