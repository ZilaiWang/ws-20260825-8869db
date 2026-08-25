#!/usr/bin/env python3
"""P04-3 确定性 calibration 子集。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from rsdet.features.p04_cache import sha256_file
from rsdet.features.p04_inputs import select_calibration_uids, write_uid_list


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 P04 raw/CleanDIFT calibration 子集")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", default="tight")
    parser.add_argument("--count", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    values = select_calibration_uids(args.manifest, crop_policy=args.policy, count=args.count)
    output = Path(args.output).expanduser().resolve()
    write_uid_list(output, values)
    selected = set(values)
    class_counts: Counter[str] = Counter()
    major_counts: Counter[str] = Counter()
    fold_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    with (
        Path(args.manifest)
        .expanduser()
        .resolve()
        .open("r", encoding="utf-8-sig", newline="") as file
    ):
        for row in csv.DictReader(file):
            if (
                row["crop_policy"].strip() != args.policy
                or row["annotation_uid"].strip() not in selected
            ):
                continue
            class_counts[f"{row['class_id'].strip()}:{row['class_name'].strip()}"] += 1
            major_counts[row["major_class"].strip()] += 1
            fold_counts[row["fold"].strip()] += 1
            if row["source_edge_risk"].strip().lower() in {"1", "true", "yes"}:
                risk_counts["source_edge_risk"] += 1
            if float(row["padding_fraction"]) > 0:
                risk_counts["padding_positive"] += 1
            if float(row["gt_coverage_fraction"]) < 0.9:
                risk_counts["coverage_lt_0p9"] += 1
            if float(row["gt_short_edge"]) < 48:
                risk_counts["short_edge_lt_48"] += 1
    meta = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "policy": args.policy,
        "count": len(values),
        "uid_list_sha256": sha256_file(output),
        "coverage": {
            "class_counts": dict(sorted(class_counts.items())),
            "major_counts": dict(sorted(major_counts.items())),
            "fold_counts": dict(sorted(fold_counts.items())),
            "risk_counts": dict(sorted(risk_counts.items())),
        },
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
