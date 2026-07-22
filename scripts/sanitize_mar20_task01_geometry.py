#!/usr/bin/env python3
"""Convert degenerate OpenCV transform outputs to explicit missing evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence

from rsdet.grouping.contracts import atomic_write_json, sha256_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize MAR20 TASK-01 geometry evidence")
    parser.add_argument("--raw-evidence", required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-rows", type=int, default=6000)
    parser.add_argument("--expected-nonfinite-fields", type=int, default=87)
    parser.add_argument("--expected-affected-pairs", type=int, default=29)
    return parser.parse_args(argv)


def _matrix_is_finite(value: str) -> bool:
    try:
        numbers = [float(part) for part in value.split(";") if part]
    except ValueError:
        return False
    return bool(numbers) and all(math.isfinite(number) for number in numbers)


def _positive_number(value: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_path = Path(args.raw_evidence).expanduser().resolve()
    actual_sha = sha256_file(raw_path)
    if actual_sha != args.expected_raw_sha256.lower():
        raise ValueError(
            f"raw evidence SHA expected={args.expected_raw_sha256}, actual={actual_sha}"
        )
    with raw_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        input_fields = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != args.expected_rows or len({row["pair_uid"] for row in rows}) != len(rows):
        raise ValueError("raw evidence row count or pair uniqueness mismatch")
    affected_pairs = set()
    affected_fields: Counter[str] = Counter()
    output_rows = []
    for row in rows:
        sanitized = dict(row)
        changed = []
        for key, text in row.items():
            if not text:
                continue
            nonfinite = False
            if key.endswith("_matrix"):
                nonfinite = not _matrix_is_finite(text)
            else:
                try:
                    number = float(text)
                except ValueError:
                    continue
                nonfinite = not math.isfinite(number)
            if nonfinite:
                sanitized[key] = ""
                changed.append(key)
                affected_fields[key] += 1
        for model in ("similarity", "affine", "homography"):
            valid = (
                _matrix_is_finite(sanitized.get(f"{model}_matrix", ""))
                and _positive_number(sanitized.get(f"{model}_inliers", ""))
                and bool(sanitized.get(f"{model}_median_error", ""))
                and bool(sanitized.get(f"{model}_p95_error", ""))
            )
            sanitized[f"{model}_fit_valid"] = int(valid)
        sanitized["sanitized_nonfinite_fields"] = ";".join(sorted(changed))
        if changed:
            affected_pairs.add(row["pair_uid"])
        output_rows.append(sanitized)
    nonfinite_count = sum(affected_fields.values())
    if (
        nonfinite_count != args.expected_nonfinite_fields
        or len(affected_pairs) != args.expected_affected_pairs
    ):
        raise ValueError(
            "unexpected degenerate evidence count: "
            f"fields={nonfinite_count}/{args.expected_nonfinite_fields}, "
            f"pairs={len(affected_pairs)}/{args.expected_affected_pairs}"
        )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / "pair_evidence_sanitized.csv"
    output_fields = [
        *input_fields,
        "similarity_fit_valid",
        "affine_fit_valid",
        "homography_fit_valid",
        "sanitized_nonfinite_fields",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)
    # Re-read the written artifact and independently prove that no numeric NaN/Inf remains.
    remaining = 0
    with output_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            for key, text in row.items():
                if not text or key == "sanitized_nonfinite_fields":
                    continue
                if key.endswith("_matrix"):
                    remaining += not _matrix_is_finite(text)
                    continue
                try:
                    number = float(text)
                except ValueError:
                    continue
                remaining += not math.isfinite(number)
    if remaining:
        raise ValueError(f"sanitized evidence still contains {remaining} non-finite values")
    sanitized_sha = sha256_file(output_path)
    summary = {
        "status": "pass",
        "policy": "degenerate_transform_is_missing_evidence_not_zero_or_positive_evidence",
        "row_count": len(output_rows),
        "raw_evidence_sha256": actual_sha,
        "raw_nonfinite_field_count": nonfinite_count,
        "affected_pair_count": len(affected_pairs),
        "affected_field_counts": dict(sorted(affected_fields.items())),
        "remaining_nonfinite_count": remaining,
        # Keep the verifier summary contract so the existing calibration and
        # blind-review programs can consume this audited amendment directly.
        "pair_evidence_sha256": sanitized_sha,
        "sanitized_evidence_sha256": sanitized_sha,
        "formal_grouping_admission": False,
    }
    atomic_write_json(output / "geometry_sanitization_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
