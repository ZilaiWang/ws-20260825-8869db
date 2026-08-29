#!/usr/bin/env python3
"""Compare audited pseudo-10K frontier artifacts without re-running inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_condition(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must be LABEL=FRONTIER_JSON")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("condition must contain a label and path")
    return label.strip(), Path(raw_path).expanduser().resolve()


def _row(label: str, path: Path, fdr_level: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"{label}: frontier status is not complete")
    frontier = payload.get("frontiers", {}).get(fdr_level)
    if not isinstance(frontier, dict) or not isinstance(frontier.get("crossfit"), dict):
        raise ValueError(f"{label}: frontier does not contain FDR={fdr_level}")
    floor = payload.get("candidate_floor")
    if not isinstance(floor, dict):
        raise ValueError(f"{label}: candidate_floor missing")
    selected = frontier["crossfit"]
    row: dict[str, Any] = {
        "condition": label,
        "frontier_path": str(path),
        "candidate_recall": float(floor["recall"]),
        "candidate_fdr": float(floor["fdr"]),
        "recall": float(selected["recall"]),
        "fdr": float(selected["fdr"]),
        "macro_recall": float(selected["macro_recall"]),
        "macro_fdr": float(selected["macro_fdr"]),
    }
    for coarse in ("ship", "aircraft", "vehicle"):
        row[f"candidate_{coarse}_recall"] = float(floor["per_coarse"][coarse]["recall"])
        row[f"{coarse}_recall"] = float(selected["per_coarse"][coarse]["recall"])
        row[f"{coarse}_fdr"] = float(selected["per_coarse"][coarse]["fdr"])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", type=_parse_condition, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--fdr-level", type=float, default=0.15)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    labels = [label for label, _ in args.condition]
    if len(set(labels)) != len(labels):
        raise ValueError("condition labels must be unique")
    if args.baseline not in labels:
        raise ValueError("baseline must name one supplied condition")
    fdr_key = f"{args.fdr_level:.3f}"
    rows = [_row(label, path, fdr_key) for label, path in args.condition]
    baseline = next(row for row in rows if row["condition"] == args.baseline)
    for row in rows:
        row["delta_recall_vs_baseline"] = row["recall"] - baseline["recall"]
        row["delta_fdr_vs_baseline"] = row["fdr"] - baseline["fdr"]
        for coarse in ("ship", "aircraft", "vehicle"):
            row[f"delta_{coarse}_recall_vs_baseline"] = (
                row[f"{coarse}_recall"] - baseline[f"{coarse}_recall"]
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "complete",
        "protocol": "audited_pseudo10k_frontier_comparison_v1",
        "fdr_level": args.fdr_level,
        "baseline": args.baseline,
        "rows": rows,
    }
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
