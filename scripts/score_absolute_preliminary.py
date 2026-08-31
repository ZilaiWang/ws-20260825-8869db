#!/usr/bin/env python3
"""Compute the 2026-08-31 absolute score without guessing class aggregation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.absolute_score import competition_score, score_coarse_interpretations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recall", type=float)
    parser.add_argument("--fdr", type=float)
    parser.add_argument("--latency-seconds", type=float, required=True)
    parser.add_argument(
        "--per-coarse-json",
        type=Path,
        help="JSON object, or an object containing a per_coarse field",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.per_coarse_json is not None:
        raw = json.loads(args.per_coarse_json.read_text(encoding="utf-8"))
        per_coarse = raw.get("per_coarse", raw)
        payload = score_coarse_interpretations(per_coarse, args.latency_seconds)
    else:
        if args.recall is None or args.fdr is None:
            parser.error("provide --recall and --fdr, or --per-coarse-json")
        payload = competition_score(args.recall, args.fdr, args.latency_seconds)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
