#!/usr/bin/env python3
"""Prove prediction parity and compare two equivalent runtime implementations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_predictions = _load(args.reference_predictions)
    candidate_predictions = _load(args.candidate_predictions)
    reference_summary = _load(args.reference_summary)
    candidate_summary = _load(args.candidate_summary)
    if reference_predictions != candidate_predictions:
        raise AssertionError("candidate runtime changed the emitted predictions")
    for field in ("images", "predictions"):
        if int(reference_summary[field]) != int(candidate_summary[field]):
            raise AssertionError(f"runtime summaries disagree on {field}")

    reference_time = float(reference_summary["mean_image_seconds"])
    candidate_time = float(candidate_summary["mean_image_seconds"])
    if reference_time <= 0 or candidate_time <= 0:
        raise ValueError("runtime means must be positive")
    payload = {
        "status": "complete",
        "protocol": args.protocol,
        "predictions_exact": True,
        "images": int(reference_summary["images"]),
        "predictions": int(reference_summary["predictions"]),
        "reference_mean_image_seconds": reference_time,
        "candidate_mean_image_seconds": candidate_time,
        "candidate_minus_reference_seconds": candidate_time - reference_time,
        "candidate_speedup": reference_time / candidate_time,
        "timing_is_proxy_diagnostic_not_official_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
