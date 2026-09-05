#!/usr/bin/env python3
"""Verify D4 batch-size output parity and report paired proxy latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch32-predictions", type=Path, required=True)
    parser.add_argument("--batch64-predictions", type=Path, required=True)
    parser.add_argument("--batch32-summary", type=Path, required=True)
    parser.add_argument("--batch64-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions32 = _load(args.batch32_predictions)
    predictions64 = _load(args.batch64_predictions)
    summary32 = _load(args.batch32_summary)
    summary64 = _load(args.batch64_summary)
    exact = predictions32 == predictions64
    if not exact:
        raise AssertionError("D4 batch=64 changed the emitted predictions")

    time32 = float(summary32["mean_image_seconds"])
    time64 = float(summary64["mean_image_seconds"])
    payload = {
        "status": "complete",
        "protocol": "d4_batch32_vs_batch64_exact_output_paired_proxy_v1",
        "predictions_exact": exact,
        "images": int(summary32["images"]),
        "predictions": int(summary32["predictions"]),
        "batch32_mean_image_seconds": time32,
        "batch64_mean_image_seconds": time64,
        "batch64_minus_batch32_seconds": time64 - time32,
        "batch64_speedup": time32 / time64,
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
