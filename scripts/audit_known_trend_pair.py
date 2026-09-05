#!/usr/bin/env python3
"""Recompute the known S1024 -> P40 historical direction without refitting.

No new accuracy claim: this reads frozen historical summaries and exposes
where their per-class signs disagreed with the already-known platform pair.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from rsdet.evaluation.absolute_score import COARSE_CLASSES
from rsdet.experiments.paired_trend import read, sha, write
from scripts.compare_candidate_trend import compare
from scripts.run_paired_anchor_audit import direction, official_anchor


def audit(root: Path) -> dict:
    official_path = root / "reports/experiments/formal_attempt2_20260903/analysis_summary.json"
    official = official_anchor(read(official_path))
    result = {
        "status": "complete_historical_summary_audit",
        "official": official, "official_source_sha256": sha(official_path),
        "historical": {}, "new_paired_benchmark_result": False,
        "official_forecast_validated": False, "prospective_comparisons": 0,
        "no_threshold_or_dataset_refitting": True,
        "limitations": [
            "single known formal pair; repeated test views are not independent submissions",
            "CV3 used 40+40 epochs; official full used 160+40 and other deployment changes",
            "historical FDR15 points retained; not the new same-model development selection",
            "summary artifacts verified and rescored, not fresh per-box inference",
        ],
    }
    for split in ("hard", "sentinel"):
        bpath = root / "outputs/SCALEROUTE-PLAN15-R2-FIXED-BENCHMARKS-V1" / split / "frozen_route.json"
        cpath = root / "outputs/SCALEROUTE-PLAN15-PROGRESSIVE40-FIXED-BENCHMARKS-V1" / split / "platform_frozen_thresholds.json"
        b, c = read(bpath), read(cpath)
        payloads = [{"platform": x["primary"]["platform"], "input_sha256": x["input_sha256"],
                     "latency_seconds": None, "diagnostic_only": True} for x in (b, c)]
        compared = compare(*payloads)
        result["historical"][split] = {
            "input_files": {str(p.relative_to(root)): sha(p) for p in (bpath, cpath)},
            "comparison": compared,
            "local_score_direction_matches_official": direction(compared["delta_quality_contribution"])
            == direction(official["delta_quality_contribution"]),
            "per_coarse_sign_matches": {
                label: {key: direction(compared["per_coarse_delta"][label][key], 1e-8)
                        == direction(official["per_coarse_delta"][label][key], 1e-8)
                        for key in ("recall_pp", "fdr_pp")}
                for label in COARSE_CLASSES
            },
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(ROOT)
    write(a.output, result)
    for name, row in result["historical"].items():
        print(name, row["comparison"]["delta_quality_contribution"], row["per_coarse_sign_matches"])
