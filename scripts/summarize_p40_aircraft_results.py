#!/usr/bin/env python3
"""Produce a small, reproducible CE transfer/risk summary from immutable runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.experiments.fixed_proxy import quality_contribution

CASES = {
    "d4_hard": "P40-AIRCRAFT-CE-D4-V1",
    "d4_sentinel": "P40-AIRCRAFT-CE-D4-V1/sentinel",
    "d4_normal_diagnostic": "P40-AIRCRAFT-CE-D4-NORMAL-V1",
    "identity_hard": "P40-AIRCRAFT-CE-IDENTITY-V1/hard",
    "identity_sentinel": "P40-AIRCRAFT-CE-IDENTITY-V1/sentinel",
    "consistency_hard": "P40-AIRCRAFT-VIEW-CONSISTENCY-V1/hard",
    "consistency_sentinel": "P40-AIRCRAFT-VIEW-CONSISTENCY-V1/sentinel",
}


def summarize(result):
    control, candidate = result["nms_only_control"], result["candidate"]
    b, c = control["platform"], candidate["platform"]
    fine = []
    for category in range(4, 24):
        before, after = (part["per_fine"][str(category)] for part in (control, candidate))
        if before["tp"] + before["fn"] != after["tp"] + after["fn"]:
            raise ValueError("changed ground-truth support")
        fine.append(
            dict(
                category_id=category,
                name=FINE_NAMES[category],
                gt=before["tp"] + before["fn"],
                before=before,
                after=after,
                delta_tp=after["tp"] - before["tp"],
                delta_fp=after["fp"] - before["fp"],
            )
        )
    for route in ("ship", "vehicle"):
        if b["per_coarse"][route] != c["per_coarse"][route]:
            raise ValueError("bypass route metric changed")
    return {
        "delta_quality_vs_nms_only": quality_contribution(c) - quality_contribution(b),
        "delta_quality_vs_original": quality_contribution(c)
        - quality_contribution(result["original_baseline"]["platform"]),
        "before": b,
        "after": c,
        "aircraft_counts_before": control["pooled_diagnostic"]["per_class"]["aircraft"],
        "aircraft_counts_after": candidate["pooled_diagnostic"]["per_class"]["aircraft"],
        "aircraft_proposals": result["aircraft_proposals"],
        "changed_fine_labels": result["changed_fine_labels"],
        "fine": fine,
        "fine_with_tp_loss": [row["name"] for row in fine if row["delta_tp"] < 0],
        "fine_with_fp_increase": [row["name"] for row in fine if row["delta_fp"] > 0],
        "load_decode_classify_seconds_total": result["elapsed_load_decode_classify_seconds"],
        "review": result["review"],
        "not_official_prediction": True,
        "formal_admission": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    combined, hashes = {}, {}
    cases = dict(CASES)
    if (args.root / "P40-AIRCRAFT-CONSISTENCY-NORMAL-V1/comparison.json").exists():
        cases["consistency_normal_diagnostic"] = "P40-AIRCRAFT-CONSISTENCY-NORMAL-V1"
    for name, relative in cases.items():
        path = args.root / relative / "comparison.json"
        result = json.loads(path.read_text())
        if result["status"] != "complete":
            raise ValueError(f"incomplete result: {relative}")
        combined[name] = summarize(result)
        hashes[relative + "/comparison.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output / f"{name}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    payload = dict(
        status="complete",
        metric_protocol="platform_observed_20260831",
        cases=combined,
        source_sha256=hashes,
        formal_admission=False,
        normal_is_diagnostic_only=True,
    )
    runtime_path = args.root / "P40-AIRCRAFT-RUNTIME-V1/runtime.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        if runtime["status"] != "complete" or not runtime["all_cached_decisions_exact"]:
            raise ValueError("runtime replay is incomplete or mismatched")
        payload["ce_runtime"] = {
            key: runtime[key]
            for key in (
                "summary",
                "scope",
                "all_cached_decisions_exact",
                "environment",
                "not_docker_or_full_pipeline_latency",
                "filesystem_cache_may_be_warm",
            )
        }
        payload["runtime_source_sha256"] = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    (args.output / "aircraft_analysis.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                name: {
                    key: value[key]
                    for key in (
                        "delta_quality_vs_nms_only",
                        "fine_with_tp_loss",
                        "fine_with_fp_increase",
                    )
                }
                for name, value in combined.items()
            }
        )
    )


if __name__ == "__main__":
    main()
