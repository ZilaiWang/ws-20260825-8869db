#!/usr/bin/env python3
"""Compile the terminal scientific state for MAR20-GROUPING-TASK-00B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rsdet.grouping.contracts import MASKED_PATCH_PROTOCOL_VERSION, atomic_write_json, sha256_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile MAR20 00B decision")
    parser.add_argument("--patch-mask-decision", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--round-b-decision", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        "patch_mask": Path(args.patch_mask_decision).expanduser().resolve(),
        "calibration": Path(args.calibration_summary).expanduser().resolve(),
        "round_b": Path(args.round_b_decision).expanduser().resolve(),
    }
    values = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    patch_pass = bool(values["patch_mask"].get("formal_patch_mask_admission"))
    calibration_pass = bool(values["calibration"].get("formal_threshold_admission"))
    calibration_status = str(values["calibration"].get("status", ""))
    round_b_pass = bool(values["round_b"].get("formal_descriptor_selection_admission"))
    positive_count = int(values["calibration"].get("positive_pair_count", 0))
    enriched_positive_count = int(
        values["calibration"].get("positive_pair_count_by_batch", {}).get("00b_enriched", 0)
    )
    recommended = bool(values["calibration"].get("recommended_evidence_target_met"))
    if not patch_pass:
        status = "complete_00b_patch_mask_no_admission"
        next_action = "conservative_core_or_one_explicit_mask_protocol_revision"
    elif calibration_status == "fail":
        status = "complete_00b_calibration_no_admission"
        next_action = "resolve_manual_or_evidence_conflicts_before_any_new_experiment"
    elif not calibration_pass:
        status = (
            "needs_second_enrichment_batch"
            if enriched_positive_count < 20
            else "complete_00b_calibration_no_admission"
        )
        next_action = (
            "one_additional_positive_enriched_review_batch_then_stop"
            if enriched_positive_count < 20
            else "build_conservative_core_and_fold_specific_embargo"
        )
    elif not round_b_pass:
        status = "complete_00b_retrieval_no_admission"
        next_action = "build_conservative_core_and_fold_specific_embargo"
    else:
        status = "ready_for_task01_retrieval_and_geometry"
        next_action = "run_formal_full_retrieval_and_geometric_verification"
    result = {
        "status": status,
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "formal_grouping_admission": False,
        "task01_retrieval_admission": patch_pass and calibration_pass and round_b_pass,
        "patch_mask_admission": patch_pass,
        "calibration_admission": calibration_pass,
        "recommended_evidence_target_met": recommended,
        "round_b_descriptor_admission": round_b_pass,
        "selected_routes": values["round_b"].get("selected_routes", []),
        "positive_pair_count": positive_count,
        "enriched_positive_pair_count": enriched_positive_count,
        "next_action": next_action,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()
        },
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
