#!/usr/bin/env python3
"""Compile the audited waiting state after the MAR20 00B1 Phase-A continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rsdet.grouping.contracts import atomic_write_json, sha256_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile MAR20 00B1 Phase-A waiting state")
    parser.add_argument("--continuation-decision", required=True)
    parser.add_argument("--quality-review", required=True)
    parser.add_argument("--codebook-manifest", required=True)
    parser.add_argument("--vlad-summary", required=True)
    parser.add_argument("--projection-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--enriched-review-summary", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _load(path: str) -> tuple[Path, dict]:
    value = Path(path).expanduser().resolve()
    return value, json.loads(value.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources = {
        "continuation": _load(args.continuation_decision),
        "quality_review": _load(args.quality_review),
        "codebook": _load(args.codebook_manifest),
        "vlad": _load(args.vlad_summary),
        "projection": _load(args.projection_summary),
        "candidate": _load(args.candidate_summary),
        "enriched_review": _load(args.enriched_review_summary),
    }
    values = {name: payload for name, (_, payload) in sources.items()}
    failures = []
    if not values["continuation"].get("continuation_admission"):
        failures.append("continuation_not_admitted")
    if values["quality_review"].get("status") != (
        "accepted_for_continuation_with_low_support_flags"
    ):
        failures.append("quality_review_not_accepted")
    if values["quality_review"].get("low_background_support_count") != 19:
        failures.append("quality_review_low_count_changed")
    entries = values["codebook"].get("entries", [])
    if values["codebook"].get("status") != "pass" or len(entries) != 6:
        failures.append("codebook_contract_failed")
    if any(int(entry.get("input_token_count", -1)) != 61_472 for entry in entries):
        failures.append("codebook_token_count_changed")
    vlad_cache = values["vlad"].get("cache", {})
    if (
        values["vlad"].get("status") != "pass"
        or vlad_cache.get("row_count") != 15_368
        or vlad_cache.get("nonfinite_count") != 0
        or values["vlad"].get("actual_shards") != 241
    ):
        failures.append("vlad_cache_contract_failed")
    projection_cache = values["projection"].get("cache", {})
    if (
        values["projection"].get("status") != "pass"
        or values["projection"].get("row_count") != 15_368
        or projection_cache.get("nonfinite_count") != 0
        or len(values["projection"].get("pca_entries", [])) != 6
    ):
        failures.append("projection_contract_failed")
    if (
        values["candidate"].get("status") != "pass"
        or values["candidate"].get("geometry_scored_count") != 1_600
        or values["candidate"].get("formal_edge_admission") is not False
    ):
        failures.append("candidate_contract_failed")
    review = values["enriched_review"]
    if (
        review.get("status") != "waiting_for_blind_manual_review"
        or review.get("unique_pair_count") != 240
        or review.get("card_count") != 264
        or review.get("blind_duplicate_count") != 24
        or int(review.get("target_target_count", 0)) < 180
        or review.get("formal_descriptor_admission") is not False
    ):
        failures.append("enriched_review_contract_failed")
    result = {
        "status": (
            "waiting_for_patch_mask_and_enriched_pair_reviews" if not failures else "blocked"
        ),
        "protocol_version": "mar20-source-grouping-v1.2-00b1-quality-amendment",
        "source_status": "blocked_low_valid_patch_fraction",
        "low_valid_review_status": values["quality_review"].get("status"),
        "formal_grouping_admission": False,
        "task01_retrieval_admission": False,
        "required_manual_inputs": [
            "manual_patch_mask_review.csv",
            "manual_enriched_decisions.csv",
        ],
        "failures": failures,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, (path, _) in sources.items()
        },
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
