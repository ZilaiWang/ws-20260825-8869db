#!/usr/bin/env python3
"""Review MAR20 low-background-support tails without rewriting original failures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rsdet.grouping.cache import PlaceFeatureCache
from rsdet.grouping.contracts import (
    MASKED_PATCH_PROTOCOL_VERSION,
    atomic_write_json,
    parse_node_uid,
    sha256_file,
)
from rsdet.grouping.registry import load_registry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review MAR20 low valid-patch tail")
    parser.add_argument("--extraction-summary", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--patch-audit-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quality-threshold", type=float, default=0.25)
    parser.add_argument("--very-low-threshold", type=float, default=0.10)
    parser.add_argument("--maximum-low-node-fraction", type=float, default=0.01)
    parser.add_argument("--maximum-audit-primary-low-fraction", type=float, default=0.01)
    parser.add_argument("--primary-dilation-key", default="dilation_0p15")
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    quantiles = (0.0, 0.001, 0.005, 0.01, 0.05, 0.5, 0.95, 1.0)
    return {f"p{100 * quantile:g}": float(np.quantile(values, quantile)) for quantile in quantiles}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.very_low_threshold < args.quality_threshold < 1:
        raise ValueError("quality thresholds must satisfy 0 < very-low < quality < 1")
    if not 0 <= args.maximum_low_node_fraction <= 1:
        raise ValueError("maximum-low-node-fraction outside [0,1]")
    if not 0 <= args.maximum_audit_primary_low_fraction <= 1:
        raise ValueError("maximum-audit-primary-low-fraction outside [0,1]")

    extraction_path = Path(args.extraction_summary).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    registry_path = Path(args.registry).expanduser().resolve()
    patch_audit_path = Path(args.patch_audit).expanduser().resolve()
    patch_summary_path = Path(args.patch_audit_summary).expanduser().resolve()
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    patch_summary = json.loads(patch_summary_path.read_text(encoding="utf-8"))
    node_count = int(extraction.get("config", {}).get("node_count", 0))
    registry = load_registry(registry_path, expected_rows=node_count)
    registry_by_uid = {row["node_uid"]: row for row in registry}
    if len(registry_by_uid) != len(registry):
        raise ValueError("registry contains duplicate node_uid")

    cache = PlaceFeatureCache(cache_dir)
    cache_audit = cache.audit()
    payload = cache.load_all()
    required_rows = {
        "row__node_uid",
        "row__rotation",
        "row__valid_patch_fraction",
        "row__valid_patch_count",
        "row__patch_count",
        "row__patch_mask_sha256",
    }
    if not required_rows <= set(payload):
        raise ValueError(f"cache rows missing: {sorted(required_rows - set(payload))}")
    if extraction.get("protocol_version") != MASKED_PATCH_PROTOCOL_VERSION:
        raise ValueError("extraction protocol mismatch")
    if extraction.get("status") != "fail_low_valid_patch_fraction":
        raise ValueError("this review only accepts the preserved low-valid failure state")
    if extraction.get("config", {}).get("minimum_valid_patch_fraction") != args.quality_threshold:
        raise ValueError("quality threshold does not match the preserved extraction contract")
    if extraction.get("index_sha256") != sha256_file(cache_dir / "index.json"):
        raise ValueError("cache index SHA does not match extraction summary")
    if extraction.get("cache", {}).get("fingerprint") != cache_audit.get("fingerprint"):
        raise ValueError("cache fingerprint does not match extraction summary")
    if extraction.get("config", {}).get("registry_sha256") != sha256_file(registry_path):
        raise ValueError("registry SHA does not match extraction summary")

    expected_rotations = tuple(int(value) for value in extraction["config"]["rotations"])
    expected_rows = node_count * len(expected_rotations)
    if cache_audit["row_count"] != expected_rows or cache_audit["nonfinite_count"] != 0:
        raise ValueError("cache row/nonfinite contract failed")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, uid in enumerate(payload["row__node_uid"].astype(str)):
        grouped[uid].append(index)
    if set(grouped) != set(registry_by_uid):
        raise ValueError("cache/registry node set mismatch")

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    patch_samples = int(extraction["config"]["patch_samples_per_node"])
    for uid in sorted(grouped, key=parse_node_uid):
        indices = grouped[uid]
        rotations = tuple(sorted(payload["row__rotation"][indices].astype(int).tolist()))
        if rotations != tuple(sorted(expected_rotations)):
            failures.append(f"rotation_contract:{uid}")
        fractions = payload["row__valid_patch_fraction"][indices].astype(float)
        valid_counts = payload["row__valid_patch_count"][indices].astype(int)
        patch_counts = payload["row__patch_count"][indices].astype(int)
        if np.ptp(fractions) > 1e-12 or len(set(valid_counts.tolist())) != 1:
            failures.append(f"rotation_quality_mismatch:{uid}")
        if len(set(patch_counts.tolist())) != 1:
            failures.append(f"rotation_patch_count_mismatch:{uid}")
        fraction = float(fractions[0])
        valid_count = int(valid_counts[0])
        if valid_count < patch_samples:
            failures.append(f"insufficient_patch_samples:{uid}")
        tier = (
            "very_low_background_support"
            if fraction < args.very_low_threshold
            else (
                "low_background_support"
                if fraction < args.quality_threshold
                else "normal_background_support"
            )
        )
        record = registry_by_uid[uid]
        rows.append(
            {
                "node_uid": uid,
                "quality_tier": tier,
                "valid_patch_fraction": fraction,
                "valid_patch_count": valid_count,
                "patch_count": int(patch_counts[0]),
                "patch_samples_per_node": patch_samples,
                "sample_capacity_ratio": valid_count / patch_samples,
                "is_target": record["is_target"],
                "is_bridge": record["is_bridge"],
                "official_side": record["official_side"],
                "bbox_count": record["bbox_count"],
                "fine_class_hist_json": record["fine_class_hist_json"],
                "low_support_requires_extra_evidence": int(tier != "normal_background_support"),
            }
        )
    low_rows = [row for row in rows if row["quality_tier"] != "normal_background_support"]
    very_low_rows = [row for row in rows if row["quality_tier"] == "very_low_background_support"]
    low_fraction = len(low_rows) / len(rows)
    if low_fraction > args.maximum_low_node_fraction:
        failures.append("low_node_fraction_above_review_limit")
    source_low = set(extraction.get("low_valid_patch_nodes", []))
    computed_low = {row["node_uid"] for row in low_rows}
    if source_low != computed_low:
        failures.append("source_low_node_list_mismatch")

    patch_rows = _read_csv(patch_audit_path)
    primary_field = f"{args.primary_dilation_key}_valid_patch_fraction"
    if not patch_rows or primary_field not in patch_rows[0]:
        raise ValueError(f"patch audit missing primary field {primary_field}")
    audit_primary_low = [
        row for row in patch_rows if float(row[primary_field]) < args.quality_threshold
    ]
    audit_primary_low_fraction = len(audit_primary_low) / len(patch_rows)
    if audit_primary_low_fraction > args.maximum_audit_primary_low_fraction:
        failures.append("audit_primary_low_fraction_above_review_limit")
    automatic_failures = patch_summary.get("automatic_failures", [])
    if patch_summary.get("sample_count") != len(patch_rows):
        failures.append("patch_audit_sample_count_mismatch")
    if patch_summary.get("artifacts", {}).get("patch_mask_audit.csv") != sha256_file(
        patch_audit_path
    ):
        failures.append("patch_audit_csv_sha_mismatch")
    if patch_summary.get("automatic_geometry_gate") != "fail":
        failures.append("source_patch_audit_gate_not_preserved_as_fail")
    if any("valid_patch_fraction" not in value for value in automatic_failures):
        failures.append("patch_audit_has_non_quality_geometry_failure")
    if patch_summary.get("configuration", {}).get("primary_dilation_ratio") != 0.15:
        failures.append("patch_audit_primary_dilation_mismatch")

    admitted = not failures
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_path = output / "all_node_background_support.csv"
    low_path = output / "low_background_support_nodes.csv"
    _write_csv(all_path, rows)
    _write_csv(low_path, low_rows)
    fractions = np.asarray([row["valid_patch_fraction"] for row in rows], dtype=np.float64)
    decision = {
        "status": ("accepted_for_continuation_with_low_support_flags" if admitted else "fail"),
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "original_extraction_status": extraction["status"],
        "original_patch_audit_gate": patch_summary.get("automatic_geometry_gate"),
        "continuation_admission": admitted,
        "formal_patch_mask_admission": False,
        "formal_grouping_admission": False,
        "node_count": len(rows),
        "low_background_support_count": len(low_rows),
        "very_low_background_support_count": len(very_low_rows),
        "low_background_support_fraction": low_fraction,
        "target_low_background_support_count": sum(row["is_target"] == "1" for row in low_rows),
        "bridge_low_background_support_count": sum(row["is_bridge"] == "1" for row in low_rows),
        "minimum_valid_patch_count": min(row["valid_patch_count"] for row in rows),
        "minimum_sample_capacity_ratio": min(row["sample_capacity_ratio"] for row in rows),
        "valid_patch_fraction_percentiles": _percentiles(fractions),
        "audit_primary_low_count": len(audit_primary_low),
        "audit_primary_low_fraction": audit_primary_low_fraction,
        "audit_primary_low_nodes": [row["node_uid"] for row in audit_primary_low],
        "policy": {
            "quality_threshold": args.quality_threshold,
            "very_low_threshold": args.very_low_threshold,
            "minimum_feasible_patch_count": patch_samples,
            "maximum_low_node_fraction": args.maximum_low_node_fraction,
            "maximum_audit_primary_low_fraction": args.maximum_audit_primary_low_fraction,
            "keep_original_mask_and_features": True,
            "reextract_features": False,
            "low_support_descriptor_only_edge_allowed": False,
            "low_support_fallback": "exact_or_phash_or_geometry_or_manual_else_singleton_embargo",
        },
        "failures": failures,
        "inputs": {
            "extraction_summary_sha256": sha256_file(extraction_path),
            "cache_index_sha256": sha256_file(cache_dir / "index.json"),
            "registry_sha256": sha256_file(registry_path),
            "patch_audit_sha256": sha256_file(patch_audit_path),
            "patch_audit_summary_sha256": sha256_file(patch_summary_path),
        },
        "artifacts": {
            "all_node_background_support.csv": sha256_file(all_path),
            "low_background_support_nodes.csv": sha256_file(low_path),
        },
    }
    decision_path = output / "low_valid_patch_fraction_review.json"
    atomic_write_json(decision_path, decision)

    extraction_admitted = dict(extraction)
    extraction_admitted.update(
        {
            "status": "pass" if admitted else "fail",
            "source_status": extraction["status"],
            "reviewed_continuation_status": decision["status"],
            "continuation_admission": admitted,
            "formal_grouping_admission": False,
            "quality_review_sha256": sha256_file(decision_path),
            "source_extraction_summary_sha256": sha256_file(extraction_path),
            "low_support_policy": decision["policy"],
        }
    )
    extraction_admitted_path = output / "extraction_summary_admitted.json"
    atomic_write_json(extraction_admitted_path, extraction_admitted)

    patch_admitted = dict(patch_summary)
    patch_admitted.update(
        {
            "automatic_geometry_gate": "pass" if admitted else "fail",
            "source_automatic_geometry_gate": patch_summary.get("automatic_geometry_gate"),
            "reviewed_continuation_status": decision["status"],
            "continuation_admission": admitted,
            "formal_patch_mask_admission": False,
            "formal_grouping_admission": False,
            "quality_review_sha256": sha256_file(decision_path),
            "source_patch_audit_summary_sha256": sha256_file(patch_summary_path),
        }
    )
    patch_admitted_path = output / "patch_mask_audit_summary_admitted.json"
    atomic_write_json(patch_admitted_path, patch_admitted)
    result = {
        "status": "ready_to_continue_vlad_with_quality_flags" if admitted else "blocked",
        "continuation_admission": admitted,
        "formal_grouping_admission": False,
        "review_sha256": sha256_file(decision_path),
        "extraction_summary_admitted_sha256": sha256_file(extraction_admitted_path),
        "patch_mask_audit_summary_admitted_sha256": sha256_file(patch_admitted_path),
    }
    atomic_write_json(output / "continuation_decision.json", result)
    print(json.dumps({"decision": decision, "result": result}, ensure_ascii=False, indent=2))
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
