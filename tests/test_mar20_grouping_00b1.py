"""MAR20 00B1 low-background-support review and continuation contracts."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from rsdet.grouping.cache import PlaceFeatureCache, PlaceFeatureCacheWriter
from rsdet.grouping.contracts import MASKED_PATCH_PROTOCOL_VERSION, sha256_file
from rsdet.grouping.registry import REGISTRY_SCHEMA_VERSION


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, str(root / "scripts/review_mar20_low_valid_patch_fraction.py"), *args],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _run_phase_decision(*args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, str(root / "scripts/compile_mar20_00b1_phase_a_decision.py"), *args],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _fixture(tmp_path: Path, *, minimum_count: int = 80) -> dict[str, Path]:
    registry_path = tmp_path / "registry.csv"
    registry = [
        {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "node_uid": f"mar20:{index}",
            "is_target": "1" if index < 3 else "0",
            "is_bridge": "0" if index < 3 else "1",
            "official_side": "train",
            "bbox_count": str(index),
            "fine_class_hist_json": '{"4":1}',
        }
        for index in (1, 2, 3)
    ]
    _write_csv(registry_path, registry)
    cache_dir = tmp_path / "cache"
    writer = PlaceFeatureCacheWriter(
        cache_dir,
        metadata={"protocol": MASKED_PATCH_PROTOCOL_VERSION},
        feature_names=("f",),
        storage_dtype="float32",
    )
    fractions = {"mar20:1": 0.08, "mar20:2": 0.30, "mar20:3": 0.80}
    counts = {"mar20:1": minimum_count, "mar20:2": 411, "mar20:3": 1095}
    rotations = (0, 90, 180, 270)
    node_uids = [uid for uid in fractions for _ in rotations]
    rotation_rows = [rotation for _ in fractions for rotation in rotations]
    writer.write_shard(
        0,
        rows={
            "node_uid": node_uids,
            "view_type": ["masked_patch_original_input"] * len(node_uids),
            "rotation": rotation_rows,
            "item_index": [0] * len(node_uids),
            "input_sha256": [f"{index:064x}" for index in range(len(node_uids))],
            "patch_mask_sha256": [f"{index + 100:064x}" for index in range(len(node_uids))],
            "valid_patch_fraction": [fractions[uid] for uid in fractions for _ in rotations],
            "valid_patch_count": [counts[uid] for uid in counts for _ in rotations],
            "patch_count": [1369] * len(node_uids),
        },
        features={"f": np.ones((len(node_uids), 4), dtype=np.float32)},
    )
    writer.finalize(expected_shards=1, expected_rows=12)
    audit = PlaceFeatureCache(cache_dir).audit()
    extraction_path = tmp_path / "extraction.json"
    extraction = {
        "status": "fail_low_valid_patch_fraction",
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "config": {
            "node_count": 3,
            "rotations": list(rotations),
            "patch_samples_per_node": 16,
            "minimum_valid_patch_fraction": 0.25,
            "registry_sha256": sha256_file(registry_path),
        },
        "cache": audit,
        "index_sha256": sha256_file(cache_dir / "index.json"),
        "sample_fingerprint": "a" * 64,
        "sample_shard_count": 1,
        "sampled_node_count": 3,
        "low_valid_patch_nodes": ["mar20:1"],
    }
    extraction_path.write_text(json.dumps(extraction), encoding="utf-8")
    patch_audit_path = tmp_path / "patch-audit.csv"
    _write_csv(
        patch_audit_path,
        [
            {"node_uid": "mar20:1", "dilation_0p15_valid_patch_fraction": 0.08},
            {"node_uid": "mar20:2", "dilation_0p15_valid_patch_fraction": 0.30},
        ],
    )
    patch_summary_path = tmp_path / "patch-summary.json"
    patch_summary_path.write_text(
        json.dumps(
            {
                "automatic_geometry_gate": "fail",
                "automatic_failures": ["mar20:1:dilation_0p15:valid_patch_fraction"],
                "sample_count": 2,
                "configuration": {"primary_dilation_ratio": 0.15},
                "artifacts": {"patch_mask_audit.csv": sha256_file(patch_audit_path)},
            }
        ),
        encoding="utf-8",
    )
    return {
        "registry": registry_path,
        "cache": cache_dir,
        "extraction": extraction_path,
        "patch_audit": patch_audit_path,
        "patch_summary": patch_summary_path,
    }


def test_review_preserves_failure_and_emits_explicit_admission(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    source_before = inputs["extraction"].read_bytes()
    output = tmp_path / "review"
    result = _run(
        "--extraction-summary",
        str(inputs["extraction"]),
        "--cache-dir",
        str(inputs["cache"]),
        "--registry",
        str(inputs["registry"]),
        "--patch-audit",
        str(inputs["patch_audit"]),
        "--patch-audit-summary",
        str(inputs["patch_summary"]),
        "--output-dir",
        str(output),
        "--maximum-low-node-fraction",
        "0.40",
        "--maximum-audit-primary-low-fraction",
        "0.60",
    )
    assert result.returncode == 0, result.stderr
    assert inputs["extraction"].read_bytes() == source_before
    review = json.loads((output / "low_valid_patch_fraction_review.json").read_text())
    assert review["status"] == "accepted_for_continuation_with_low_support_flags"
    assert review["original_extraction_status"] == "fail_low_valid_patch_fraction"
    assert review["low_background_support_count"] == 1
    assert review["minimum_valid_patch_count"] == 80
    admitted = json.loads((output / "extraction_summary_admitted.json").read_text())
    assert admitted["status"] == "pass"
    assert admitted["source_status"] == "fail_low_valid_patch_fraction"
    patch_admitted = json.loads((output / "patch_mask_audit_summary_admitted.json").read_text())
    assert patch_admitted["automatic_geometry_gate"] == "pass"
    assert patch_admitted["source_automatic_geometry_gate"] == "fail"


def test_review_blocks_when_patch_sampling_is_not_feasible(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path, minimum_count=8)
    output = tmp_path / "review"
    result = _run(
        "--extraction-summary",
        str(inputs["extraction"]),
        "--cache-dir",
        str(inputs["cache"]),
        "--registry",
        str(inputs["registry"]),
        "--patch-audit",
        str(inputs["patch_audit"]),
        "--patch-audit-summary",
        str(inputs["patch_summary"]),
        "--output-dir",
        str(output),
        "--maximum-low-node-fraction",
        "0.40",
        "--maximum-audit-primary-low-fraction",
        "0.60",
    )
    assert result.returncode == 2
    review = json.loads((output / "low_valid_patch_fraction_review.json").read_text())
    assert "insufficient_patch_samples:mar20:1" in review["failures"]
    assert review["continuation_admission"] is False


def test_phase_a_decision_requires_every_continuation_artifact(tmp_path: Path) -> None:
    values = {
        "continuation": {"continuation_admission": True},
        "quality": {
            "status": "accepted_for_continuation_with_low_support_flags",
            "low_background_support_count": 19,
        },
        "codebook": {
            "status": "pass",
            "entries": [{"input_token_count": 61_472} for _ in range(6)],
        },
        "vlad": {
            "status": "pass",
            "cache": {"row_count": 15_368, "nonfinite_count": 0},
            "actual_shards": 241,
        },
        "projection": {
            "status": "pass",
            "row_count": 15_368,
            "cache": {"nonfinite_count": 0},
            "pca_entries": [{} for _ in range(6)],
        },
        "candidate": {
            "status": "pass",
            "geometry_scored_count": 1_600,
            "formal_edge_admission": False,
        },
        "review": {
            "status": "waiting_for_blind_manual_review",
            "unique_pair_count": 240,
            "card_count": 264,
            "blind_duplicate_count": 24,
            "target_target_count": 180,
            "formal_descriptor_admission": False,
        },
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    output = tmp_path / "decision.json"
    result = _run_phase_decision(
        "--continuation-decision",
        str(paths["continuation"]),
        "--quality-review",
        str(paths["quality"]),
        "--codebook-manifest",
        str(paths["codebook"]),
        "--vlad-summary",
        str(paths["vlad"]),
        "--projection-summary",
        str(paths["projection"]),
        "--candidate-summary",
        str(paths["candidate"]),
        "--enriched-review-summary",
        str(paths["review"]),
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    decision = json.loads(output.read_text())
    assert decision["status"] == "waiting_for_patch_mask_and_enriched_pair_reviews"
    assert decision["formal_grouping_admission"] is False
    assert decision["task01_retrieval_admission"] is False
