"""Focused contracts for MAR20 TASK-01 retrieval and local evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rsdet.grouping.geometry import (
    SiftFeatures,
    _transform_metrics,
    patch_pair_evidence,
    pool_masked_patch_tokens,
    sift_pair_evidence,
)
from rsdet.grouping.retrieval import RouteFeatures, directional_topk, union_recall


def test_directional_topk_excludes_self_and_tracks_best_rotations() -> None:
    base = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    values = np.stack([np.stack([row, row, row, row]) for row in base])
    values /= np.linalg.norm(values, axis=2, keepdims=True)
    route = RouteFeatures(
        name="fixture",
        nodes=("mar20:1", "mar20:2", "mar20:3"),
        rotations=(0, 90, 180, 270),
        values=values,
        cache_fingerprint="f" * 64,
        feature_name="fixture",
    )
    result = directional_topk(route, k=1, device="cpu", batch_size=2)
    assert result["mar20:1"][0].node_uid == "mar20:2"
    assert result["mar20:2"][0].node_uid == "mar20:1"
    assert all(query != values[0].node_uid for query, values in result.items())
    hits, total = union_recall([result], [("mar20:1", "mar20:2")], k=1)
    assert (hits, total) == (1, 1)


def test_mask_aware_patch_pooling_never_uses_invalid_tokens() -> None:
    tokens = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    valid = np.asarray([True, False, True, True])
    pooled, pooled_valid = pool_masked_patch_tokens(
        tokens,
        valid,
        input_grid=2,
        output_grid=1,
        minimum_valid_fraction=0.5,
    )
    expected = tokens[valid].mean(axis=0)
    expected /= np.linalg.norm(expected)
    assert pooled_valid.tolist() == [True]
    assert np.allclose(pooled[0], expected)


def test_patch_pair_evidence_recovers_identical_spatial_tokens() -> None:
    tokens = np.eye(4, dtype=np.float32)
    valid = np.ones(4, dtype=bool)
    result = patch_pair_evidence(tokens, valid, tokens, valid, grid=2)
    assert result["patch_mutual_count"] == 4
    assert result["patch_matches_ge_0p9"] == 4
    assert result["patch_similarity_median"] == 1.0
    assert result["patch_grid_occupancy_u"] == 4


def test_sift_pair_evidence_uses_mutual_matches_and_stable_similarity() -> None:
    pytest.importorskip("cv2")
    points = np.asarray([(x * 20.0, y * 20.0) for y in range(4) for x in range(4)])
    shifted = points + np.asarray([7.0, 11.0], dtype=np.float32)
    descriptors = np.zeros((16, 128), dtype=np.float32)
    descriptors[np.arange(16), np.arange(16)] = 1.0
    left = SiftFeatures(points, descriptors, (100, 100))
    right = SiftFeatures(shifted, descriptors.copy(), (100, 100))
    result = sift_pair_evidence(left, right, repeat_count=3, seed=5)
    assert result["sift_mutual_ratio_matches"] == 16
    assert result["similarity_inliers"] == 16
    assert result["similarity_median_error"] < 1e-3
    assert result["sift_grid_occupancy_u"] >= 9
    assert result["sift_ransac_repeat_pass_rate"] == 1.0


def test_geometry_sanitizer_converts_degenerate_fit_to_missing_evidence(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "pair_evidence.csv"
    fieldnames = [
        "pair_uid",
        "similarity_matrix",
        "similarity_inliers",
        "similarity_median_error",
        "similarity_p95_error",
        "affine_matrix",
        "affine_inliers",
        "affine_median_error",
        "affine_p95_error",
        "homography_matrix",
        "homography_inliers",
        "homography_median_error",
        "homography_p95_error",
    ]
    rows = [
        {
            "pair_uid": "pair-1",
            "similarity_matrix": "1;0;2;0;1;3",
            "similarity_inliers": "8",
            "similarity_median_error": "0.2",
            "similarity_p95_error": "0.5",
            "affine_matrix": "1;0;2;0;1;3",
            "affine_inliers": "8",
            "affine_median_error": "nan",
            "affine_p95_error": "inf",
            "homography_matrix": "1;0;0;0;1;0;0;0;1",
            "homography_inliers": "8",
            "homography_median_error": "0.3",
            "homography_p95_error": "0.8",
        },
        {
            "pair_uid": "pair-2",
            "similarity_matrix": "1;0;0;0;1;0",
            "similarity_inliers": "4",
            "similarity_median_error": "0.4",
            "similarity_p95_error": "0.7",
            "affine_matrix": "1;0;0;0;1;0",
            "affine_inliers": "4",
            "affine_median_error": "0.4",
            "affine_p95_error": "0.7",
            "homography_matrix": "1;0;0;0;1;0;0;0;1",
            "homography_inliers": "4",
            "homography_median_error": "0.4",
            "homography_p95_error": "0.7",
        },
    ]
    with raw.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    expected_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    output = tmp_path / "sanitized"
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/sanitize_mar20_task01_geometry.py"),
            "--raw-evidence",
            str(raw),
            "--expected-raw-sha256",
            expected_sha,
            "--output-dir",
            str(output),
            "--expected-rows",
            "2",
            "--expected-nonfinite-fields",
            "2",
            "--expected-affected-pairs",
            "1",
        ],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    with (output / "pair_evidence_sanitized.csv").open(
        "r", encoding="utf-8", newline=""
    ) as file:
        sanitized = list(csv.DictReader(file))
    assert sanitized[0]["affine_median_error"] == ""
    assert sanitized[0]["affine_p95_error"] == ""
    assert sanitized[0]["affine_fit_valid"] == "0"
    assert sanitized[0]["similarity_fit_valid"] == "1"
    assert sanitized[0]["homography_fit_valid"] == "1"
    assert sanitized[1]["affine_fit_valid"] == "1"
    summary = json.loads(
        (output / "geometry_sanitization_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "pass"
    assert summary["raw_nonfinite_field_count"] == 2
    assert summary["affected_pair_count"] == 1
    assert summary["remaining_nonfinite_count"] == 0
    assert summary["pair_evidence_sha256"] == summary["sanitized_evidence_sha256"]


def test_transform_metrics_rejects_nonfinite_opencv_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cv2 = SimpleNamespace(
        RANSAC=8,
        setRNGSeed=lambda _seed: None,
        estimateAffine2D=lambda *_args, **_kwargs: (
            np.asarray([[1.0, 0.0, np.inf], [0.0, 1.0, 0.0]], dtype=np.float64),
            np.ones((4, 1), dtype=np.uint8),
        ),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    points = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    result = _transform_metrics(points, points, model="affine", threshold=3.0, seed=1)
    assert result["affine_inliers"] == 0
    assert result["affine_median_error"] is None
    assert result["affine_p95_error"] is None
    assert result["affine_matrix"] == ""
