"""MAR20 v1.2 feature-level masking, VLAD, and enriched calibration contracts."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rsdet.grouping.contracts import MASKED_PATCH_PROTOCOL_VERSION
from rsdet.grouping.masks import (
    build_protocol_foreground_mask,
    patch_validity_from_mask,
    render_masked_patch_inputs,
)
from rsdet.grouping.vlad import (
    LocalPcaVladCodebook,
    apply_global_pca,
    fit_global_pca,
    fit_local_pca_vlad,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, str(root / "scripts" / script), *args],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_v1p2_protocol_is_separate_from_round_a() -> None:
    assert MASKED_PATCH_PROTOCOL_VERSION == "mar20-source-grouping-v1.2"


def test_protocol_mask_uses_clipped_isotropic_padding() -> None:
    mask = np.asarray(
        build_protocol_foreground_mask(
            (100, 100),
            [(40, 40, 50, 50)],
            dilation_ratio=0.15,
        )
    )
    # 1.5 px is raised to the frozen 8 px minimum: [32,58) in both axes.
    ys, xs = np.where(mask > 0)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (32, 57, 32, 57)


def test_patch_validity_uses_area_threshold_and_row_major_order() -> None:
    mask = Image.new("L", (28, 28), 0)
    array = np.asarray(mask).copy()
    array[:14, :14] = 255
    valid, fractions = patch_validity_from_mask(
        Image.fromarray(array), input_size=28, patch_size=14, maximum_foreground_fraction=0.20
    )
    assert valid.tolist() == [False, True, True, True]
    assert fractions.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_masked_patch_rotation_preserves_valid_count() -> None:
    image = Image.new("RGB", (96, 64), (90, 100, 110))
    items = render_masked_patch_inputs(
        node_uid="mar20:1",
        image=image,
        boxes=[(30, 20, 50, 40)],
        rotations=(0, 90, 180, 270),
        input_size=56,
        patch_size=14,
        dilation_ratio=0.15,
    )
    assert len(items) == 4
    assert {item.valid_patch_count for item in items}.__len__() == 1
    assert all(item.patch_count == 16 for item in items)
    assert len({item.patch_mask_sha256 for item in items}) >= 2


def test_vlad_codebook_roundtrip_and_finite_encoding() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(4)
    tokens = rng.normal(size=(240, 12)).astype(np.float32)
    codebook = fit_local_pca_vlad(tokens, layer=9, local_dimension=6, cluster_count=4, seed=11)
    # Keep the Euclidean MiniBatchKMeans centres; forcing unit norms would
    # change the learned assignments and residual geometry.
    assert not np.allclose(np.linalg.norm(codebook.centers, axis=1), 1.0)
    encoded = codebook.encode(tokens[:30])
    assert encoded.shape == (24,)
    assert np.isfinite(encoded).all()
    assert np.isclose(np.linalg.norm(encoded), 1.0, atol=1e-5)
    restored = LocalPcaVladCodebook.from_payload(codebook.to_payload())
    assert np.allclose(encoded, restored.encode(tokens[:30]))


def test_vlad_uses_euclidean_not_cosine_assignment() -> None:
    codebook = LocalPcaVladCodebook(
        layer=9,
        local_input_dimension=2,
        local_dimension=2,
        cluster_count=2,
        pca_mean=np.zeros(2, dtype=np.float32),
        pca_components=np.eye(2, dtype=np.float32),
        centers=np.asarray([[0.2, 0.0], [0.9, 0.5]], dtype=np.float32),
    )
    # [1,0] is closer in Euclidean distance to centre 1, although cosine
    # similarity would prefer centre 0. Only the second residual block may live.
    encoded = codebook.encode(np.asarray([[1.0, 0.0]], dtype=np.float32))
    assert np.allclose(encoded[:2], 0.0)
    assert np.linalg.norm(encoded[2:]) > 0.99


def test_global_pca_projection_is_normalized() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(8)
    values = rng.normal(size=(80, 24)).astype(np.float32)
    mean, components = fit_global_pca(values, output_dimension=8, seed=9)
    projected = apply_global_pca(values, mean, components)
    assert projected.shape == (80, 8)
    assert np.allclose(np.linalg.norm(projected, axis=1), 1.0, atol=1e-5)


def test_patch_mask_review_compiler_passes_complete_review(tmp_path: Path) -> None:
    audit = [{"node_uid": "mar20:1"}, {"node_uid": "mar20:2"}]
    review = [
        {
            "node_uid": uid,
            "valid": 1,
            "dilation_0p10_aircraft_covered": 1,
            "dilation_0p15_aircraft_covered": 1,
            "dilation_0p20_aircraft_covered": 1,
            "dilation_0p15_excessive_background_loss": 0,
            "notes": "",
        }
        for uid in ("mar20:1", "mar20:2")
    ]
    audit_path, review_path = tmp_path / "audit.csv", tmp_path / "review.csv"
    summary_path, output_path = tmp_path / "summary.json", tmp_path / "decision.json"
    _write_csv(audit_path, audit)
    _write_csv(review_path, review)
    summary_path.write_text(json.dumps({"automatic_geometry_gate": "pass"}), encoding="utf-8")
    result = _run(
        "compile_mar20_patch_mask_review.py",
        "--audit",
        str(audit_path),
        "--audit-summary",
        str(summary_path),
        "--review",
        str(review_path),
        "--output",
        str(output_path),
    )
    assert result.returncode == 0, result.stderr
    decision = json.loads(output_path.read_text(encoding="utf-8"))
    assert decision["formal_patch_mask_admission"] is True


def test_enriched_compiler_keeps_positive_components_in_one_split(tmp_path: Path) -> None:
    prior = [
        {
            "pair_uid": "mar20:1--mar20:2",
            "node_u": "mar20:1",
            "node_v": "mar20:2",
            "label": "same_local_site",
            "binary_role": "positive",
            "split": "calibration",
            "confidence": 0.9,
            "route": "prior",
            "supporting_evidence": "",
            "counter_evidence": "",
            "notes": "",
        }
    ]
    mapping = [
        {
            "card_id": "ENR-0001",
            "pair_uid": "mar20:2--mar20:3",
            "node_u": "mar20:2",
            "node_v": "mar20:3",
            "route": "dino+sift",
            "duplicate_group": "mar20:2--mar20:3",
            "swapped": 0,
            "target_relation": "target_target",
            "cross_official_side": 1,
            "sift_inliers": 30,
            "sift_inlier_ratio": 0.5,
            "sift_coverage_u": 0.2,
            "sift_coverage_v": 0.2,
        },
        {
            "card_id": "ENR-0002",
            "pair_uid": "mar20:2--mar20:3",
            "node_u": "mar20:3",
            "node_v": "mar20:2",
            "route": "dino+sift",
            "duplicate_group": "mar20:2--mar20:3",
            "swapped": 1,
            "target_relation": "target_target",
            "cross_official_side": 1,
            "sift_inliers": 30,
            "sift_inlier_ratio": 0.5,
            "sift_coverage_u": 0.2,
            "sift_coverage_v": 0.2,
        },
        {
            "card_id": "ENR-0003",
            "pair_uid": "mar20:4--mar20:5",
            "node_u": "mar20:4",
            "node_v": "mar20:5",
            "route": "dino+sift",
            "duplicate_group": "mar20:4--mar20:5",
            "swapped": 0,
            "target_relation": "target_target",
            "cross_official_side": 0,
            "sift_inliers": 5,
            "sift_inlier_ratio": 0.1,
            "sift_coverage_u": 0.01,
            "sift_coverage_v": 0.01,
        },
    ]
    decisions = [
        {
            "card_id": row["card_id"],
            "label": "same_local_site"
            if row["pair_uid"] == "mar20:2--mar20:3"
            else "not_same_local_site",
            "confidence": 0.9,
            "supporting_evidence": "geometry",
            "counter_evidence": "",
            "notes": "",
        }
        for row in mapping
    ]
    prior_path, mapping_path, decision_path = (
        tmp_path / "prior.csv",
        tmp_path / "mapping.csv",
        tmp_path / "decisions.csv",
    )
    _write_csv(prior_path, prior)
    _write_csv(mapping_path, mapping)
    _write_csv(decision_path, decisions)
    output = tmp_path / "compiled"
    result = _run(
        "compile_mar20_enriched_calibration.py",
        "--prior-pairs",
        str(prior_path),
        "--mapping",
        str(mapping_path),
        "--decisions",
        str(decision_path),
        "--output-dir",
        str(output),
        "--minimum-positive-pairs",
        "2",
        "--recommended-positive-pairs",
        "2",
        "--minimum-heldout-positive-pairs",
        "1",
        "--recommended-heldout-positive-pairs",
        "1",
    )
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader((output / "calibration_pairs_v1p2.csv").open(encoding="utf-8")))
    component_rows = [row for row in rows if row["binary_role"] == "positive"]
    assert len({row["strict_component_id"] for row in component_rows}) == 1
    assert len({row["split"] for row in component_rows}) == 1
    summary = json.loads(
        (output / "calibration_compile_summary_v1p2.json").read_text(encoding="utf-8")
    )
    assert summary["repeat_agreement"] == 1.0
    assert summary["strict_component_cross_split_count"] == 0
