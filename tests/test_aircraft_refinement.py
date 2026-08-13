"""Tests for the R1-1 aircraft-only refinement contract."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from rsdet.analysis.aircraft_refinement import (
    AIRCRAFT_CLASS_IDS,
    adaptive_d4_probabilities,
    aircraft_conditional_probabilities,
    audit_aircraft_training_rows,
    build_aircraft_variants,
    load_aircraft_bundle,
    load_aircraft_training_rows,
)
from rsdet.analysis.proposal_reranking import ProposalRecord, apply_variant


def _record(uid: str, category: int, fold: int = 0) -> ProposalRecord:
    return ProposalRecord(
        proposal_uid=uid,
        image_id=fold + 1,
        fold=fold,
        relative_path=f"images/{fold}.jpg",
        category_id=category,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        detector_score=0.7,
        source_prediction_index=0,
    )


def test_aircraft_variant_structurally_bypasses_ship_and_vehicle() -> None:
    records = [_record("ship", 0), _record("aircraft", 4), _record("vehicle", 24)]
    probabilities = {uid: np.zeros(25) for uid in ("ship", "aircraft", "vehicle")}
    probabilities["aircraft"][7] = 1.0
    variant = build_aircraft_variants(
        {
            "min_top_probabilities": [0.5],
            "min_margins": [0.1],
            "fusion_alphas": [0.2],
            "gated_fusion_alpha": 0.2,
        }
    )[1]
    predictions, audit = apply_variant(records, probabilities, variant, image_ids=[1])
    assert [item["category_id"] for item in predictions[1]] == [0, 7, 24]
    assert [item["score"] for item in predictions[1]] == [0.7, 0.7, 0.7]
    assert audit == {"relabelled": 1, "gated_out": 0, "bypassed": 2}


def test_aircraft_probabilities_are_conditional_and_zero_outside_route() -> None:
    logits = np.full(25, -100.0, dtype=np.float32)
    logits[0] = 100.0  # global ship evidence must not change p(class|aircraft)
    logits[4] = 2.0
    logits[5] = 1.0
    output = aircraft_conditional_probabilities({"p": logits})["p"]
    assert output.shape == (25,)
    assert output[list(AIRCRAFT_CLASS_IDS)].sum() == pytest.approx(1.0)
    assert output[:4].sum() == 0.0
    assert output[24] == 0.0
    assert output[4] > output[5]


def test_training_manifest_filters_cross_coarse_oracle_rows(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    fields = [
        "proposal_uid",
        "source_relative_path",
        "fold",
        "class_id",
        "detector_category_id",
        "crop_xyxy",
        "view",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "proposal_uid": "keep",
                "source_relative_path": "a.jpg",
                "fold": 0,
                "class_id": 4,
                "detector_category_id": 7,
                "crop_xyxy": "0,0,10,10",
                "view": "deployable_positive",
            }
        )
        writer.writerow(
            {
                "proposal_uid": "drop",
                "source_relative_path": "b.jpg",
                "fold": 1,
                "class_id": 4,
                "detector_category_id": 0,
                "crop_xyxy": "0,0,10,10",
                "view": "oracle_positive",
            }
        )
    rows = load_aircraft_training_rows(path)
    assert [row["proposal_uid"] for row in rows] == ["keep"]


def test_training_audit_requires_all_folds_and_all_aircraft_classes() -> None:
    rows = []
    for offset, class_id in enumerate(AIRCRAFT_CLASS_IDS):
        rows.append(
            {
                "proposal_uid": f"p{offset}",
                "fold": str(offset % 3),
                "class_id": str(class_id),
                "view": "deployable_positive",
            }
        )
    audit = audit_aircraft_training_rows(rows)
    assert audit["row_count"] == 20
    assert set(audit["fold_counts"]) == {0, 1, 2}


def test_bundle_requires_exact_aircraft_uid_coverage(tmp_path: Path) -> None:
    records = [_record(f"p{fold}", 4, fold) for fold in (0, 1, 2)]
    for fold in (0, 1, 2):
        values = np.zeros((1, 20), dtype=np.float32)
        values[:, 0] = 1.0
        np.savez_compressed(
            tmp_path / f"fold_{fold}_aircraft_bundle.npz",
            proposal_uids=np.asarray([f"p{fold}"], dtype="<U2"),
            identity_logits=np.zeros((1, 25), dtype=np.float32),
            d4_probabilities=values,
        )
    identity, d4 = load_aircraft_bundle(tmp_path, records)
    assert set(identity) == {"p0", "p1", "p2"}
    assert d4["p0"][4] == pytest.approx(1.0)


def test_adaptive_d4_only_expands_low_confidence_aircraft(tmp_path: Path) -> None:
    records = [_record(f"p{fold}", 4, fold) for fold in (0, 1, 2)]
    base_dir = tmp_path / "base"
    bundle_dir = tmp_path / "bundle"
    base_dir.mkdir()
    bundle_dir.mkdir()
    for fold in (0, 1, 2):
        all_uids = np.asarray([f"p{item}" for item in (0, 1, 2)], dtype="<U2")
        all_logits = np.zeros((3, 25), dtype=np.float32)
        np.savez_compressed(
            base_dir / f"fold_{fold}_logits.npz",
            proposal_uids=all_uids[fold : fold + 1],
            logits=all_logits[fold : fold + 1],
        )
        identity = np.zeros((1, 25), dtype=np.float32)
        identity[0, 4] = 8.0 if fold == 0 else 0.1
        d4 = np.zeros((1, 20), dtype=np.float32)
        d4[0, 1] = 1.0
        np.savez_compressed(
            bundle_dir / f"fold_{fold}_aircraft_bundle.npz",
            proposal_uids=np.asarray([f"p{fold}"], dtype="<U2"),
            identity_logits=identity,
            d4_probabilities=d4,
        )
    probabilities, audit = adaptive_d4_probabilities(
        records,
        base_dir,
        bundle_dir,
        maximum_identity_probability=0.8,
    )
    assert probabilities["p0"].argmax() == 4
    assert probabilities["p1"].argmax() == 5
    assert probabilities["p2"].argmax() == 5
    assert audit["d4_proposal_count"] == 2
    assert audit["view_compute_ratio_vs_full_d4"] == pytest.approx(17 / 24)
