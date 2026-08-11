"""Unit tests for the R1-0 proposal reranking contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from rsdet.analysis.proposal_reranking import (
    ProposalRecord,
    RerankVariant,
    apply_variant,
    build_variants,
    load_logits,
    prepare_proposal_manifest,
)


def _record(uid: str, category_id: int = 4, fold: int = 0) -> ProposalRecord:
    return ProposalRecord(
        proposal_uid=uid,
        image_id=fold + 1,
        fold=fold,
        relative_path=f"images/{fold}.jpg",
        category_id=category_id,
        bbox_xyxy=(1.0, 2.0, 11.0, 12.0),
        detector_score=0.8,
        source_prediction_index=0,
    )


def test_variant_grid_is_compact_and_deterministic() -> None:
    config = {
        "min_top_probabilities": [0.75, 0.90],
        "min_margins": [0.15, 0.35],
        "fusion_alphas": [0.20, 0.40],
        "gated_fusion_alpha": 0.30,
    }
    first = build_variants(config)
    second = build_variants(config)
    assert first == second
    assert len(first) == 12
    assert first[0].name == "D0_detector"


def test_same_coarse_relabel_never_crosses_major_class() -> None:
    record = _record("p0", category_id=4)
    probability = np.zeros(25, dtype=np.float64)
    probability[0] = 0.99  # strongest global prediction is ship and must be ignored
    probability[7] = 0.80
    probability[4] = 0.05
    predictions, audit = apply_variant(
        [record],
        {"p0": probability},
        RerankVariant("hard", relabel=True),
        image_ids=[1],
    )
    assert predictions[1][0]["category_id"] == 7
    assert audit["relabelled"] == 1


def test_gate_preserves_detector_label_when_margin_is_small() -> None:
    record = _record("p0", category_id=4)
    probability = np.zeros(25, dtype=np.float64)
    probability[7] = 0.51
    probability[4] = 0.49
    predictions, audit = apply_variant(
        [record],
        {"p0": probability},
        RerankVariant("gate", relabel=True, min_top_probability=0.5, min_margin=0.1),
        image_ids=[1],
    )
    assert predictions[1][0]["category_id"] == 4
    assert audit["gated_out"] == 1


def test_fusion_reduces_score_when_crop_disagrees() -> None:
    record = _record("p0", category_id=4)
    probability = np.full(25, 0.04, dtype=np.float64)
    predictions, _ = apply_variant(
        [record],
        {"p0": probability},
        RerankVariant("fusion", relabel=False, fusion_alpha=0.5),
        image_ids=[1],
    )
    assert predictions[1][0]["score"] == pytest.approx((0.8 * 0.04) ** 0.5)


def test_logits_require_exact_uid_coverage(tmp_path: Path) -> None:
    records = [_record(f"p{fold}", fold=fold) for fold in (0, 1, 2)]
    for fold in (0, 1, 2):
        np.savez_compressed(
            tmp_path / f"fold_{fold}_logits.npz",
            logits=np.zeros((1, 25), dtype=np.float32),
            proposal_uids=np.asarray([f"p{fold}"], dtype="<U2"),
        )
    loaded = load_logits(tmp_path, records)
    assert set(loaded) == {"p0", "p1", "p2"}
    (tmp_path / "fold_2_logits.npz").unlink()
    with pytest.raises(FileNotFoundError):
        load_logits(tmp_path, records)


def test_prepare_manifest_joins_images_and_proposals(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    (aggregate / "oof_metadata.json").write_text("{}\n", encoding="utf-8")
    (aggregate / "predictions_oof_low.json").write_text("[]\n", encoding="utf-8")
    with (aggregate / "oof_images.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "relative_path", "fold"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow({"image_id": 1, "relative_path": "images/a.jpg", "fold": 0})
    with (aggregate / "oof_proposals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "proposal_uid",
                "image_id",
                "fold",
                "category_id",
                "x",
                "y",
                "width",
                "height",
                "score",
                "source_prediction_index",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "proposal_uid": "p0",
                "image_id": 1,
                "fold": 0,
                "category_id": 24,
                "x": 2,
                "y": 3,
                "width": 10,
                "height": 20,
                "score": 0.5,
                "source_prediction_index": 0,
            }
        )
    output = tmp_path / "manifest.csv"
    audit = prepare_proposal_manifest(
        aggregate,
        output,
        expected_proposals=1,
        expected_artifact_sha256={},
    )
    assert audit["proposal_count"] == 1
    row = next(csv.DictReader(output.open("r", encoding="utf-8", newline="")))
    assert row["relative_path"] == "images/a.jpg"
    assert float(row["x1"]) == 12.0
    assert json.loads((aggregate / "oof_metadata.json").read_text()) == {}
