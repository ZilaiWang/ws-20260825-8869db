from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rsdet.analysis.oof_detection import (
    FormalGroundTruth,
    GroundTruthObject,
    build_paired_object_outcomes,
    build_threshold_curve,
    decompose_official_errors,
    load_analysis_config,
    load_formal_ground_truth,
    load_oof_aggregate,
    select_exploratory_workpoint,
)
from rsdet.evaluation.protocol import EvaluationProtocol
from rsdet.experiments.cv3_oof import sha256_file


def _protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        contract_version="contract_v1",
        eval_version="official_eval_v1",
        ranking_version="official_ranking_v1_6",
        class_names=["ship"],
        category_mapping={0: "ship", 1: "ship", 2: "ship"},
        iou_thresholds={"ship": 0.5},
        recall_min=0.85,
        fdr_max=0.20,
    )


def _formal_gt() -> FormalGroundTruth:
    specs = [
        ("gt-tp", 0, [0.0, 0.0, 10.0, 10.0]),
        ("gt-cls", 1, [20.0, 0.0, 30.0, 10.0]),
        ("gt-loc", 2, [40.0, 0.0, 50.0, 10.0]),
        ("gt-miss", 0, [60.0, 0.0, 70.0, 10.0]),
    ]
    boxes = {1: [{"category_id": category_id, "bbox_xyxy": bbox} for _, category_id, bbox in specs]}
    objects = {
        (1, index): GroundTruthObject(
            annotation_uid=uid,
            image_id=1,
            ground_truth_index=index,
            fold=0,
            group_id="group-0",
            category_id=category_id,
            class_name="ship",
            bbox_xyxy=tuple(bbox),
        )
        for index, (uid, category_id, bbox) in enumerate(specs)
    }
    return FormalGroundTruth(
        boxes=boxes,
        objects=objects,
        image_ids=frozenset({1}),
        annotation_count=4,
    )


def _predictions() -> dict[int, list[dict[str, object]]]:
    return {
        1: [
            {
                "category_id": 0,
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                "score": 0.90,
            },
            {
                "category_id": 0,
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                "score": 0.80,
            },
            {
                "category_id": 0,
                "bbox_xyxy": [20.0, 0.0, 30.0, 10.0],
                "score": 0.70,
            },
            {
                "category_id": 2,
                "bbox_xyxy": [45.0, 0.0, 55.0, 10.0],
                "score": 0.60,
            },
            {
                "category_id": 0,
                "bbox_xyxy": [100.0, 0.0, 110.0, 10.0],
                "score": 0.50,
            },
        ]
    }


def test_frozen_analysis_config_loads() -> None:
    config = load_analysis_config("configs/experiments/m1_m3_cv3_oof_analysis_v1.yaml")
    assert config["candidate_floor"] == 0.001
    assert config["thresholds"][0] == 0.001
    assert config["thresholds"][-1] == 1.0


def test_error_decomposition_is_count_conserving_and_hierarchical() -> None:
    summary, cases, fn_reason = decompose_official_errors(
        _formal_gt(),
        _predictions(),
        threshold=0.001,
        protocol=_protocol(),
        model_key="M1",
    )

    assert summary["official_metrics"]["details"] == {
        "tp": 1,
        "fp": 4,
        "fn": 3,
        "total_gt": 4,
        "total_pred": 5,
        "matching_policy": "same_fine_category_id",
        "aggregation_policy": "fine_match_then_coarse_and_overall",
        "iou_thresholds": {"ship": 0.5},
        "empty_gt_recall_policy": 1.0,
        "empty_prediction_fdr_policy": 0.0,
    }
    assert summary["fp_counts"] == {
        "FP_DUP": 1,
        "FP_CLS": 1,
        "FP_LOC": 1,
        "FP_BG": 1,
    }
    assert summary["fn_counts"] == {
        "FN_CLS": 1,
        "FN_LOC": 1,
        "FN_MISS": 1,
    }
    assert summary["conservation"]["passed"] is True
    assert fn_reason == {
        (1, 1): "FN_CLS",
        (1, 2): "FN_LOC",
        (1, 3): "FN_MISS",
    }
    assert len(cases) == 7


def test_threshold_curve_is_exact_and_selection_is_not_deployment_admission() -> None:
    curve, parity = build_threshold_curve(
        _formal_gt().boxes,
        _predictions(),
        thresholds=[0.001, 0.65, 0.85, 1.0],
        protocol=_protocol(),
    )
    by_threshold = {row["threshold"]: row for row in curve}
    assert by_threshold[0.001]["tp"] == 1
    assert by_threshold[0.001]["fp"] == 4
    assert by_threshold[0.85]["tp"] == 1
    assert by_threshold[0.85]["fp"] == 0
    assert parity["status"] == "pass"

    selected = select_exploratory_workpoint(
        curve,
        recall_min=0.85,
        fdr_max=0.20,
    )
    assert selected["threshold"] == 0.85
    assert selected["official_gate_passed"] is False
    assert selected["same_oof_selection"] is True
    assert selected["deployment_admission"] is False


def test_paired_objects_and_oracle_recall_use_official_tp() -> None:
    gt = _formal_gt()
    m1 = {
        1: [
            {
                "category_id": 0,
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                "score": 0.9,
            }
        ]
    }
    m3 = {
        1: [
            {
                "category_id": 1,
                "bbox_xyxy": [20.0, 0.0, 30.0, 10.0],
                "score": 0.9,
            }
        ]
    }

    rows, summary = build_paired_object_outcomes(
        gt,
        {"M1": m1, "M3": m3},
        thresholds={"M1": 0.5, "M3": 0.5},
        protocol=_protocol(),
        label="fixture",
    )

    outcomes = {row["annotation_uid"]: row["paired_outcome"] for row in rows}
    assert outcomes == {
        "gt-tp": "M1_only",
        "gt-cls": "M3_only",
        "gt-loc": "neither",
        "gt-miss": "neither",
    }
    assert summary["summary"]["overall"]["M1_only"] == 1
    assert summary["summary"]["overall"]["M3_only"] == 1
    assert summary["summary"]["overall"]["oracle_union_recall"] == 0.5
    assert summary["oracle_fdr"] is None
    assert summary["deployable_ensemble_claim"] is False


def test_formal_gt_requires_exact_sha_and_one_tight_row(tmp_path: Path) -> None:
    manifest = tmp_path / "formal.csv"
    fieldnames = [
        "manifest_version",
        "annotation_uid",
        "formal_image_id",
        "fold",
        "group_id",
        "class_id",
        "major_class",
        "gt_x0",
        "gt_y0",
        "gt_x1",
        "gt_y1",
        "crop_policy",
        "coordinate_semantics",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "manifest_version": "formal_crop_manifest_v2",
                "annotation_uid": "a",
                "formal_image_id": 1,
                "fold": 0,
                "group_id": "g",
                "class_id": 0,
                "major_class": "ship",
                "gt_x0": 0,
                "gt_y0": 0,
                "gt_x1": 10,
                "gt_y1": 10,
                "crop_policy": "tight",
                "coordinate_semantics": "continuous_float_xyxy_half_open",
            }
        )
    sha = sha256_file(manifest)

    result = load_formal_ground_truth(
        manifest,
        expected_sha256=sha,
        expected_images=1,
        expected_annotations=1,
    )
    assert result.annotation_count == 1
    with pytest.raises(ValueError, match="SHA"):
        load_formal_ground_truth(
            manifest,
            expected_sha256="0" * 64,
            expected_images=1,
            expected_annotations=1,
        )


def test_oof_aggregate_requires_sha_closure(tmp_path: Path) -> None:
    root = tmp_path / "aggregate"
    root.mkdir()
    predictions = root / "predictions_oof_low.json"
    predictions.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [0, 0, 10, 10],
                    "score": 0.1,
                }
            ]
        ),
        encoding="utf-8",
    )
    images = root / "oof_images.csv"
    images.write_text(
        f"image_id,fold,model_key,checkpoint_sha256,prediction_count\n1,0,M1,{'a' * 64},1\n",
        encoding="utf-8",
    )
    proposals = root / "oof_proposals.csv"
    proposals.write_text(
        "proposal_uid,image_id,fold,category_id,x,y,width,height,score,"
        "model_key,checkpoint_sha256,source_prediction_index\n"
        f"p1,1,0,0,0,0,10,10,0.1,M1,{'a' * 64},0\n",
        encoding="utf-8",
    )
    metadata = {
        "contract_version": "cv3_oof_v1",
        "status": "complete_downstream_ready",
        "downstream_admission": True,
        "model_key": "M1",
        "source_manifest_sha256": "1" * 64,
        "image_count": 1,
        "proposal_count": 1,
        "fold_checkpoint_sha256": ["a" * 64, "b" * 64, "c" * 64],
        "low_score_threshold": 0.001,
        "formal_crop_manifest": {"sha256": "2" * 64},
        "artifacts": {
            "oof_images": {
                "path": str(images),
                "sha256": sha256_file(images),
            },
            "oof_proposals": {
                "path": str(proposals),
                "sha256": sha256_file(proposals),
            },
            "predictions_oof_low": {
                "path": str(predictions),
                "sha256": sha256_file(predictions),
            },
        },
    }
    metadata_path = root / "oof_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded, pred = load_oof_aggregate(
        root,
        expected_model_key="M1",
        expected_manifest_sha256="1" * 64,
        expected_formal_crop_sha256="2" * 64,
        expected_images=1,
        candidate_floor=0.001,
    )
    assert loaded["predictions_sha256"] == sha256_file(predictions)
    assert loaded["aggregate_cross_file_audit"]["status"] == "pass"
    assert len(pred[1]) == 1

    predictions.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        load_oof_aggregate(
            root,
            expected_model_key="M1",
            expected_manifest_sha256="1" * 64,
            expected_formal_crop_sha256="2" * 64,
            expected_images=1,
            candidate_floor=0.001,
        )


def test_oof_aggregate_rejects_self_consistent_proposal_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "aggregate"
    root.mkdir()
    predictions = root / "predictions_oof_low.json"
    predictions.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [0, 0, 10, 10],
                    "score": 0.1,
                }
            ]
        ),
        encoding="utf-8",
    )
    images = root / "oof_images.csv"
    images.write_text(
        f"image_id,fold,model_key,checkpoint_sha256,prediction_count\n1,0,M1,{'a' * 64},1\n",
        encoding="utf-8",
    )
    proposals = root / "oof_proposals.csv"
    proposals.write_text(
        "proposal_uid,image_id,fold,category_id,x,y,width,height,score,"
        "model_key,checkpoint_sha256,source_prediction_index\n"
        f"p1,1,0,0,0,0,10,10,0.2,M1,{'a' * 64},0\n",
        encoding="utf-8",
    )
    metadata = {
        "contract_version": "cv3_oof_v1",
        "status": "complete_downstream_ready",
        "downstream_admission": True,
        "model_key": "M1",
        "source_manifest_sha256": "1" * 64,
        "image_count": 1,
        "proposal_count": 1,
        "fold_checkpoint_sha256": ["a" * 64, "b" * 64, "c" * 64],
        "low_score_threshold": 0.001,
        "formal_crop_manifest": {"sha256": "2" * 64},
        "artifacts": {
            "oof_images": {
                "path": str(images),
                "sha256": sha256_file(images),
            },
            "oof_proposals": {
                "path": str(proposals),
                "sha256": sha256_file(proposals),
            },
            "predictions_oof_low": {
                "path": str(predictions),
                "sha256": sha256_file(predictions),
            },
        },
    }
    (root / "oof_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="数值"):
        load_oof_aggregate(
            root,
            expected_model_key="M1",
            expected_manifest_sha256="1" * 64,
            expected_formal_crop_sha256="2" * 64,
            expected_images=1,
            candidate_floor=0.001,
        )
