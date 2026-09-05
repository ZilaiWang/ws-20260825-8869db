import importlib.util
from pathlib import Path

import pytest

from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_module = load_script("audit_max_det_saturation")
config_module = load_script("materialize_yolo_cross_resolution_infer_config")
matrix_module = load_script("analyze_resolution_cross_matrix")
route_module = load_script("analyze_cv3_class_resolution_route")


def test_max_det_saturation_detects_exact_cap() -> None:
    rows = [{"image_id": 1, "category_id": 4} for _ in range(5)] + [
        {"image_id": 2, "category_id": 24} for _ in range(3)
    ]
    result = audit_module.audit(rows, 5)
    assert result["saturated_image_ids"] == [1]
    assert result["decision"] == "rerun_paired_models_with_higher_max_det"


def test_cross_resolution_config_changes_only_resolution_and_output(tmp_path: Path) -> None:
    source = {
        "model": {
            "family": "yolo",
            "checkpoint": "/tmp/model.pt",
            "imgsz": 1024,
            "confidence": 0.001,
            "max_detections": 500,
        },
        "tiling": {"enabled": False, "force": False},
        "output_json": "/tmp/old.json",
        "sentinel": {"frozen": True},
    }
    output = config_module.materialize(source, imgsz=1280, output_json=tmp_path / "new.json")
    assert output["model"]["imgsz"] == 1280
    assert output["model"]["checkpoint"] == "/tmp/model.pt"
    assert output["sentinel"] == {"frozen": True}
    assert source["model"]["imgsz"] == 1024


def test_cross_resolution_config_can_replace_only_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapted.pt"
    checkpoint.write_bytes(b"checkpoint")
    source = {
        "model": {
            "family": "yolo",
            "checkpoint": "/tmp/original.pt",
            "imgsz": 1024,
            "confidence": 0.001,
            "max_detections": 500,
        },
        "tiling": {"enabled": False, "force": False},
        "output_json": "/tmp/old.json",
        "sentinel": {"frozen": True},
    }
    output = config_module.materialize(
        source,
        imgsz=1280,
        output_json=tmp_path / "new.json",
        checkpoint=checkpoint,
    )
    assert output["model"]["checkpoint"] == str(checkpoint.resolve())
    assert output["model"]["imgsz"] == 1280
    assert output["sentinel"] == {"frozen": True}
    assert source["model"]["checkpoint"] == "/tmp/original.pt"


def crossfit_frontier(recall: float, fdr: float, vehicle_recall: float):
    coarse = {
        "ship": {"macro_recall": recall, "macro_fdr": fdr},
        "aircraft": {"macro_recall": recall, "macro_fdr": fdr},
        "vehicle": {"macro_recall": vehicle_recall, "macro_fdr": fdr},
    }
    return {
        "fold_image_ids": {"0": [0], "1": [1], "2": [2]},
        "frontiers": {
            "0.100": {
                "crossfit_thresholds": {"0": 0.7, "1": 0.7, "2": 0.7},
                "crossfit": {
                    "platform": {
                        "gate_recall": (2 * recall + vehicle_recall) / 3,
                        "gate_fdr": fdr,
                        "per_coarse": coarse,
                    }
                },
            },
            "0.150": {
                "crossfit_thresholds": {"0": 0.4, "1": 0.5, "2": 0.6},
                "crossfit": {
                    "platform": {
                        "gate_recall": (2 * recall + vehicle_recall) / 3,
                        "gate_fdr": fdr,
                        "per_coarse": coarse,
                    }
                },
            },
        },
    }


def test_resolution_matrix_accepts_outer_crossfit_frontiers() -> None:
    inputs = {
        "t1024_i1024": crossfit_frontier(0.50, 0.15, 0.40),
        "t1024_i1280": crossfit_frontier(0.52, 0.15, 0.50),
        "t1280_i1024": crossfit_frontier(0.51, 0.15, 0.42),
        "t1280_i1280": crossfit_frontier(0.56, 0.14, 0.60),
    }
    result = matrix_module.analyze(inputs)
    assert result["selection"] == "outer_crossfit"
    assert result["effects"]["inference_resolution_at_train1024"]["gate_recall_pp"] > 0


def test_route_is_disjoint_and_preserves_branch_ownership() -> None:
    gt = {}
    primary = {}
    highres = {}
    fold_ids = {"0": [], "1": [], "2": []}
    for fold in range(3):
        image_id = fold + 1
        fold_ids[str(fold)].append(image_id)
        gt[image_id] = []
        primary[image_id] = []
        highres[image_id] = []
        for category_id in range(25):
            x = float(category_id * 20)
            row = {"bbox_xyxy": [x, 0.0, x + 10.0, 10.0], "category_id": category_id}
            gt[image_id].append(row)
            primary[image_id].append({**row, "score": 0.9})
            highres[image_id].append({**row, "score": 0.9})
    frontier = {
        "fold_image_ids": fold_ids,
        "frontiers": {"0.150": {"crossfit_thresholds": {"0": 0.5, "1": 0.5, "2": 0.5}}},
    }
    protocol = parse_evaluation_protocol(load_config(ROOT / "configs/project.yaml"))
    result = route_module.analyze(
        gt=gt,
        primary=primary,
        highres=highres,
        primary_frontier=frontier,
        highres_frontier=frontier,
        primary_labels=frozenset(range(24)),
        highres_labels=frozenset({24}),
        level="0.150",
        protocol=protocol,
    )
    assert result["route"]["platform"]["gate_recall"] == pytest.approx(1.0)
    assert result["route"]["platform"]["gate_fdr"] == pytest.approx(0.0)


def test_route_supports_branch_specific_risk_levels() -> None:
    gt = {}
    primary = {}
    highres = {}
    fold_ids = {"0": [], "1": [], "2": []}
    for fold in range(3):
        image_id = fold + 1
        fold_ids[str(fold)].append(image_id)
        gt[image_id] = []
        primary[image_id] = []
        highres[image_id] = []
        for category_id in range(25):
            x = float(category_id * 20)
            row = {"bbox_xyxy": [x, 0.0, x + 10.0, 10.0], "category_id": category_id}
            gt[image_id].append(row)
            primary[image_id].append({**row, "score": 0.6})
            highres[image_id].append({**row, "score": 0.8})
    frontier = crossfit_frontier(1.0, 0.0, 1.0)
    frontier["fold_image_ids"] = fold_ids
    protocol = parse_evaluation_protocol(load_config(ROOT / "configs/project.yaml"))
    result = route_module.analyze(
        gt=gt,
        primary=primary,
        highres=highres,
        primary_frontier=frontier,
        highres_frontier=frontier,
        primary_labels=frozenset(range(4, 24)),
        highres_labels=frozenset({0, 1, 2, 3, 24}),
        primary_level="0.150",
        highres_level="0.100",
        protocol=protocol,
    )
    assert result["fdr_level"] is None
    assert result["fdr_level_by_branch"] == {
        "primary": "0.150",
        "highres": "0.100",
    }
    assert result["route"]["platform"]["gate_recall"] == pytest.approx(1.0)


def test_route_transfers_source_thresholds_to_distinct_target_folds() -> None:
    source = crossfit_frontier(1.0, 0.0, 1.0)
    target_folds = {101: 0, 102: 1, 103: 2}
    gt = {
        image_id: [
            {
                "bbox_xyxy": [float(category_id * 20), 0.0, float(category_id * 20 + 10), 10.0],
                "category_id": category_id,
            }
            for category_id in range(25)
        ]
        for image_id in target_folds
    }
    predictions = {
        image_id: [
            {
                "bbox_xyxy": [float(category_id * 20), 0.0, float(category_id * 20 + 10), 10.0],
                "category_id": category_id,
                "score": 0.9,
            }
            for category_id in range(25)
        ]
        for image_id in target_folds
    }
    protocol = parse_evaluation_protocol(load_config(ROOT / "configs/project.yaml"))
    result = route_module.analyze(
        gt=gt,
        primary=predictions,
        highres=predictions,
        primary_frontier=source,
        highres_frontier=source,
        primary_labels=frozenset(range(24)),
        highres_labels=frozenset({24}),
        level="0.150",
        protocol=protocol,
        target_image_folds=target_folds,
    )
    assert result["threshold_source_is_evaluation_target"] is False
    assert result["target_image_count"] == 3
    assert result["route"]["platform"]["per_coarse"]["vehicle"]["macro_recall"] == 1.0


def test_route_rejects_overlapping_labels() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        route_module.analyze(
            gt={},
            primary={},
            highres={},
            primary_frontier={},
            highres_frontier={},
            primary_labels=frozenset(range(25)),
            highres_labels=frozenset({24}),
            level="0.150",
            protocol=None,
        )
