from copy import deepcopy
from pathlib import Path

import yaml

from scripts.materialize_standard_yolo_infer_config import (
    materialize as materialize_infer,
)
from scripts.materialize_yolo_capacity_scale_config import (
    materialize as materialize_train,
)
from scripts.summarize_yolo_capacity_scale_screen import summarize

ROOT = Path(__file__).resolve().parents[1]


def _frontier(recall: float, fdr: float, *, floor_gain: float = 0.0) -> dict:
    coarse = {
        "ship": 0.70 + floor_gain,
        "aircraft": 0.90,
        "vehicle": 3 * recall - (0.70 + floor_gain) - 0.90,
    }

    def point(gate_recall: float, gate_fdr: float) -> dict:
        return {
            "threshold": 0.1,
            "pooled_recall": 0.9,
            "pooled_fdr": 0.1,
            "platform": {
                "metric_protocol": "platform_observed_20260831",
                "gate_recall": gate_recall,
                "gate_fdr": gate_fdr,
                "per_coarse": {
                    key: {"macro_recall": value, "macro_fdr": gate_fdr}
                    for key, value in coarse.items()
                },
            },
        }

    return {
        "status": "complete_diagnostic_only",
        "input_sha256": {"gt": "same"},
        "score_floor_metrics": point(0.9 + floor_gain / 3, 0.5),
        "frontiers": {"0.150": point(recall, fdr)},
    }


def test_four_training_contracts_are_a_clean_two_by_two_matrix() -> None:
    expected = {
        "s25_yolo26s_1024_fold0_40ep.yaml": 1024,
        "s25_yolo26s_1280_fold0_40ep.yaml": 1280,
        "m25_yolo26m_1024_fold0_40ep.yaml": 1024,
        "m25_yolo26m_1280_fold0_40ep.yaml": 1280,
    }
    normalized = []
    for filename, imgsz in expected.items():
        payload = yaml.safe_load(
            (ROOT / "configs/experiments" / filename).read_text(encoding="utf-8")
        )
        assert payload["seed"] == 42
        assert payload["train"]["args"]["imgsz"] == imgsz
        assert payload["train"]["args"]["batch"] == 8
        assert payload["train"]["stages"][0]["epochs"] == 40
        assert payload["train"]["stages"][0]["args"]["close_mosaic"] == 20
        payload["experiment_id"] = "condition"
        payload["model"]["weights"] = "weight"
        payload["train"]["args"]["imgsz"] = "scale"
        normalized.append(payload)
    assert all(payload == normalized[0] for payload in normalized[1:])


def test_training_materializer_changes_paths_without_changing_contract(tmp_path: Path) -> None:
    template = yaml.safe_load(
        (
            ROOT
            / "configs/experiments/m25_yolo26m_1024_fold0_40ep.yaml"
        ).read_text(encoding="utf-8")
    )
    payload = materialize_train(
        template,
        output_dir=tmp_path / "result",
        data_root=tmp_path / "data",
        split_view=tmp_path / "split.json",
        weights=tmp_path / "model.pt",
        device="cuda:0",
    )
    assert payload["output_dir"] == str((tmp_path / "result").resolve())
    assert payload["train"]["args"]["imgsz"] == 1024
    assert payload["train"]["stages"][0]["epochs"] == 40


def test_inference_materializer_removes_specialist_routing(tmp_path: Path) -> None:
    base = {
        "model": {
            "family": "yolo",
            "checkpoint": "old.pt",
            "label_map": {4: 24},
            "drop_labels": [5],
            "score_transform": "sqrt",
        },
        "input": {"data_root": "/data", "manifest": "/split.json"},
        "tiling": {"enabled": True},
    }
    payload = materialize_infer(
        base,
        checkpoint=tmp_path / "new.pt",
        predictions=tmp_path / "pred.json",
        data_root=tmp_path / "portable-data",
        split_view=tmp_path / "portable-split.json",
        imgsz=1280,
        batch_size=4,
        device="cuda:0",
    )
    assert payload["model"]["imgsz"] == 1280
    assert payload["model"]["confidence"] == 0.001
    assert "label_map" not in payload["model"]
    assert "drop_labels" not in payload["model"]
    assert "score_transform" not in payload["model"]
    assert payload["tiling"] == {"enabled": False, "force": False}
    assert payload["input"] == {
        "data_root": str((tmp_path / "portable-data").resolve()),
        "manifest": str((tmp_path / "portable-split.json").resolve()),
        "split": "val",
    }


def test_matrix_selects_only_platform_admitted_condition() -> None:
    baseline = _frontier(0.75, 0.15)
    s1280 = _frontier(0.751, 0.149, floor_gain=0.003)
    m1024 = _frontier(0.74, 0.14, floor_gain=0.02)
    m1280 = _frontier(0.76, 0.14, floor_gain=0.02)
    result = summarize(
        {
            "s1024": baseline,
            "s1280": s1280,
            "m1024": m1024,
            "m1280": m1280,
        }
    )
    assert result["selected_for_cv3"] == "m1280"
    assert result["next_action"].startswith("run_selected_condition")


def test_matrix_rejects_pooled_only_improvement() -> None:
    baseline = _frontier(0.75, 0.15)
    candidate = _frontier(0.74, 0.14, floor_gain=0.02)
    candidate = deepcopy(candidate)
    candidate["frontiers"]["0.150"]["pooled_recall"] = 0.99
    result = summarize(
        {
            "s1024": baseline,
            "s1280": candidate,
            "m1024": candidate,
            "m1280": candidate,
        }
    )
    assert result["selected_for_cv3"] is None
    assert result["next_action"] == "stop_capacity_scale_route"


def test_server_driver_locks_assets_and_forbids_resume() -> None:
    source = (
        ROOT / "scripts/server/run_yolo_capacity_scale_fold0_condition.sh"
    ).read_text(encoding="utf-8")
    assert "a647ce030fa832aadc6a6c286a3f6464ac1783f71797a52cc598ec340f128943" in source
    assert "2641d3bb15388b9a19812ab514b993d5f68ef90d7a59fb02834bf7903e585977" in source
    assert "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b" in source
    assert "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7" in source
    assert "--innovation y5 --rotate90-p 1.0" in source
    assert "--resume" not in source
