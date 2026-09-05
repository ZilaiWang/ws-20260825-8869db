from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_p40_hier_d4_runtime_config import build_config
from run_competition_runtime_coco import resolve_image


def test_build_hier_d4_config_uses_disjoint_owners(tmp_path: Path) -> None:
    expert = tmp_path / "expert.pt"
    expert.write_bytes(b"hierarchy")
    base = {
        "deployment_role": "aircraft_only",
        "workpoint_id": "old",
        "model": {
            "weight_path": "/models/p40.pt",
            "expected_sha256": "a" * 64,
            "imgsz": 1280,
        },
        "pipeline": {"score_threshold": 0.001},
        "post_fusion_score_threshold": 0.536,
        "aircraft_classifier_model": {"method": "view_consistency_d4"},
    }
    result = build_config(
        json.loads(json.dumps(base)),
        expert_weight=expert,
        expert_threshold=0.546,
        workpoint_id="candidate",
    )
    assert "post_fusion_score_threshold" not in result
    assert result["resolution_route"] == {
        "primary_labels": list(range(24)),
        "expert_labels": [24],
        "primary_threshold": 0.536,
        "expert_threshold": 0.546,
    }
    assert result["resolution_expert_model"]["weight_path"] == str(expert)
    assert result["aircraft_classifier_model"]["method"] == "view_consistency_d4"


def test_build_hier_d4_config_can_add_primary_vehicle_fallback(tmp_path: Path) -> None:
    expert = tmp_path / "expert.pt"
    expert.write_bytes(b"hierarchy")
    base = {
        "post_fusion_score_threshold": 0.536,
        "model": {"weight_path": "/models/p40.pt"},
        "pipeline": {"score_threshold": 0.001},
        "aircraft_classifier_model": {"method": "view_consistency_d4"},
    }
    result = build_config(
        base,
        expert_weight=expert,
        expert_threshold=0.42,
        workpoint_id="rescue",
        primary_rescue_threshold=0.60,
        primary_rescue_dedup_iou=0.70,
    )
    assert result["resolution_primary_rescue"] == {
        "labels": [24],
        "threshold": 0.60,
        "dedup_iou": 0.70,
    }


def test_resolve_image_uses_declared_fold(tmp_path: Path) -> None:
    image = tmp_path / "fold_2" / "images" / "x.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")
    assert resolve_image(tmp_path, {"id": 3, "fold": 2, "file_name": "x.jpg"}) == image
