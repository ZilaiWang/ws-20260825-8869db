from __future__ import annotations

import hashlib
import runpy
import sys
from pathlib import Path

import pytest
import yaml


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_coarse_dry_run_rejects_wrong_class_order(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(yaml.safe_dump({"names": ["ship", "aircraft"]}))
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--dataset",
            str(dataset),
            "--weights",
            str(weights),
            "--expected-weight-sha256",
            _sha(weights),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )
    with pytest.raises(ValueError, match="frozen four-class"):
        runpy.run_path("scripts/train_external_y5_coarse.py", run_name="__main__")


def test_head_transfer_occurs_before_optimizer_and_ema_creation() -> None:
    source = Path("src/rsdet/external/transfer.py").read_text(encoding="utf-8")
    assert "def setup_model" in source
    assert "def _setup_train" not in source
    assert "before optimizer/EMA creation" in source
    assert "DetectionModel" in source
    assert "after Ultralytics bias_init" in source
    assert "reset_parameters" not in source


def test_fine_transfer_contract_is_staged() -> None:
    source = Path("scripts/train_external_initialized_y5_fine.py").read_text(
        encoding="utf-8"
    )
    assert "external_backbone_neck_to_reviewed_official_fine_staged_v2" in source
    assert '"freeze": args.freeze_layers' in source
    assert "YOLO(str(warmup_checkpoint.resolve()))" in source
    assert "optimizer_is_recreated_between_stages" in source
