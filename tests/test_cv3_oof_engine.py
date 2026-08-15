"""CV3 OOF 训练/推理引擎契约测试（补齐 model 仓库 train.py/infer.py）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_cv3_oof import (  # noqa: E402
    _build_train_arguments,
    build_dataset_yaml,
    _write_train_summary,
)

from rsdet.experiments.cv3_oof import (  # noqa: E402
    FORMAL_TRAINING_CONTRACTS,
    _validate_train_summary,
)


def _write_split_view(tmp_path: Path) -> Path:
    view = tmp_path / "split_view.json"
    view.write_text(
        json.dumps(
            {
                "held_out_fold": 0,
                "train_images": 2,
                "val_images": 1,
                "samples": [
                    {"image_id": 1, "relative_path": "images/train/a.jpg", "split": "train",
                     "source_fold": 1, "group_id": "g1", "group_rule": "r"},
                    {"image_id": 2, "relative_path": "images/train/b.jpg", "split": "train",
                     "source_fold": 2, "group_id": "g2", "group_rule": "r"},
                    {"image_id": 3, "relative_path": "images/train/c.jpg", "split": "val",
                     "source_fold": 0, "group_id": "g3", "group_rule": "r"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return view


def _write_train_request(tmp_path: Path, split_view: Path, weights: Path) -> Path:
    request = tmp_path / "train_request.yaml"
    request.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "test",
                "seed": 42,
                "output_dir": str(tmp_path / "training"),
                "model": {"family": "rtdetr", "weights": str(weights)},
                "data": {"root": str(tmp_path / "data"), "manifest": str(split_view)},
                "train": {
                    "device": "cuda:0",
                    "checkpoint_selection": "last",
                    "args": {
                        "imgsz": 1024, "batch": 4, "workers": 8, "optimizer": "AdamW",
                        "lr0": 0.0002, "lrf": 0.01, "weight_decay": 0.0001,
                        "warmup_epochs": 3, "cos_lr": True, "amp": True,
                        "deterministic": True, "patience": 0, "val": False, "plots": False,
                    },
                    "stages": [{"name": "foundation", "epochs": 120, "balanced": False, "args": {}}],
                },
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return request


def test_build_dataset_yaml_splits(tmp_path: Path) -> None:
    split_view = _write_split_view(tmp_path)
    dataset_yaml = build_dataset_yaml(split_view, tmp_path / "data", tmp_path / "out")
    content = dataset_yaml.read_text(encoding="utf-8")
    assert "nc: 25" in content
    train_txt = (tmp_path / "out" / "train.txt").read_text(encoding="utf-8").splitlines()
    val_txt = (tmp_path / "out" / "val.txt").read_text(encoding="utf-8").splitlines()
    assert len(train_txt) == 2
    assert len(val_txt) == 1
    assert all(str(tmp_path / "data") in line for line in train_txt)


def test_train_summary_satisfies_contract(tmp_path: Path) -> None:
    weights = tmp_path / "rtdetr-l.pt"
    weights.write_bytes(b"fake-weights")
    import hashlib

    weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()

    split_view = _write_split_view(tmp_path)
    request = _write_train_request(tmp_path, split_view, weights)
    config = yaml.safe_load(request.read_text(encoding="utf-8"))

    output_dir = tmp_path / "training"
    dataset_yaml = build_dataset_yaml(split_view, tmp_path / "data", output_dir)
    arguments = _build_train_arguments(config, dataset_yaml)
    _write_train_summary(config, arguments, output_dir, dry_run=False)

    checkpoint = output_dir / "runs" / "foundation" / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"ckpt")

    _validate_train_summary(
        output_dir / "train_summary.json",
        expected_family="rtdetr",
        expected_pretrained_weight=str(weights),
        expected_pretrained_sha256=weights_sha,
        expected_seed=42,
        expected_foundation_epochs=120,
        expected_checkpoint_path=checkpoint,
        expected_training_contract=FORMAL_TRAINING_CONTRACTS["M3"],
    )


def test_train_arguments_match_m3_contract(tmp_path: Path) -> None:
    split_view = _write_split_view(tmp_path)
    request = _write_train_request(tmp_path, split_view, tmp_path / "w.pt")
    config = yaml.safe_load(request.read_text(encoding="utf-8"))
    dataset_yaml = build_dataset_yaml(split_view, tmp_path / "data", tmp_path / "out")
    arguments = _build_train_arguments(config, dataset_yaml)

    contract = FORMAL_TRAINING_CONTRACTS["M3"]
    for key, value in contract["train_args"].items():
        assert arguments[key] == value, f"train_args.{key}"
    for key, value in contract["stage_args"].items():
        assert arguments[key] == value, f"stage_args.{key}"
    assert arguments["epochs"] == 120
    assert arguments["device"] == "cuda:0"
    assert arguments["seed"] == 42
