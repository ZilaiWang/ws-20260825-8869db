"""不均衡小样本清单生成测试。"""

import json
from pathlib import Path

import pytest

from rsdet.data.imbalance import prepare_ultralytics_data
from rsdet.engine.trainer import train


def _make_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "data"
    samples = []
    image_id = 1
    # 高频类 0 出现在 20 张图，其余类各 1 张，确保训练 split 覆盖 25 类。
    train_labels = [0] * 20 + list(range(1, 25))
    for class_id in train_labels:
        stem = f"train_{image_id:03d}"
        image = root / "images" / "train" / f"{stem}.jpg"
        label = root / "labels" / "train" / f"{stem}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image-placeholder")
        label.write_text(f"{class_id} 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        samples.append(
            {
                "image_id": image_id,
                "relative_path": f"images/train/{stem}.jpg",
                "split": "train",
                "group_id": stem,
            }
        )
        image_id += 1
    for index in range(2):
        stem = f"val_{index:03d}"
        image = root / "images" / "train" / f"{stem}.jpg"
        label = root / "labels" / "train" / f"{stem}.txt"
        image.write_bytes(b"image-placeholder")
        label.write_text(f"{index} 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        samples.append(
            {
                "image_id": image_id,
                "relative_path": f"images/train/{stem}.jpg",
                "split": "val",
                "group_id": stem,
            }
        )
        image_id += 1
    manifest = tmp_path / "dev_v1.json"
    manifest.write_text(
        json.dumps({"version": "dev_v1", "samples": samples}), encoding="utf-8"
    )
    return root, manifest


def test_prepare_balanced_ultralytics_lists(tmp_path: Path) -> None:
    root, manifest = _make_dataset(tmp_path)
    prepared = prepare_ultralytics_data(
        data_root=root,
        manifest_path=manifest,
        output_dir=tmp_path / "prepared",
        seed=7,
        repeat_threshold=0.2,
        max_repeat=4.0,
    )

    base_lines = prepared.train_list.read_text(encoding="utf-8").splitlines()
    balanced_lines = prepared.balanced_train_list.read_text(encoding="utf-8").splitlines()
    val_lines = prepared.val_list.read_text(encoding="utf-8").splitlines()
    report = json.loads(prepared.statistics_json.read_text(encoding="utf-8"))

    assert len(base_lines) == 44
    assert len(balanced_lines) > len(base_lines)
    assert len(val_lines) == 2
    assert report["classes"][1]["effective_number_weight"] > report["classes"][0][
        "effective_number_weight"
    ]
    rare_sample = next(sample for sample in report["samples"] if sample["labels"] == [1])
    common_sample = next(sample for sample in report["samples"] if sample["labels"] == [0])
    assert rare_sample["repeat_factor"] > common_sample["repeat_factor"]


def test_balanced_list_is_reproducible(tmp_path: Path) -> None:
    root, manifest = _make_dataset(tmp_path)
    first = prepare_ultralytics_data(
        data_root=root,
        manifest_path=manifest,
        output_dir=tmp_path / "first",
        seed=42,
    )
    second = prepare_ultralytics_data(
        data_root=root,
        manifest_path=manifest,
        output_dir=tmp_path / "second",
        seed=42,
    )
    first_text = first.balanced_train_list.read_text(encoding="utf-8")
    second_text = second.balanced_train_list.read_text(encoding="utf-8")
    assert first_text == second_text


@pytest.mark.skip(
    reason="旧 YOLO train() 接口已被 BHC-DETR 训练引擎替换；"
    "train() 仅接受 model: bhcdetr，本测试的 yolo 配置为历史遗留"
)
def test_training_dry_run_materializes_both_stages(tmp_path: Path) -> None:
    root, manifest = _make_dataset(tmp_path)
    config = {
        "seed": 42,
        "output_dir": str(tmp_path / "experiment"),
        "model": {"family": "yolo", "weights": "pretrained.pt"},
        "data": {"root": str(root), "manifest": str(manifest)},
        "imbalance": {"repeat_threshold": 0.05, "max_repeat": 4.0},
        "train": {
            "device": "cuda:0",
            "args": {"imgsz": 640, "batch": 2},
            "stages": [
                {"name": "foundation", "epochs": 2, "balanced": False, "args": {}},
                {"name": "rebalance", "epochs": 1, "balanced": True, "args": {}},
            ],
        },
    }

    summary = train(config, dry_run=True)

    assert summary["dry_run"] is True
    assert [stage["balanced"] for stage in summary["stages"]] == [False, True]
    assert (tmp_path / "experiment" / "train_summary.json").is_file()
