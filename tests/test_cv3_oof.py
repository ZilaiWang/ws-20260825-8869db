from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import rsdet.experiments.cv3_oof as cv3_oof_module
from rsdet.experiments.cv3_oof import (
    audit_and_aggregate_oof,
    finalize_fold_delivery,
    load_cv3_manifest,
    prepare_oof_run_plan,
    sha256_file,
)


def _manifest(path: Path) -> Path:
    samples = []
    image_id = 1
    for fold in range(3):
        for member in range(2):
            samples.append(
                {
                    "image_id": image_id,
                    "relative_path": f"images/train/image_{image_id}.jpg",
                    "fold": fold,
                    "group_id": f"group_{fold}_{member}",
                    "group_rule": "fixture",
                }
            )
            image_id += 1
    path.write_text(
        json.dumps(
            {
                "version": "cv3_fixture_v1",
                "data_version": "fixture",
                "fold_count": 3,
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    return path


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _data_lock(path: Path, manifest: Path) -> Path:
    inventory = [
        {
            "image_id": image_id,
            "image": {"sha256": f"{image_id:064x}"},
            "label": {"sha256": f"{image_id + 10:064x}"},
        }
        for image_id in range(1, 7)
    ]
    payload = {
        "schema_version": "formal_detection_data_lock_v1",
        "contract": {"cv3_manifest_sha256": sha256_file(manifest)},
        "summary": {
            "image_count": 6,
            "label_file_count": 6,
            "object_count": 6,
            "p02_formal_gt_equivalence": True,
            "yolo_formal_gt_equivalence": True,
        },
        "inventory_fingerprint": _canonical_sha256(inventory),
        "inventory": inventory,
    }
    payload["lock_fingerprint"] = _canonical_sha256(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _write_data_lock_report(
    path: Path,
    plan: dict[str, object],
    *,
    overrides: dict[str, object] | None = None,
) -> Path:
    lock = plan["detection_data_lock"]
    assert isinstance(lock, dict)
    payload: dict[str, object] = {
        "status": "pass",
        "schema_version": "formal_detection_data_lock_v1",
        "lock_path": lock["path"],
        "lock_file_sha256": lock["sha256"],
        "lock_fingerprint": lock["lock_fingerprint"],
        "inventory_fingerprint": lock["inventory_fingerprint"],
        "image_count": lock["image_count"],
        "label_file_count": lock["label_file_count"],
        "object_count": lock["object_count"],
        "p02_formal_gt_equivalence": True,
        "yolo_formal_gt_equivalence": True,
    }
    payload.update(overrides or {})
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    manifest = _manifest(tmp_path / "cv3.json")
    pretrained = tmp_path / "yolo26s.pt"
    pretrained.write_bytes(b"official-pretrained-fixture")
    data_lock = _data_lock(tmp_path / "formal_detection_data_lock.json", manifest)
    run_root = tmp_path / "run"
    prepare_oof_run_plan(
        manifest_path=manifest,
        output_dir=run_root,
        model_key="M1",
        model_family="yolo",
        model_name="yolo26s",
        seed=42,
        input_size=1024,
        foundation_epochs=160,
        low_score_threshold=0.001,
        max_detections=500,
        pretrained_weight=str(pretrained),
        pretrained_weight_sha256=sha256_file(pretrained),
        detection_data_lock=str(data_lock),
        detection_data_lock_sha256=sha256_file(data_lock),
        expected_manifest_sha256=sha256_file(manifest),
        expected_image_count=6,
    )
    return manifest, run_root


def _complete_fold(
    run_root: Path,
    fold: int,
    *,
    wrong_prediction_fold: bool = False,
    prediction_count: int = 1,
    wrong_infer_output: bool = False,
    dry_run_summary: bool = False,
    wrong_runtime_checkpoint_sha: bool = False,
    wrong_runtime_predictions_path: bool = False,
    wrong_infer_iou: bool = False,
    data_lock_report_overrides: dict[str, object] | None = None,
) -> None:
    fold_dir = run_root / f"fold_{fold}"
    view_path = fold_dir / "split_view.json"
    plan = json.loads((run_root / "oof_run_plan.json").read_text(encoding="utf-8"))
    train_config = {
        "seed": 42,
        "model": {
            "family": "yolo",
            "weights": plan["initial_pretrained_weight"],
        },
        "data": {"root": "/fixture", "manifest": str(view_path)},
        "train": {
            "device": "cuda:0",
            "checkpoint_selection": "last",
            "args": {
                "imgsz": 1024,
                "batch": 12,
                "workers": 8,
                "optimizer": "AdamW",
                "lr0": 0.002,
                "lrf": 0.01,
                "weight_decay": 0.0005,
                "warmup_epochs": 3,
                "cos_lr": True,
                "amp": True,
                "deterministic": True,
                "patience": 0,
                "val": False,
                "plots": False,
            },
            "stages": [
                {
                    "name": "foundation",
                    "epochs": 160,
                    "balanced": False,
                    "args": {"close_mosaic": 20},
                }
            ],
        },
    }
    infer_config = {
        "model": {
            "adapter": "ultralytics",
            "family": "yolo",
            "checkpoint": str(fold_dir / "last.pt"),
            "imgsz": 1024,
            "confidence": 0.001,
            "iou": 0.69 if wrong_infer_iou else 0.70,
            "max_detections": 500,
            "half": True,
            "agnostic_nms": False,
        },
        "device": "cuda:0",
        "batch_size": 8,
        "input": {
            "data_root": "/fixture",
            "manifest": str(view_path),
            "split": "val",
        },
        "tiling": {"enabled": False, "force": False},
        "score_thresholds": {
            "ship": 0.001,
            "aircraft": 0.001,
            "vehicle": 0.001,
        },
        "fine_score_thresholds": {},
        "output_json": str(
            fold_dir / ("wrong_predictions.json" if wrong_infer_output else "predictions_low.json")
        ),
        "evaluation": {
            "gt": "",
            "project_config": "configs/project.yaml",
            "metrics_output": "",
        },
    }
    train_path = fold_dir / "resolved_train.yaml"
    infer_path = fold_dir / "resolved_infer.yaml"
    train_path.write_text(yaml.safe_dump(train_config), encoding="utf-8")
    infer_path.write_text(yaml.safe_dump(infer_config), encoding="utf-8")
    (fold_dir / "environment.txt").write_text("fixture", encoding="utf-8")
    (fold_dir / "last.pt").write_bytes(f"checkpoint-{fold}".encode())
    train_summary_path = fold_dir / "train_summary.json"
    train_summary_path.write_text(
        json.dumps(
            {
                "dry_run": dry_run_summary,
                "seed": 42,
                "model_family": "yolo",
                "initial_weights": plan["initial_pretrained_weight"],
                "checkpoint_selection": "last",
                "stages": [
                    {
                        "name": "foundation",
                        "balanced": False,
                        "input_weights": plan["initial_pretrained_weight"],
                        "arguments": {
                            "imgsz": 1024,
                            "batch": 12,
                            "workers": 8,
                            "optimizer": "AdamW",
                            "lr0": 0.002,
                            "lrf": 0.01,
                            "weight_decay": 0.0005,
                            "warmup_epochs": 3,
                            "cos_lr": True,
                            "amp": True,
                            "deterministic": True,
                            "patience": 0,
                            "val": False,
                            "plots": False,
                            "close_mosaic": 20,
                            "epochs": 160,
                            "device": "cuda:0",
                            "seed": 42,
                            "data": str(fold_dir / "prepared_data/data.yaml"),
                            "project": str(fold_dir / "training/runs"),
                            "name": "foundation",
                            "exist_ok": True,
                        },
                        "best": "",
                        "last": str((fold_dir / "last.pt").resolve()),
                        "selected_checkpoint": str((fold_dir / "last.pt").resolve()),
                        "checkpoint_selection": "last",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    view = json.loads(view_path.read_text(encoding="utf-8"))
    val_ids = [int(record["image_id"]) for record in view["samples"] if record["split"] == "val"]
    image_id = ((fold + 1) % 3) * 2 + 1 if wrong_prediction_fold else val_ids[0]
    prediction_path = fold_dir / "predictions_low.json"
    prediction_path.write_text(
        json.dumps(
            [
                {
                    "image_id": image_id,
                    "category_id": fold,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "score": 0.25,
                }
                for _ in range(prediction_count)
            ]
        ),
        encoding="utf-8",
    )
    runtime_path = fold_dir / "predictions_low.runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "schema_version": "rsdet_inference_runtime_v2",
                "images": len(val_ids),
                "artifacts": {
                    "config": {
                        "path": str(infer_path.resolve()),
                        "sha256": sha256_file(infer_path),
                    },
                    "checkpoint": {
                        "path": str((fold_dir / "last.pt").resolve()),
                        "sha256": (
                            "0" * 64
                            if wrong_runtime_checkpoint_sha
                            else sha256_file(fold_dir / "last.pt")
                        ),
                    },
                    "predictions": {
                        "path": str(
                            (
                                fold_dir / "wrong-runtime-predictions.json"
                                if wrong_runtime_predictions_path
                                else prediction_path
                            ).resolve()
                        ),
                        "sha256": sha256_file(prediction_path),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    data_lock_report = _write_data_lock_report(
        fold_dir / "detection_data_lock_verification.json",
        plan,
        overrides=data_lock_report_overrides,
    )
    finalize_fold_delivery(
        plan_path=run_root / "oof_run_plan.json",
        held_out_fold=fold,
        train_config_path=train_path,
        train_summary_path=train_summary_path,
        infer_config_path=infer_path,
        environment_path=fold_dir / "environment.txt",
        checkpoint_path=fold_dir / "last.pt",
        predictions_path=prediction_path,
        runtime_path=runtime_path,
        data_lock_verification_path=data_lock_report,
        output_path=fold_dir / "fold_metadata.json",
    )


def test_prepare_fold_views_are_loader_compatible(tmp_path: Path) -> None:
    manifest, run_root = _prepare(tmp_path)
    _, samples = load_cv3_manifest(
        manifest,
        expected_sha256=sha256_file(manifest),
        expected_image_count=6,
    )
    assert len(samples) == 6
    for fold in range(3):
        view = json.loads(
            (run_root / f"fold_{fold}" / "split_view.json").read_text(encoding="utf-8")
        )
        assert {record["split"] for record in view["samples"]} == {"train", "val"}
        assert sum(record["split"] == "val" for record in view["samples"]) == 2
        assert all(
            (record["source_fold"] == fold) == (record["split"] == "val")
            for record in view["samples"]
        )


def test_prepare_rejects_tampered_data_lock_without_creating_output(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "cv3.json")
    pretrained = tmp_path / "yolo26s.pt"
    pretrained.write_bytes(b"official-pretrained-fixture")
    data_lock = _data_lock(tmp_path / "formal_detection_data_lock.json", manifest)
    payload = json.loads(data_lock.read_text(encoding="utf-8"))
    payload["inventory"][0]["image_id"] = 999
    data_lock.write_text(json.dumps(payload), encoding="utf-8")
    run_root = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="lock_fingerprint"):
        prepare_oof_run_plan(
            manifest_path=manifest,
            output_dir=run_root,
            model_key="M1",
            model_family="yolo",
            model_name="yolo26s",
            seed=42,
            input_size=1024,
            foundation_epochs=160,
            low_score_threshold=0.001,
            max_detections=500,
            pretrained_weight=str(pretrained),
            pretrained_weight_sha256=sha256_file(pretrained),
            detection_data_lock=str(data_lock),
            detection_data_lock_sha256=sha256_file(data_lock),
            expected_manifest_sha256=sha256_file(manifest),
            expected_image_count=6,
        )
    assert not run_root.exists()


def test_prepare_and_finalize_clis_bind_detection_data_lock(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = _manifest(tmp_path / "cv3.json")
    pretrained = tmp_path / "yolo26s.pt"
    pretrained.write_bytes(b"official-pretrained-fixture")
    data_lock = _data_lock(tmp_path / "formal_detection_data_lock.json", manifest)
    run_root = tmp_path / "run"
    prepare = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/prepare_cv3_oof.py"),
            "--manifest",
            str(manifest),
            "--manifest-sha256",
            sha256_file(manifest),
            "--output-dir",
            str(run_root),
            "--model-key",
            "M1",
            "--model-family",
            "yolo",
            "--model-name",
            "yolo26s",
            "--input-size",
            "1024",
            "--foundation-epochs",
            "160",
            "--max-detections",
            "500",
            "--pretrained-weight",
            str(pretrained),
            "--pretrained-weight-sha256",
            sha256_file(pretrained),
            "--detection-data-lock",
            str(data_lock),
            "--detection-data-lock-sha256",
            sha256_file(data_lock),
            "--expected-image-count",
            "6",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
    )
    assert prepare.returncode == 0, prepare.stderr
    plan = json.loads((run_root / "oof_run_plan.json").read_text(encoding="utf-8"))
    assert plan["detection_data_lock"]["sha256"] == sha256_file(data_lock)

    _complete_fold(run_root, 0)
    fold_dir = run_root / "fold_0"
    (fold_dir / "fold_metadata.json").unlink()
    finalize = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/finalize_cv3_oof_fold.py"),
            "--plan",
            str(run_root / "oof_run_plan.json"),
            "--fold",
            "0",
            "--train-config",
            str(fold_dir / "resolved_train.yaml"),
            "--train-summary",
            str(fold_dir / "train_summary.json"),
            "--infer-config",
            str(fold_dir / "resolved_infer.yaml"),
            "--environment",
            str(fold_dir / "environment.txt"),
            "--checkpoint",
            str(fold_dir / "last.pt"),
            "--predictions",
            str(fold_dir / "predictions_low.json"),
            "--runtime",
            str(fold_dir / "predictions_low.runtime.json"),
            "--data-lock-verification",
            str(fold_dir / "detection_data_lock_verification.json"),
            "--output",
            str(fold_dir / "fold_metadata.json"),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
    )
    assert finalize.returncode == 0, finalize.stderr
    metadata = json.loads((fold_dir / "fold_metadata.json").read_text(encoding="utf-8"))
    assert metadata["artifacts"]["data_lock_verification_sha256"] == sha256_file(
        fold_dir / "detection_data_lock_verification.json"
    )


def test_checked_in_m1_templates_materialize_fixed_last_protocol(
    tmp_path: Path,
) -> None:
    _, run_root = _prepare(tmp_path)
    repository = Path(__file__).resolve().parents[1]
    fold_dir = run_root / "fold_0"
    data_root = tmp_path / "data"
    data_root.mkdir()
    train_request = fold_dir / "train_request.yaml"
    checkpoint = fold_dir / "training/runs/foundation/weights/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"last")
    infer_request = fold_dir / "resolved_infer.yaml"
    common = [
        sys.executable,
        str(repository / "scripts/materialize_cv3_oof_config.py"),
        "--fold",
        "0",
        "--data-root",
        str(data_root),
        "--split-view",
        str(fold_dir / "split_view.json"),
        "--fold-output-dir",
        str(fold_dir),
    ]
    subprocess.run(
        [
            *common,
            "--template",
            str(repository / "configs/experiments/m1_yolo26s_1024_cv3_oof.template.yaml"),
            "--output",
            str(train_request),
            "--pretrained-weight",
            str(tmp_path / "yolo26s.pt"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            *common,
            "--template",
            str(repository / "configs/experiments/m1_yolo26s_1024_cv3_oof_infer.template.yaml"),
            "--output",
            str(infer_request),
            "--checkpoint",
            str(checkpoint),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    train_config = yaml.safe_load(train_request.read_text(encoding="utf-8"))
    infer_config = yaml.safe_load(infer_request.read_text(encoding="utf-8"))
    assert train_config["train"]["checkpoint_selection"] == "last"
    assert train_config["train"]["args"]["val"] is False
    assert train_config["train"]["args"]["patience"] == 0
    assert infer_config["model"]["checkpoint"] == str(checkpoint.resolve())
    assert infer_config["model"]["iou"] == pytest.approx(0.70)
    assert infer_config["tiling"] == {"enabled": False, "force": False}


def test_checked_in_m3_templates_freeze_fixed_last_and_full_inference() -> None:
    repository = Path(__file__).resolve().parents[1]
    train_config = yaml.safe_load(
        (repository / "configs/experiments/m3_rtdetr_l_1024_cv3_oof.template.yaml").read_text(
            encoding="utf-8"
        )
    )
    infer_config = yaml.safe_load(
        (repository / "configs/experiments/m3_rtdetr_l_1024_cv3_oof_infer.template.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert train_config["train"]["checkpoint_selection"] == "last"
    assert train_config["train"]["args"]["val"] is False
    assert train_config["train"]["args"]["patience"] == 0
    assert infer_config["model"]["checkpoint"] == "__FOLD_LAST_CHECKPOINT__"
    assert infer_config["model"]["agnostic_nms"] is False
    assert infer_config["model"]["max_detections"] == 300
    assert infer_config["tiling"] == {"enabled": False, "force": False}


def test_finalize_rejects_prediction_from_another_fold(tmp_path: Path) -> None:
    _, run_root = _prepare(tmp_path)
    with pytest.raises(ValueError, match="非本折"):
        _complete_fold(run_root, 0, wrong_prediction_fold=True)


def test_finalize_rejects_mismatched_data_lock_verification_report(
    tmp_path: Path,
) -> None:
    _, run_root = _prepare(tmp_path)
    with pytest.raises(ValueError, match="verify 报告与计划不一致"):
        _complete_fold(
            run_root,
            0,
            data_lock_report_overrides={"inventory_fingerprint": "0" * 64},
        )


def test_finalize_rejects_wrong_training_manifest(tmp_path: Path) -> None:
    _, run_root = _prepare(tmp_path)
    wrong_view = run_root / "fold_1" / "split_view.json"
    fold_dir = run_root / "fold_0"
    plan = json.loads((run_root / "oof_run_plan.json").read_text(encoding="utf-8"))
    config = {
        "seed": 42,
        "model": {
            "family": "yolo",
            "weights": plan["initial_pretrained_weight"],
        },
        "data": {"root": "/fixture", "manifest": str(wrong_view)},
        "train": {
            "device": plan["training_config_contract"]["device"],
            "checkpoint_selection": plan["training_config_contract"]["checkpoint_selection"],
            "args": plan["training_config_contract"]["train_args"],
            "stages": [
                {
                    "name": "foundation",
                    "epochs": 160,
                    "balanced": False,
                    "args": plan["training_config_contract"]["stage_args"],
                }
            ],
        },
    }
    train_path = fold_dir / "resolved_train.yaml"
    train_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    infer_path = fold_dir / "resolved_infer.yaml"
    infer_path.write_text("{}", encoding="utf-8")
    for path in (
        fold_dir / "environment.txt",
        fold_dir / "train_summary.json",
        fold_dir / "last.pt",
        fold_dir / "predictions_low.json",
        fold_dir / "runtime.json",
    ):
        path.write_text("[]" if path.suffix == ".json" else "x", encoding="utf-8")
    data_lock_report = _write_data_lock_report(
        fold_dir / "detection_data_lock_verification.json",
        plan,
    )
    with pytest.raises(ValueError, match="本折冻结 split view"):
        finalize_fold_delivery(
            plan_path=run_root / "oof_run_plan.json",
            held_out_fold=0,
            train_config_path=train_path,
            train_summary_path=fold_dir / "train_summary.json",
            infer_config_path=infer_path,
            environment_path=fold_dir / "environment.txt",
            checkpoint_path=fold_dir / "last.pt",
            predictions_path=fold_dir / "predictions_low.json",
            runtime_path=fold_dir / "runtime.json",
            data_lock_verification_path=data_lock_report,
            output_path=fold_dir / "fold_metadata.json",
        )


def test_finalize_rejects_unregistered_inference_parameter_drift(
    tmp_path: Path,
) -> None:
    _, run_root = _prepare(tmp_path)
    with pytest.raises(ValueError, match="冻结完整配置合同"):
        _complete_fold(run_root, 0, wrong_infer_iou=True)


def test_complete_oof_includes_zero_prediction_images(tmp_path: Path) -> None:
    manifest, run_root = _prepare(tmp_path)
    for fold in range(3):
        _complete_fold(run_root, fold)
    output = tmp_path / "aggregate"
    metadata = audit_and_aggregate_oof(
        manifest_path=manifest,
        plan_path=run_root / "oof_run_plan.json",
        run_root=run_root,
        output_dir=output,
        expected_manifest_sha256=sha256_file(manifest),
        expected_image_count=6,
        diagnostic_without_formal_crop=True,
    )
    assert metadata["status"] == "diagnostic_only_no_formal_crop"
    assert metadata["downstream_admission"] is False
    assert metadata["image_count"] == 6
    assert metadata["proposal_count"] == 3
    with (output / "oof_images.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert sum(int(row["prediction_count"]) == 0 for row in rows) == 3
    with (output / "oof_proposals.csv").open(encoding="utf-8", newline="") as handle:
        proposals = list(csv.DictReader(handle))
    assert len(proposals) == 3
    assert {int(row["fold"]) for row in proposals} == {0, 1, 2}


def test_oof_aggregate_rechecks_checkpoint_entity_and_sha(tmp_path: Path) -> None:
    manifest, run_root = _prepare(tmp_path)
    for fold in range(3):
        _complete_fold(run_root, fold)
    (run_root / "fold_1" / "last.pt").write_bytes(b"tampered-after-finalize")
    with pytest.raises(ValueError, match="fold 1 checkpoint SHA 不一致"):
        audit_and_aggregate_oof(
            manifest_path=manifest,
            plan_path=run_root / "oof_run_plan.json",
            run_root=run_root,
            output_dir=tmp_path / "aggregate",
            expected_manifest_sha256=sha256_file(manifest),
            expected_image_count=6,
            diagnostic_without_formal_crop=True,
        )


def test_oof_aggregate_rejects_plan_changed_after_finalize(tmp_path: Path) -> None:
    manifest, run_root = _prepare(tmp_path)
    for fold in range(3):
        _complete_fold(run_root, fold)
    plan_path = run_root / "oof_run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["max_detections"] = 499
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="run plan SHA 不一致"):
        audit_and_aggregate_oof(
            manifest_path=manifest,
            plan_path=plan_path,
            run_root=run_root,
            output_dir=tmp_path / "aggregate",
            expected_manifest_sha256=sha256_file(manifest),
            expected_image_count=6,
            diagnostic_without_formal_crop=True,
        )


def test_formal_aggregate_requires_exact_frozen_crop(tmp_path: Path) -> None:
    manifest, run_root = _prepare(tmp_path)
    for fold in range(3):
        _complete_fold(run_root, fold)
    with pytest.raises(ValueError, match="必须一次性绑定"):
        audit_and_aggregate_oof(
            manifest_path=manifest,
            plan_path=run_root / "oof_run_plan.json",
            run_root=run_root,
            output_dir=tmp_path / "missing-formal",
            expected_manifest_sha256=sha256_file(manifest),
            expected_image_count=6,
        )
    fake_formal = tmp_path / "fake-formal.csv"
    fake_formal.write_text("manifest_version\nformal_crop_manifest_v2\n")
    with pytest.raises(ValueError, match="唯一冻结"):
        audit_and_aggregate_oof(
            manifest_path=manifest,
            plan_path=run_root / "oof_run_plan.json",
            run_root=run_root,
            output_dir=tmp_path / "wrong-formal",
            expected_manifest_sha256=sha256_file(manifest),
            expected_image_count=6,
            formal_crop_manifest_path=fake_formal,
        )


def test_finalize_rejects_more_than_model_max_per_image(tmp_path: Path) -> None:
    _, run_root = _prepare(tmp_path)
    with pytest.raises(ValueError, match="proposal_count 超出"):
        _complete_fold(run_root, 0, prediction_count=501)


def test_finalize_binds_infer_output_and_real_train_summary(tmp_path: Path) -> None:
    _, run_root = _prepare(tmp_path)
    with pytest.raises(ValueError, match="output_json.*不是同一实体"):
        _complete_fold(run_root, 0, wrong_infer_output=True)

    second = tmp_path / "second"
    second.mkdir()
    _, second_root = _prepare(second)
    with pytest.raises(ValueError, match="dry_run=false"):
        _complete_fold(second_root, 0, dry_run_summary=True)

    third = tmp_path / "third"
    third.mkdir()
    _, third_root = _prepare(third)
    with pytest.raises(ValueError, match="runtime checkpoint SHA"):
        _complete_fold(third_root, 0, wrong_runtime_checkpoint_sha=True)

    fourth = tmp_path / "fourth"
    fourth.mkdir()
    _, fourth_root = _prepare(fourth)
    with pytest.raises(ValueError, match="runtime predictions path"):
        _complete_fold(fourth_root, 0, wrong_runtime_predictions_path=True)


def test_aggregate_rejects_out_of_bounds_bbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, run_root = _prepare(tmp_path)
    for fold in range(3):
        _complete_fold(run_root, fold)
    prediction_path = run_root / "fold_0" / "predictions_low.json"
    records = json.loads(prediction_path.read_text(encoding="utf-8"))
    records[0]["bbox"] = [9.0, 1.0, 2.0, 2.0]
    prediction_path.write_text(json.dumps(records), encoding="utf-8")
    runtime_path = run_root / "fold_0" / "predictions_low.runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["artifacts"]["predictions"]["sha256"] = sha256_file(prediction_path)
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    metadata_path = run_root / "fold_0" / "fold_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"]["predictions_sha256"] = sha256_file(prediction_path)
    metadata["artifacts"]["runtime_sha256"] = sha256_file(runtime_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    formal = tmp_path / "formal.csv"
    formal.write_text("test", encoding="utf-8")
    monkeypatch.setattr(
        cv3_oof_module,
        "_load_formal_crop_image_sizes",
        lambda _: {image_id: (10, 10) for image_id in range(1, 7)},
    )
    with pytest.raises(ValueError, match="超出图像宽度"):
        audit_and_aggregate_oof(
            manifest_path=manifest,
            plan_path=run_root / "oof_run_plan.json",
            run_root=run_root,
            output_dir=tmp_path / "aggregate",
            expected_manifest_sha256=sha256_file(manifest),
            expected_image_count=6,
            formal_crop_manifest_path=formal,
        )
