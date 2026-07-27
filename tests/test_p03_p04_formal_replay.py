"""P03/P04 正式复验输入、cache 复用与 train-only 变换测试。"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from rsdet.analysis.formal_replay import (
    FORMAL_CROP_VERSION,
    audit_cache_reuse,
    audit_formal_crop_manifest,
    class_recall_support,
    fit_train_rms,
    pooled_classification_metrics,
    sha256_file,
)
from rsdet.features.p04_cache import FeatureCacheWriter
from rsdet.features.p04_inputs import (
    D4_VIEW_IDS,
    canonical_image_sha256,
    load_all_crop_records,
    render_canonical_crop,
)
from rsdet.features.p04_probe import load_probe_data

FIELDS = (
    "manifest_version",
    "schema_version",
    "crop_id",
    "annotation_uid",
    "source_image_id",
    "source_relative_path",
    "source_width",
    "source_height",
    "source_checksum_sha256",
    "class_id",
    "class_name",
    "major_class",
    "crop_policy",
    "crop_x0",
    "crop_y0",
    "crop_x1",
    "crop_y1",
    "fold",
    "group_id",
    "leakage_group_id",
    "color_mode",
    "outside_policy",
    "resize_semantics",
)
HISTORICAL_TEST_FIELDS = {"fold", "group_id", "leakage_group_id"}
FORMAL_FIELDS = tuple(
    f"historical_p02_{field}" if field in HISTORICAL_TEST_FIELDS else field
    for field in FIELDS
) + ("fold", "group_id", "leakage_group_id")


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: tuple[str, ...] = FIELDS,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    image_dir = tmp_path / "images" / "train"
    image_dir.mkdir(parents=True)
    samples = []
    exploratory_rows: list[dict[str, object]] = []
    formal_rows: list[dict[str, object]] = []
    policies = ("tight", "context_1p25", "jitter_light")
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        name = f"sample-{index}.png"
        Image.new("RGB", (4, 4), color).save(image_dir / name)
        samples.append(
            {
                "image_id": index + 1,
                "relative_path": f"images/train/{name}",
                "fold": index,
                "group_id": f"formal-group-{index}",
                "group_rule": "fixture",
            }
        )
        for policy in policies:
            base: dict[str, object] = {
                "manifest_version": "exploratory_crop_manifest_v1",
                "schema_version": "crop_manifest_v1",
                "crop_id": f"crop-{index}-{policy}",
                "annotation_uid": f"annotation-{index}",
                "source_image_id": f"image-{index}",
                "source_relative_path": f"images/train/{name}",
                "source_width": 4,
                "source_height": 4,
                "source_checksum_sha256": str(index) * 64,
                "class_id": 0,
                "class_name": "HM",
                "major_class": "ship",
                "crop_policy": policy,
                "crop_x0": 0,
                "crop_y0": 0,
                "crop_x1": 4,
                "crop_y1": 4,
                "fold": (index + 1) % 3,
                "group_id": f"old-group-{index}",
                "leakage_group_id": f"old-group-{index}",
                "color_mode": "RGB",
                "outside_policy": "pad",
                "resize_semantics": "direct_square_resize",
            }
            exploratory_rows.append(base)
            formal_row = {
                key: value
                for key, value in base.items()
                if key not in HISTORICAL_TEST_FIELDS
            }
            formal_row.update(
                {
                    "manifest_version": FORMAL_CROP_VERSION,
                    "schema_version": FORMAL_CROP_VERSION,
                    "historical_p02_fold": base["fold"],
                    "historical_p02_group_id": base["group_id"],
                    "historical_p02_leakage_group_id": base["leakage_group_id"],
                    "fold": index,
                    "group_id": f"formal-group-{index}",
                    "leakage_group_id": f"formal-group-{index}",
                }
            )
            formal_rows.append(formal_row)
    exploratory = tmp_path / "exploratory.csv"
    formal = tmp_path / "formal.csv"
    _write_csv(exploratory, exploratory_rows)
    _write_csv(formal, formal_rows, fieldnames=FORMAL_FIELDS)
    cv3 = tmp_path / "cv3.json"
    cv3.write_text(
        json.dumps(
            {
                "version": "cv3_airport_proxy_k60_v2",
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    return exploratory, formal, cv3, tmp_path


def _write_cache(
    cache_dir: Path,
    formal: Path,
    data_root: Path,
    *,
    extraction_manifest_sha256: str,
    asset_lock: Path,
) -> str:
    records = load_all_crop_records(formal, crop_policy="tight")
    rows = {
        "annotation_uid": [],
        "crop_id": [],
        "canonical_input_sha256": [],
        "view_id": [],
    }
    features: list[list[float]] = []
    for index, record in enumerate(records):
        canonical = render_canonical_crop(record, data_root, resolution=224)
        input_sha = canonical_image_sha256(canonical)
        for view_index, view in enumerate(D4_VIEW_IDS):
            rows["annotation_uid"].append(record.annotation_uid)
            rows["crop_id"].append(record.crop_id)
            rows["canonical_input_sha256"].append(input_sha)
            rows["view_id"].append(view)
            features.append([float(index), float(view_index)])
    writer = FeatureCacheWriter(
        cache_dir,
        metadata={
            "manifest_sha256": extraction_manifest_sha256,
            "object_count": 3,
            "view_ids": list(D4_VIEW_IDS),
            "asset_lock_sha256": sha256_file(asset_lock),
            "teacher": {"teacher_id": "fixture_teacher"},
        },
        feature_names=("fixture_feature",),
        storage_dtype="float32",
    )
    writer.write_shard(
        0,
        rows,
        {"fixture_feature": np.asarray(features, dtype=np.float32)},
    )
    writer.finalize(expected_shards=1, expected_rows=24)
    return writer.config_fingerprint


def test_formal_manifest_only_changes_split_fields(tmp_path: Path) -> None:
    exploratory, formal, cv3, _ = _fixture(tmp_path)
    audit = audit_formal_crop_manifest(
        formal,
        exploratory,
        cv3,
        expected_cv3_sha256=sha256_file(cv3),
        expected_object_count=3,
        expected_cv3_sample_count=3,
    )
    assert audit["status"] == "formal_crop_manifest_gate_pass"
    assert audit["tight_fold_object_counts"] == [1, 1, 1]
    assert audit["immutable_mismatch_count"] == 0

    changed = tmp_path / "changed.csv"
    text = formal.read_text(encoding="utf-8").replace("crop-0-tight", "changed-crop")
    changed.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="不可变字段"):
        audit_formal_crop_manifest(
            changed,
            exploratory,
            cv3,
            expected_cv3_sha256=sha256_file(cv3),
            expected_object_count=3,
            expected_cv3_sample_count=3,
        )


def test_cache_reuse_requires_uid_crop_and_rendered_canonical_sha(tmp_path: Path) -> None:
    exploratory, formal, _, data_root = _fixture(tmp_path)
    cache_dir = tmp_path / "cache"
    asset_lock = tmp_path / "ASSET_LOCK.json"
    asset_lock.write_text('{"fixture":true}\n', encoding="utf-8")
    fingerprint = _write_cache(
        cache_dir,
        formal,
        data_root,
        extraction_manifest_sha256=sha256_file(exploratory),
        asset_lock=asset_lock,
    )
    cache_audit = audit_cache_reuse(
        formal,
        data_root,
        {"fixture": cache_dir},
        expected_object_count=3,
        expected_fingerprint_prefixes={"fixture": fingerprint[:8]},
        expected_teacher_ids={"fixture": "fixture_teacher"},
        asset_lock_path=asset_lock,
    )
    item = cache_audit["fixture"]
    assert item["canonical_mismatch_count"] == 0
    assert item["object_count"] == 3

    formal_audit = {
        "status": "formal_replay_inputs_ready",
        "formal_manifest_sha256": sha256_file(formal),
        "caches": cache_audit,
    }
    audit_path = tmp_path / "input_audit.json"
    audit_path.write_text(json.dumps(formal_audit), encoding="utf-8")
    data = load_probe_data(
        cache_dir,
        feature_name="fixture_feature",
        manifest_path=formal,
        reuse_audit_path=audit_path,
    )
    assert data.folds.tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="P04-TASK-05"):
        load_probe_data(
            cache_dir,
            feature_name="fixture_feature",
            manifest_path=formal,
        )


def test_cache_reuse_rejects_pixel_drift(tmp_path: Path) -> None:
    exploratory, formal, _, data_root = _fixture(tmp_path)
    cache_dir = tmp_path / "cache"
    asset_lock = tmp_path / "ASSET_LOCK.json"
    asset_lock.write_text('{"fixture":true}\n', encoding="utf-8")
    fingerprint = _write_cache(
        cache_dir,
        formal,
        data_root,
        extraction_manifest_sha256=sha256_file(exploratory),
        asset_lock=asset_lock,
    )
    Image.new("RGB", (4, 4), (1, 2, 3)).save(data_root / "images/train/sample-0.png")
    with pytest.raises(ValueError, match="不可复用"):
        audit_cache_reuse(
            formal,
            data_root,
            {"fixture": cache_dir},
            expected_object_count=3,
            expected_fingerprint_prefixes={"fixture": fingerprint[:8]},
            expected_teacher_ids={"fixture": "fixture_teacher"},
            asset_lock_path=asset_lock,
        )


def test_train_rms_is_fit_only_from_given_training_tensor() -> None:
    train = np.asarray([[[1.0, 3.0]], [[5.0, 7.0]]], dtype=np.float32)
    validation_a = np.full((2, 1, 2), 10.0, dtype=np.float32)
    validation_b = np.full((2, 1, 2), 1e9, dtype=np.float32)
    rms_a = fit_train_rms(train)
    rms_b = fit_train_rms(train)
    assert rms_a == pytest.approx(np.sqrt(np.mean(train.astype(np.float64) ** 2)))
    assert rms_a == rms_b
    # 验证值再极端，也不参与 normalizer 的拟合。
    assert not np.isclose(np.sqrt(np.mean(validation_a**2)), rms_a)
    assert not np.isclose(np.sqrt(np.mean(validation_b**2)), rms_a)


def test_formal_training_clis_reject_unfrozen_runtime_before_gpu_work(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    environment = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    p03 = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_crop_classifier.py"),
            "--config",
            str(repo_root / "configs" / "experiments" / "p03_formal_cv3_v2.yaml"),
            "--manifest",
            str(tmp_path / "missing.csv"),
            "--data-root",
            str(tmp_path),
            "--weights",
            str(tmp_path / "missing.pth"),
            "--output-dir",
            str(tmp_path / "p03"),
            "--fold",
            "0",
            "--policy",
            "tight",
            "--resolution",
            "224",
            "--regime",
            "fine_tune",
            "--epochs",
            "29",
        ],
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert p03.returncode != 0
    assert "P03 formal CLI 非冻结值" in p03.stderr

    p04 = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_p04_feature_probe.py"),
            "--cache-dir",
            str(tmp_path / "missing-cache"),
            "--feature-name",
            "convnext_gap",
            "--manifest",
            str(tmp_path / "missing.csv"),
            "--output-dir",
            str(tmp_path / "p04"),
            "--fold",
            "0",
            "--reuse-audit",
            str(tmp_path / "missing-audit.json"),
        ],
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert p04.returncode != 0
    assert "P04 formal CLI 非冻结值" in p04.stderr


def test_pooled_diagnostics_include_fixed_tiers_major_classes_and_tu160() -> None:
    labels = np.arange(25, dtype=np.int64)
    logits = np.full((25, 25), -2.0, dtype=np.float32)
    logits[np.arange(25), labels] = 2.0
    pooled = pooled_classification_metrics(
        [logits[:9], logits[9:17], logits[17:]],
        [labels[:9], labels[9:17], labels[17:]],
    )
    assert pooled["n_samples"] == 25
    assert len(pooled["per_class"]) == 25
    assert set(pooled["major_classes"]) == {"ship4", "aircraft20", "vehicle1"}
    assert [len(pooled["support_tiers"][tier]["class_ids"]) for tier in ("tail", "middle", "head")] == [
        9,
        8,
        8,
    ]
    tu = class_recall_support(logits, labels, 9)
    assert tu == {"class_id": 9, "support": 1, "true_positive": 1, "recall": 1.0}


def _perfect_fold_outputs() -> tuple[np.ndarray, np.ndarray]:
    labels = np.arange(25, dtype=np.int64)
    logits = np.full((25, 25), -2.0, dtype=np.float32)
    logits[np.arange(25), labels] = 2.0
    return logits, labels


def _write_formal_summary_fixture(path: Path) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "annotation_uid",
                "crop_id",
                "crop_policy",
                "fold",
                "class_id",
            ),
        )
        writer.writeheader()
        for fold in range(3):
            for class_id in range(25):
                writer.writerow(
                    {
                        "annotation_uid": f"fold{fold}-class{class_id}",
                        "crop_id": f"crop-fold{fold}-class{class_id}",
                        "crop_policy": "tight",
                        "fold": fold,
                        "class_id": class_id,
                    }
                )
    return sha256_file(path)


def _write_perfect_predictions(path: Path, fold: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "annotation_uid",
                "crop_id",
                "true_class_id",
                "pred_class_id",
            ),
        )
        writer.writeheader()
        for class_id in range(25):
            writer.writerow(
                {
                    "annotation_uid": f"fold{fold}-class{class_id}",
                    "crop_id": f"crop-fold{fold}-class{class_id}",
                    "true_class_id": class_id,
                    "pred_class_id": class_id,
                }
            )


def _run_formal_summary(
    repo_root: Path,
    *,
    stage: str,
    runs: Path,
    audit: Path,
    output: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "summarize_p03_p04_formal.py"),
            "--stage",
            stage,
            "--runs-root",
            str(runs),
            "--input-audit",
            str(audit),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_p03_formal_summary_cli_enforces_and_pools_exact_three_folds(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    runs = tmp_path / "p03-runs"
    formal_manifest = tmp_path / "formal.csv"
    manifest_sha = _write_formal_summary_fixture(formal_manifest)
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "formal_replay_inputs_ready",
                "cross_fold_group_count": 0,
                "formal_manifest": str(formal_manifest),
                "formal_manifest_sha256": manifest_sha,
                "cv3_manifest_sha256": "b" * 64,
                "tight_fold_object_counts": [25, 25, 25],
                "object_count": 75,
            }
        ),
        encoding="utf-8",
    )
    logits, labels = _perfect_fold_outputs()
    for fold in range(3):
        run = runs / f"fold{fold}"
        run.mkdir(parents=True)
        (run / "meta.json").write_text(
            json.dumps(
                {
                    "manifest_sha256": manifest_sha,
                    "weight_sha256": (
                        "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"
                    ),
                    "expected_weight_sha256": (
                        "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"
                    ),
                    "model_parameters": {
                        "total_parameters": 100,
                        "trainable_parameters": 100,
                    },
                }
            ),
            encoding="utf-8",
        )
        (run / "final_checkpoint.pt").write_bytes(f"p03-{fold}".encode())
        (run / "run_summary.json").write_text(
            json.dumps(
                {
                    "condition": {
                        "fold": fold,
                        "policy": "tight",
                        "resolution": 224,
                        "regime": "fine_tune",
                        "sampler": "natural",
                        "seed": 42,
                        "smoke": False,
                        "eval_only": False,
                    },
                    "n_train": 50,
                    "n_val": 25,
                    "checkpoint_selection": "fixed_epoch_last",
                    "selection_metric": "fixed_epoch_last",
                    "epochs_completed": 30,
                    "early_stopped": False,
                    "selected_checkpoint": "final_checkpoint.pt",
                    "selected_checkpoint_sha256": sha256_file(
                        run / "final_checkpoint.pt"
                    ),
                    "final_metrics": {
                        "macro_recall": 1.0,
                        "macro_f1": 1.0,
                        "accuracy": 1.0,
                    },
                    "aircraft20": {"macro_recall": 1.0},
                }
            ),
            encoding="utf-8",
        )
        (run / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "experiment_id": "P03-FORMAL-CV3-V2-CONVNEXT-T224",
                    "model": "convnext_tiny",
                    "pretraining": "ImageNet-1K V1",
                    "task": "all25",
                    "fold": fold,
                    "policy": "tight",
                    "resolution": 224,
                    "regime": "fine_tune",
                    "sampler": "natural",
                    "seed": 42,
                    "epochs": 30,
                    "batch_size": 96,
                    "num_workers": 8,
                    "smoke": False,
                    "eval_only": False,
                    "checkpoint_selection": "fixed_epoch_last",
                    "n_train": 50,
                    "n_val": 25,
                }
            ),
            encoding="utf-8",
        )
        np.savez_compressed(run / "validation_logits.npz", logits=logits, labels=labels)
        _write_perfect_predictions(run / "predictions.csv", fold)
    output = tmp_path / "p03-summary.json"
    result = _run_formal_summary(
        repo_root,
        stage="p03",
        runs=runs,
        audit=audit,
        output=output,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "p03_formal_cv3_v2_complete"
    assert summary["pooled_oof"]["n_samples"] == 75
    assert [item["support"] for item in summary["groups"][0]["tu160_by_fold"]] == [
        1,
        1,
        1,
    ]
    repeated = _run_formal_summary(
        repo_root,
        stage="p03",
        runs=runs,
        audit=audit,
        output=output,
    )
    assert repeated.returncode != 0
    assert "拒绝覆盖 formal summary" in repeated.stderr

    (runs / "fold0" / "final_checkpoint.pt").write_bytes(b"tampered")
    tampered_output = tmp_path / "p03-summary-tampered.json"
    tampered = _run_formal_summary(
        repo_root,
        stage="p03",
        runs=runs,
        audit=audit,
        output=tampered_output,
    )
    assert tampered.returncode != 0
    assert "final checkpoint 缺失或 SHA 不一致" in tampered.stderr
    assert not tampered_output.exists()


def test_p04_formal_summary_cli_enforces_teacher_cache_and_train_only_contract(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    runs = tmp_path / "p04-runs"
    formal_manifest = tmp_path / "formal.csv"
    manifest_sha = _write_formal_summary_fixture(formal_manifest)
    definitions = {
        "convnext_gap": ("convnext-fingerprint", 768),
        "dino_cls_patchmean": ("dino-fingerprint", 1536),
        "clean_map0": ("clean-fingerprint", 1280),
    }
    caches = {
        name: {
            "status": "pass",
            "cache_fingerprint": fingerprint,
            "feature_names": [feature],
            "cache_audit": {"feature_dimensions": {feature: dimension}},
            "canonical_mismatch_count": 0,
            "crop_id_mismatch_count": 0,
            "missing_annotation_count": 0,
            "unknown_annotation_count": 0,
            "incomplete_view_object_count": 0,
        }
        for name, (feature, (fingerprint, dimension)) in zip(
            ("convnext", "dinov2b", "cleandift"),
            definitions.items(),
            strict=True,
        )
    }
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "formal_replay_inputs_ready",
                "cross_fold_group_count": 0,
                "formal_manifest": str(formal_manifest),
                "formal_manifest_sha256": manifest_sha,
                "cv3_manifest_sha256": "b" * 64,
                "tight_fold_object_counts": [25, 25, 25],
                "object_count": 75,
                "cache_count": 3,
                "caches": caches,
            }
        ),
        encoding="utf-8",
    )
    audit_sha = sha256_file(audit)
    logits, labels = _perfect_fold_outputs()
    for feature_name, (fingerprint, native_dimension) in definitions.items():
        for pca_dim in (None, 384):
            for fold in range(3):
                run = runs / f"{feature_name}-pca{pca_dim}-fold{fold}"
                run.mkdir(parents=True)
                (run / "final_checkpoint.pt").write_bytes(
                    f"{feature_name}-{pca_dim}-{fold}".encode()
                )
                (run / "run_summary.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "feature_name": feature_name,
                            "cache_fingerprint": fingerprint,
                            "manifest_sha256": manifest_sha,
                            "fold": fold,
                            "seed": 42,
                            "normalization": "train_rms",
                            "pca_dim": pca_dim,
                            "pca_fit_scope": (
                                "train_fold_objects_all_cached_views_only"
                                if pca_dim
                                else None
                            ),
                            "train_rms": 0.5,
                            "normalizer_fit_scope": (
                                "train_fold_objects_all_cached_views_only"
                            ),
                            "head_init": "p04_default",
                            "reuse_audit": {"sha256": audit_sha},
                            "n_train": 50,
                            "n_val": 25,
                            "n_views": 8,
                            "ignored_cache_annotations": 0,
                            "feature_dimension": pca_dim or native_dimension,
                            "checkpoint_selection": "fixed_epoch_last",
                            "selection_metric": "fixed_epoch_last",
                            "completed_epochs": 15,
                            "early_stopped": False,
                            "selected_checkpoint": "final_checkpoint.pt",
                            "selected_checkpoint_sha256": sha256_file(
                                run / "final_checkpoint.pt"
                            ),
                            "metrics": {
                                "macro_recall": 1.0,
                                "macro_f1": 1.0,
                                "accuracy": 1.0,
                                "aircraft20_macro_recall": 1.0,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "resolved_config.json").write_text(
                    json.dumps(
                        {
                            "batch_size": 96,
                            "epochs": 15,
                            "minimum_epochs": 15,
                            "lr": 0.001,
                            "weight_decay": 0.01,
                            "warmup_epochs": 1.0,
                            "patience": 0,
                            "min_delta": 0.0,
                            "grad_clip": 1.0,
                            "policy": "tight",
                            "seed": 42,
                            "normalization": "train_rms",
                            "head_init": "p04_default",
                            "allow_cache_subset": False,
                            "checkpoint_selection": "fixed_epoch_last",
                            "feature_name": feature_name,
                            "fold": fold,
                            "pca_dim": pca_dim,
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "train_rms_train_only.json").write_text(
                    json.dumps(
                        {
                            "fit_scope": "train_fold_objects_all_cached_views_only",
                            "fold": fold,
                            "train_object_count": 50,
                            "view_count": 8,
                            "rms": 0.5,
                        }
                    ),
                    encoding="utf-8",
                )
                if pca_dim is not None:
                    np.savez_compressed(
                        run / "pca_train_only.npz",
                        components=np.zeros((384, native_dimension), dtype=np.float32),
                    )
                np.savez_compressed(
                    run / "validation_logits.npz",
                    logits=logits,
                    labels=labels,
                )
                _write_perfect_predictions(run / "predictions.csv", fold)
    output = tmp_path / "p04-summary.json"
    result = _run_formal_summary(
        repo_root,
        stage="p04",
        runs=runs,
        audit=audit,
        output=output,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "p04_formal_cv3_v2_complete"
    assert len(summary["groups"]) == 6
    assert len(summary["pooled_oof"]) == 6
    assert {item["n_samples"] for item in summary["pooled_oof"].values()} == {75}
    (runs / "convnext_gap-pcaNone-fold0" / "final_checkpoint.pt").write_bytes(
        b"tampered"
    )
    tampered_output = tmp_path / "p04-summary-tampered.json"
    tampered = _run_formal_summary(
        repo_root,
        stage="p04",
        runs=runs,
        audit=audit,
        output=tampered_output,
    )
    assert tampered.returncode != 0
    assert "final checkpoint 缺失或 SHA 不一致" in tampered.stderr
    assert not tampered_output.exists()
