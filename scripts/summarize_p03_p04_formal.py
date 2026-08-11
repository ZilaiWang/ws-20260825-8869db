#!/usr/bin/env python3
"""P03/P04 正式 CV3 v2 复验的完整性门禁、三折与 pooled OOF 汇总。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from rsdet.analysis.formal_replay import (
    aggregate_metric_rows,
    class_recall_support,
    pooled_classification_metrics,
    sha256_file,
)
from rsdet.evaluation.classification import evaluate_classification

P04_EXPECTED = {
    ("convnext_gap", None),
    ("convnext_gap", 384),
    ("dino_cls_patchmean", None),
    ("dino_cls_patchmean", 384),
    ("clean_map0", None),
    ("clean_map0", 384),
}
P04_NATIVE_DIMENSIONS = {
    "convnext_gap": 768,
    "dino_cls_patchmean": 1536,
    "clean_map0": 1280,
}
P03_WEIGHT_SHA256 = "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"
P04_FROZEN_RUNTIME = {
    "batch_size": 96,
    "epochs": 15,
    "minimum_epochs": 15,
    "lr": 0.001,
    "weight_decay": 0.01,
    "warmup_epochs": 1.0,
    "patience": 0,
    "min_delta": 0.0,
    "grad_clip": 1.0,
    "checkpoint_selection": "fixed_epoch_last",
    "policy": "tight",
    "seed": 42,
    "normalization": "train_rms",
    "head_init": "p04_default",
    "allow_cache_subset": False,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="summarize P03/P04 formal CV3 replay")
    parser.add_argument("--stage", required=True, choices=("p03", "p04"))
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--input-audit", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _load_input_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "formal_replay_inputs_ready":
        raise ValueError("formal input audit 未通过")
    if payload.get("cross_fold_group_count") != 0:
        raise ValueError("formal input audit 仍有跨折 group")
    return payload


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["logits"], dtype=np.float64),
            np.asarray(payload["labels"], dtype=np.int64),
        )


def _formal_object_contract(
    audit: dict[str, Any],
) -> dict[str, tuple[int, int, str]]:
    path = Path(audit["formal_manifest"]).expanduser().resolve()
    if sha256_file(path) != audit["formal_manifest_sha256"]:
        raise ValueError("formal manifest 在输入审计后发生变化")
    result: dict[str, tuple[int, int, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"annotation_uid", "crop_id", "crop_policy", "fold", "class_id"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"formal manifest 缺列: {sorted(missing)}")
        for row in reader:
            if row["crop_policy"].strip() != "tight":
                continue
            uid = row["annotation_uid"].strip()
            if not uid or uid in result:
                raise ValueError("formal tight annotation_uid 为空或重复")
            result[uid] = (
                int(row["fold"]),
                int(row["class_id"]),
                row["crop_id"].strip(),
            )
    if len(result) != int(audit["object_count"]):
        raise ValueError("formal tight 对象数与输入审计不一致")
    fold_counts = [sum(contract[0] == fold for contract in result.values()) for fold in range(3)]
    if fold_counts != list(audit["tight_fold_object_counts"]):
        raise ValueError("formal tight fold 对象数与输入审计不一致")
    return result


def _validate_fold_predictions(
    run: Path,
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    fold: int,
    expected_objects: dict[str, tuple[int, int, str]],
) -> None:
    path = run / "predictions.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"annotation_uid", "crop_id", "true_class_id", "pred_class_id"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} 缺列: {sorted(missing)}")
        rows = list(reader)
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(logits)
    if scores.shape != (len(rows), 25) or truth.shape != (len(rows),):
        raise ValueError(f"{run} logits/labels/predictions 行数或维度不一致")
    expected_uids = {uid for uid, contract in expected_objects.items() if contract[0] == fold}
    actual_uids = [row["annotation_uid"].strip() for row in rows]
    if len(actual_uids) != len(set(actual_uids)) or set(actual_uids) != expected_uids:
        raise ValueError(f"{run} OOF UID 不完整、重复或跨折")
    predicted = scores.argmax(axis=1)
    for index, row in enumerate(rows):
        uid = actual_uids[index]
        expected_fold, expected_class, expected_crop = expected_objects[uid]
        if (
            expected_fold != fold
            or row["crop_id"].strip() != expected_crop
            or int(row["true_class_id"]) != expected_class
            or int(truth[index]) != expected_class
            or int(row["pred_class_id"]) != int(predicted[index])
        ):
            raise ValueError(f"{run} predictions/logits/formal manifest 未逐行对齐")


def _validate_reported_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    reported: dict[str, Any],
    *,
    label: str,
) -> None:
    recomputed, _ = evaluate_classification(labels, logits)
    expected = {
        "macro_recall": recomputed["macro_recall"],
        "macro_f1": recomputed["macro_f1"],
        "accuracy": recomputed["accuracy"],
        "aircraft20_macro_recall": recomputed["subgroups"]["aircraft20"]["macro_recall"],
    }
    mismatches = {
        key: {"reported": reported.get(key), "recomputed": value}
        for key, value in expected.items()
        if not np.isclose(float(reported.get(key, np.nan)), float(value), atol=1e-12)
    }
    if mismatches:
        raise ValueError(f"{label} 指标与保存 logits 独立复算不一致: {mismatches}")


def _require_mapping_values(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} 非冻结值: {mismatches}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖 formal summary: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"拒绝覆盖 formal summary: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _p03(root: Path, audit: dict[str, Any]) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    fold_outputs: list[tuple[int, np.ndarray, np.ndarray]] = []
    expected_objects = _formal_object_contract(audit)
    tu160_total = sum(contract[1] == 9 for contract in expected_objects.values())
    if int(audit["object_count"]) == 20_933 and tu160_total != 361:
        raise ValueError(f"formal manifest 的 TU-160 support 异常: {tu160_total}")
    for path in sorted(root.rglob("run_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        condition = summary["condition"]
        if condition.get("smoke") or condition.get("eval_only"):
            continue
        expected = {
            "policy": "tight",
            "resolution": 224,
            "regime": "fine_tune",
            "sampler": "natural",
            "seed": 42,
        }
        if any(condition.get(key) != value for key, value in expected.items()):
            raise ValueError(f"P03 formal 出现未预注册条件: {path}")
        meta_path = path.parent / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("manifest_sha256") != audit["formal_manifest_sha256"]:
            raise ValueError(f"P03 run manifest SHA 不匹配: {path.parent}")
        if (
            meta.get("weight_sha256") != P03_WEIGHT_SHA256
            or meta.get("expected_weight_sha256") != P03_WEIGHT_SHA256
        ):
            raise ValueError(f"P03 ImageNet 权重 SHA 不匹配: {path.parent}")
        parameters = meta.get("model_parameters") or {}
        if int(parameters.get("total_parameters", -1)) <= 0 or parameters.get(
            "trainable_parameters"
        ) != parameters.get("total_parameters"):
            raise ValueError(f"P03 formal 不是全参数微调: {path.parent}")
        fold = int(condition["fold"])
        if int(summary["n_val"]) != int(audit["tight_fold_object_counts"][fold]):
            raise ValueError(f"P03 fold{fold} n_val 与 formal audit 不一致")
        if int(summary["n_train"]) != int(audit["object_count"]) - int(summary["n_val"]):
            raise ValueError(f"P03 fold{fold} n_train 与 formal audit 不一致")
        if (
            summary.get("checkpoint_selection") != "fixed_epoch_last"
            or summary.get("selection_metric") != "fixed_epoch_last"
            or summary.get("early_stopped") is not False
            or int(summary.get("epochs_completed", -1)) != 30
            or summary.get("selected_checkpoint") != "final_checkpoint.pt"
        ):
            raise ValueError(f"P03 fold{fold} 使用了 held-out 选 best/早停，而非固定 epoch")
        selected_checkpoint = path.parent / "final_checkpoint.pt"
        if not selected_checkpoint.is_file() or sha256_file(selected_checkpoint) != summary.get(
            "selected_checkpoint_sha256"
        ):
            raise ValueError(f"P03 fold{fold} final checkpoint 缺失或 SHA 不一致")
        resolved_path = path.parent / "resolved_config.yaml"
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        _require_mapping_values(
            resolved,
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
                "n_train": int(summary["n_train"]),
                "n_val": int(summary["n_val"]),
            },
            label=f"P03 resolved config {path.parent}",
        )
        values.append(
            {
                "fold": fold,
                "metrics": {
                    **summary["final_metrics"],
                    "aircraft20_macro_recall": summary["aircraft20"]["macro_recall"],
                },
            }
        )
        fold_logits, fold_labels = _load_npz(path.parent / "validation_logits.npz")
        _validate_fold_predictions(
            path.parent,
            fold_logits,
            fold_labels,
            fold=fold,
            expected_objects=expected_objects,
        )
        _validate_reported_metrics(
            fold_logits,
            fold_labels,
            {
                **summary["final_metrics"],
                "aircraft20_macro_recall": summary["aircraft20"]["macro_recall"],
            },
            label=f"P03 fold{fold}",
        )
        fold_outputs.append((fold, fold_logits, fold_labels))
    if len(values) != 3:
        raise ValueError(f"P03 formal 必须恰有 3 个正式 run，实际 {len(values)}")
    groups = {("tight", 224, "fine_tune", "natural", 42): values}
    aggregate = aggregate_metric_rows(
        groups,
        metric_names=("macro_recall", "macro_f1", "accuracy", "aircraft20_macro_recall"),
    )
    fold_outputs.sort(key=lambda item: item[0])
    aggregate[0]["tu160_by_fold"] = [
        {
            "fold": fold,
            "train_support": tu160_total - int((fold_labels == 9).sum()),
            **class_recall_support(fold_logits, fold_labels, 9),
        }
        for fold, fold_logits, fold_labels in fold_outputs
    ]
    pooled = pooled_classification_metrics(
        [item[1] for item in fold_outputs],
        [item[2] for item in fold_outputs],
    )
    if pooled["n_samples"] != int(audit["object_count"]):
        raise ValueError("P03 pooled OOF 对象数不完整")
    return {
        "status": "p03_formal_cv3_v2_complete",
        "scientific_scope": "given_gt_object_crop_classification_not_detection",
        "groups": aggregate,
        "pooled_oof": pooled,
    }


def _p04(root: Path, audit: dict[str, Any]) -> dict[str, Any]:
    groups: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    oof: defaultdict[tuple[str, int | None], list[tuple[int, np.ndarray, np.ndarray]]] = (
        defaultdict(list)
    )
    expected_objects = _formal_object_contract(audit)
    tu160_total = sum(contract[1] == 9 for contract in expected_objects.values())
    if int(audit["object_count"]) == 20_933 and tu160_total != 361:
        raise ValueError(f"formal manifest 的 TU-160 support 异常: {tu160_total}")
    if int(audit.get("cache_count", -1)) != 3 or len(audit.get("caches", {})) != 3:
        raise ValueError("P04 formal cache 复用门禁必须恰含三个教师 cache")
    audit_sha = sha256_file(Path(audit["_audit_path"]))
    for path in sorted(root.rglob("run_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise ValueError(f"P04 run 未完成: {path.parent}")
        condition = (str(summary["feature_name"]), summary.get("pca_dim"))
        if condition not in P04_EXPECTED:
            raise ValueError(f"P04 formal 出现未预注册条件: {condition}")
        if summary.get("seed") != 42 or summary.get("normalization") != "train_rms":
            raise ValueError(f"P04 formal seed/normalization 非冻结值: {path.parent}")
        if summary.get("head_init") != "p04_default":
            raise ValueError(f"P04 formal head init 非冻结值: {path.parent}")
        if (
            summary.get("checkpoint_selection") != "fixed_epoch_last"
            or summary.get("selection_metric") != "fixed_epoch_last"
            or summary.get("early_stopped") is not False
            or int(summary.get("completed_epochs", -1)) != 15
            or summary.get("selected_checkpoint") != "final_checkpoint.pt"
        ):
            raise ValueError(f"P04 formal 使用了 held-out 选 best/早停: {path.parent}")
        selected_checkpoint = path.parent / "final_checkpoint.pt"
        if not selected_checkpoint.is_file() or sha256_file(selected_checkpoint) != summary.get(
            "selected_checkpoint_sha256"
        ):
            raise ValueError(f"P04 final checkpoint 缺失或 SHA 不一致: {path.parent}")
        if summary.get("manifest_sha256") != audit["formal_manifest_sha256"]:
            raise ValueError(f"P04 run manifest SHA 不匹配: {path.parent}")
        reuse = summary.get("reuse_audit") or {}
        if reuse.get("sha256") != audit_sha:
            raise ValueError(f"P04 run 未绑定同一 reuse audit: {path.parent}")
        matching_caches = [
            item
            for item in audit.get("caches", {}).values()
            if item.get("cache_fingerprint") == summary.get("cache_fingerprint")
        ]
        if len(matching_caches) != 1:
            raise ValueError(f"P04 run cache 未在复用门禁中唯一准入: {path.parent}")
        admitted_cache = matching_caches[0]
        expected_native_dim = P04_NATIVE_DIMENSIONS[condition[0]]
        actual_native_dim = (
            admitted_cache.get("cache_audit", {}).get("feature_dimensions", {}).get(condition[0])
        )
        if (
            condition[0] not in admitted_cache.get("feature_names", ())
            or actual_native_dim != expected_native_dim
        ):
            raise ValueError(f"P04 teacher/feature/native dimension 合同不匹配: {path.parent}")
        expected_output_dim = condition[1] or expected_native_dim
        if int(summary.get("feature_dimension", -1)) != expected_output_dim:
            raise ValueError(f"P04 probe 输出维度不匹配: {path.parent}")
        if summary.get("train_rms") is None:
            raise ValueError(f"P04 run 缺 train-only RMS: {path.parent}")
        expected_scope = "train_fold_objects_all_cached_views_only"
        if summary.get("normalizer_fit_scope") != expected_scope:
            raise ValueError(f"P04 normalizer fit scope 非 train-only: {path.parent}")
        if condition[1] is not None and summary.get("pca_fit_scope") != expected_scope:
            raise ValueError(f"P04 PCA fit scope 非 train-only: {path.parent}")
        fold = int(summary["fold"])
        if int(summary["n_val"]) != int(audit["tight_fold_object_counts"][fold]):
            raise ValueError(f"P04 fold{fold} n_val 与 formal audit 不一致")
        if int(summary["n_train"]) != int(audit["object_count"]) - int(summary["n_val"]):
            raise ValueError(f"P04 fold{fold} n_train 与 formal audit 不一致")
        if int(summary.get("n_views", -1)) != 8:
            raise ValueError(f"P04 fold{fold} D4 视图数不完整")
        if int(summary.get("ignored_cache_annotations", -1)) != 0:
            raise ValueError(f"P04 fold{fold} 忽略了 cache 对象")
        resolved_path = path.parent / "resolved_config.json"
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        _require_mapping_values(
            resolved,
            {
                **P04_FROZEN_RUNTIME,
                "feature_name": condition[0],
                "fold": fold,
                "pca_dim": condition[1],
            },
            label=f"P04 resolved config {path.parent}",
        )
        rms_path = path.parent / "train_rms_train_only.json"
        rms_contract = json.loads(rms_path.read_text(encoding="utf-8"))
        _require_mapping_values(
            rms_contract,
            {
                "fit_scope": "train_fold_objects_all_cached_views_only",
                "fold": fold,
                "train_object_count": int(summary["n_train"]),
                "view_count": 8,
                "rms": summary["train_rms"],
            },
            label=f"P04 train RMS {path.parent}",
        )
        pca_path = path.parent / "pca_train_only.npz"
        if (condition[1] is not None) != pca_path.is_file():
            raise ValueError(f"P04 PCA artifact 与条件不一致: {path.parent}")
        groups[condition].append(summary)
        fold_logits, fold_labels = _load_npz(path.parent / "validation_logits.npz")
        _validate_fold_predictions(
            path.parent,
            fold_logits,
            fold_labels,
            fold=fold,
            expected_objects=expected_objects,
        )
        _validate_reported_metrics(
            fold_logits,
            fold_labels,
            summary["metrics"],
            label=f"P04 {condition} fold{fold}",
        )
        oof[condition].append((fold, fold_logits, fold_labels))
    if set(groups) != P04_EXPECTED:
        raise ValueError(f"P04 条件集不完整: missing={sorted(P04_EXPECTED - set(groups))}")
    if sum(len(values) for values in groups.values()) != 18:
        raise ValueError("P04 formal 必须恰有 18 个 run")

    keyed = {
        key: [
            {
                "fold": int(value["fold"]),
                "metrics": value["metrics"],
            }
            for value in values
        ]
        for key, values in groups.items()
    }
    aggregate = aggregate_metric_rows(
        keyed,
        metric_names=("macro_recall", "macro_f1", "accuracy", "aircraft20_macro_recall"),
    )
    aggregate_by_condition = {(row["condition"][0], row["condition"][1]): row for row in aggregate}
    pooled: dict[str, Any] = {}
    for condition, fold_values in sorted(oof.items(), key=lambda item: str(item[0])):
        fold_values.sort(key=lambda item: item[0])
        aggregate_by_condition[condition]["tu160_by_fold"] = [
            {
                "fold": fold,
                "train_support": tu160_total - int((fold_labels == 9).sum()),
                **class_recall_support(fold_logits, fold_labels, 9),
            }
            for fold, fold_logits, fold_labels in fold_values
        ]
        # pooled 指标与 fold 顺序无关，直接合并即可。
        condition_pooled = pooled_classification_metrics(
            [item[1] for item in fold_values],
            [item[2] for item in fold_values],
        )
        if condition_pooled["n_samples"] != int(audit["object_count"]):
            raise ValueError(f"P04 {condition} pooled OOF 对象数不完整")
        pooled[f"{condition[0]}|pca={condition[1]}"] = condition_pooled
    native = {
        row["condition"][0]: row["macro_recall_mean"]
        for row in aggregate
        if row["condition"][1] is None
    }
    pca384 = {
        row["condition"][0]: row["macro_recall_mean"]
        for row in aggregate
        if row["condition"][1] == 384
    }
    return {
        "status": "p04_formal_cv3_v2_complete",
        "scientific_scope": "frozen_teacher_representation_probe_not_detection",
        "groups": aggregate,
        "pooled_oof": pooled,
        "native_macro_recall_ranking": sorted(native.items(), key=lambda item: (-item[1], item[0])),
        "pca384_macro_recall_ranking": sorted(pca384.items(), key=lambda item: (-item[1], item[0])),
        "decision_rule": {
            "primary": "native and PCA384 three-fold macro recall with paired OOF review",
            "cleandift_admission": (
                "CleanDIFT is retained only if it adds stable value beyond "
                "DINOv2-B; model name alone is not an admission criterion"
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit_path = Path(args.input_audit).expanduser().resolve()
    audit = _load_input_audit(audit_path)
    audit["_audit_path"] = str(audit_path)
    root = Path(args.runs_root).expanduser().resolve()
    result = _p03(root, audit) if args.stage == "p03" else _p04(root, audit)
    result["formal_manifest_sha256"] = audit["formal_manifest_sha256"]
    result["cv3_manifest_sha256"] = audit["cv3_manifest_sha256"]
    result["input_audit_sha256"] = sha256_file(audit_path)
    _atomic_json(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
