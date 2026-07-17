#!/usr/bin/env python3
"""P03-1/P03-2 回传结果的独立复算、配对比较和误差分层。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.evaluation.classification import evaluate_classification

LINEAR_PROBE_CONDITIONS = (
    ("tight", 224),
    ("tight", 336),
    ("context_1p25", 224),
    ("context_1p25", 336),
)
FINE_TUNE_CONDITIONS = (("tight", 224), ("tight", 336))
EXPECTED_CONDITIONS = LINEAR_PROBE_CONDITIONS
CONDITION_LABELS = {
    ("tight", 224): "tight_224",
    ("tight", 336): "tight_336",
    ("context_1p25", 224): "context_224",
    ("context_1p25", 336): "context_336",
}


@dataclass
class RunData:
    path: Path
    policy: str
    resolution: int
    fold: int
    regime: str
    sampler: str
    seed: int
    eval_only: bool
    checkpoint_source_condition: dict[str, Any] | None
    records: list[dict[str, str]]
    logits: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    metrics: dict[str, Any]
    confusion: np.ndarray
    best_epoch: int
    epochs_completed: int
    max_epochs: int
    early_stopped: bool
    peak_cuda_memory_bytes: int
    train_samples_per_second: float
    val_samples_per_second: float


@dataclass
class ConditionData:
    policy: str
    resolution: int
    annotation_uids: np.ndarray
    source_image_ids: np.ndarray
    labels: np.ndarray
    logits: np.ndarray
    predictions: np.ndarray
    folds: np.ndarray
    metrics: dict[str, Any]
    confusion: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立分析 P03-1/P03-2 crop 分类结果")
    parser.add_argument(
        "--stage",
        choices=("linear_probe", "fine_tune"),
        default="linear_probe",
        help="决定应验收的 run 前缀和条件集",
    )
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} 不一致: actual={actual}, expected={expected}")


def _load_run(path: Path) -> RunData:
    required = {
        "resolved_config.yaml",
        "meta.json",
        "metrics.json",
        "confusion_matrix.csv",
        "predictions.csv",
        "validation_logits.npz",
        "run_summary.json",
    }
    missing = sorted(name for name in required if not (path / name).is_file())
    if missing:
        raise ValueError(f"{path} 缺少产物: {missing}")
    config = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    stored_metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "run_summary.json").read_text(encoding="utf-8"))
    records = _read_csv(path / "predictions.csv")
    with np.load(path / "validation_logits.npz") as values:
        logits = np.asarray(values["logits"], dtype=np.float64)
        labels = np.asarray(values["labels"], dtype=np.int64)
        record_indices = np.asarray(values["record_indices"], dtype=np.int64)
    if len(records) != len(logits) or labels.shape != (len(records),):
        raise ValueError(f"{path} 的 predictions/logits/labels 行数不一致")
    if not np.array_equal(record_indices, np.arange(len(records), dtype=np.int64)):
        raise ValueError(f"{path} 验证 DataLoader 顺序非预期确定顺序")
    csv_labels = np.asarray([int(row["true_class_id"]) for row in records], dtype=np.int64)
    csv_predictions = np.asarray([int(row["pred_class_id"]) for row in records], dtype=np.int64)
    predictions = logits.argmax(axis=1)
    if not np.array_equal(labels, csv_labels) or not np.array_equal(predictions, csv_predictions):
        raise ValueError(f"{path} 的 CSV 与原始 logits 不一致")
    probabilities = _softmax(logits)
    csv_confidence = np.asarray([float(row["confidence"]) for row in records])
    if not np.allclose(
        probabilities[np.arange(len(records)), predictions],
        csv_confidence,
        rtol=0.0,
        # 服务器 CSV 由 float32 logits 导出并只保留 9 位有效数；
        # 本地以 float64 复算 softmax 时保留只能解释该精度边界内的差异。
        atol=5e-7,
    ):
        raise ValueError(f"{path} 的 confidence 与 logits 不一致")

    training_counts = {
        int(key): int(value) for key, value in config["training_class_counts"].items()
    }
    metrics, confusion = evaluate_classification(
        labels, logits, training_class_counts=training_counts
    )
    stored_confusion = np.loadtxt(path / "confusion_matrix.csv", delimiter=",", dtype=np.int64)
    if not np.array_equal(confusion, stored_confusion):
        raise ValueError(f"{path} 的混淆矩阵无法从 logits 复现")
    for key in ("accuracy", "macro_recall", "macro_f1", "top5_accuracy"):
        _assert_close(metrics[key], stored_metrics[key], f"{path.name}:{key}")
        _assert_close(metrics[key], summary["final_metrics"][key], f"{path.name}:summary:{key}")
    if int(stored_metrics["n_samples"]) != len(records):
        raise ValueError(f"{path} n_samples 不一致")
    if len({row["annotation_uid"] for row in records}) != len(records):
        raise ValueError(f"{path} annotation_uid 重复")
    if any(
        (row["correct"] == "True") != (labels[i] == predictions[i]) for i, row in enumerate(records)
    ):
        raise ValueError(f"{path} correct 字段不一致")

    best_epoch = int(summary["best_epoch"])
    if bool(config["eval_only"]):
        if (path / "history.csv").exists():
            raise ValueError(f"{path} eval-only run 不应包含 history.csv")
        max_epochs = 0
        epochs_completed = 0
        early_stopped = False
        train_samples_per_second = float("nan")
    else:
        history_path = path / "history.csv"
        if not history_path.is_file():
            raise ValueError(f"{path} 训练 run 缺少 history.csv")
        history = _read_csv(history_path)
        if not history or not 1 <= best_epoch <= len(history):
            raise ValueError(f"{path} best_epoch 非法")
        max_epochs = int(config["epochs"])
        epochs_completed = len(history)
        if epochs_completed > max_epochs:
            raise ValueError(f"{path} 实际 epoch 数超过配置上限")
        best_history = next(row for row in history if int(row["epoch"]) == best_epoch)
        _assert_close(
            float(best_history["val_macro_recall"]),
            metrics["macro_recall"],
            f"{path.name}:best_epoch_metric",
        )
        early_stopped = epochs_completed < max_epochs
        train_samples_per_second = float(best_history["train_samples_per_second"])
    return RunData(
        path=path,
        policy=str(config["policy"]),
        resolution=int(config["resolution"]),
        fold=int(config["fold"]),
        regime=str(config["regime"]),
        sampler=str(config["sampler"]),
        seed=int(config["seed"]),
        eval_only=bool(config["eval_only"]),
        checkpoint_source_condition=summary.get("checkpoint_source_condition"),
        records=records,
        logits=logits,
        labels=labels,
        predictions=predictions,
        metrics=metrics,
        confusion=confusion,
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
        max_epochs=max_epochs,
        early_stopped=early_stopped,
        peak_cuda_memory_bytes=int(meta["peak_cuda_memory_bytes"]),
        train_samples_per_second=train_samples_per_second,
        val_samples_per_second=float(stored_metrics["samples_per_second"]),
    )


def _verify_return_checksums(runs_root: Path) -> dict[str, Any]:
    checksum_path = runs_root / "RETURN_FILES_SHA256.txt"
    if not checksum_path.is_file():
        return {"present": False}
    checked = matched = self_entry_mismatch = 0
    mismatches: list[str] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.strip().lstrip("*")
        path = runs_root.parent / relative
        checked += 1
        actual = _sha256(path)
        if actual == expected:
            matched += 1
        elif (
            path.resolve() == checksum_path.resolve()
            and expected == hashlib.sha256(b"").hexdigest()
        ):
            self_entry_mismatch += 1
        else:
            mismatches.append(relative)
    return {
        "present": True,
        "entries": checked,
        "matched": matched,
        "known_self_entry_mismatch": self_entry_mismatch,
        "unexpected_mismatches": mismatches,
    }


def _configure_stage(stage: str) -> tuple[str, int]:
    global EXPECTED_CONDITIONS
    if stage == "linear_probe":
        EXPECTED_CONDITIONS = LINEAR_PROBE_CONDITIONS
        return "lp-*", 12
    if stage == "fine_tune":
        EXPECTED_CONDITIONS = FINE_TUNE_CONDITIONS
        return "ft-*", 6
    raise ValueError(f"未知 stage: {stage}")


def _collect_runs(root: Path, stage: str) -> dict[tuple[str, int], list[RunData]]:
    result: dict[tuple[str, int], list[RunData]] = {}
    run_glob, expected_count = _configure_stage(stage)
    run_paths = sorted(path for path in root.glob(run_glob) if path.is_dir())
    if len(run_paths) != expected_count:
        raise ValueError(f"预期 {expected_count} 个 {run_glob} run，实际 {len(run_paths)}")
    for path in run_paths:
        run = _load_run(path)
        key = (run.policy, run.resolution)
        result.setdefault(key, []).append(run)
    if set(result) != set(EXPECTED_CONDITIONS):
        raise ValueError(f"条件集不一致: {sorted(result)}")
    for key, runs in result.items():
        runs.sort(key=lambda item: item.fold)
        if [item.fold for item in runs] != [0, 1, 2]:
            raise ValueError(f"{key} 不是完整三折")
    return result


def _build_condition(runs: Sequence[RunData]) -> ConditionData:
    rows: list[tuple[str, str, int, np.ndarray, int]] = []
    for run in runs:
        for index, record in enumerate(run.records):
            rows.append(
                (
                    record["annotation_uid"],
                    record["source_image_id"],
                    int(run.labels[index]),
                    run.logits[index],
                    run.fold,
                )
            )
    rows.sort(key=lambda item: item[0])
    annotation_uids = np.asarray([item[0] for item in rows], dtype=object)
    if len(annotation_uids) != 20933 or len(set(annotation_uids.tolist())) != 20933:
        raise ValueError("条件 OOF 对象数或唯一性异常")
    source_ids = np.asarray([item[1] for item in rows], dtype=object)
    labels = np.asarray([item[2] for item in rows], dtype=np.int64)
    logits = np.stack([item[3] for item in rows])
    folds = np.asarray([item[4] for item in rows], dtype=np.int64)
    metrics, confusion = evaluate_classification(labels, logits)
    return ConditionData(
        policy=runs[0].policy,
        resolution=runs[0].resolution,
        annotation_uids=annotation_uids,
        source_image_ids=source_ids,
        labels=labels,
        logits=logits,
        predictions=logits.argmax(axis=1),
        folds=folds,
        metrics=metrics,
        confusion=confusion,
    )


def _calibration_metrics(
    labels: np.ndarray, logits: np.ndarray, bins: int = 15
) -> dict[str, float]:
    probabilities = _softmax(logits)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    nll = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (confidence >= lower) & (
            confidence < upper if index < bins - 1 else confidence <= upper
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return {
        "nll": float(nll),
        "ece_15": ece,
        "mean_confidence": float(confidence.mean()),
        "mean_confidence_correct": float(confidence[correct].mean()),
        "mean_confidence_wrong": float(confidence[~correct].mean()),
    }


def _mcnemar_pvalue(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    statistic = (max(0, abs(first_only - second_only) - 1) ** 2) / discordant
    return math.erfc(math.sqrt(statistic / 2.0))


def _pairwise_row(first: ConditionData, second: ConditionData) -> dict[str, Any]:
    if not np.array_equal(first.annotation_uids, second.annotation_uids):
        raise ValueError("配对条件 annotation_uid 不对齐")
    if not np.array_equal(first.labels, second.labels):
        raise ValueError("配对条件真值不对齐")
    first_correct = first.predictions == first.labels
    second_correct = second.predictions == second.labels
    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    return {
        "first": CONDITION_LABELS[(first.policy, first.resolution)],
        "second": CONDITION_LABELS[(second.policy, second.resolution)],
        "delta_definition": "second_minus_first",
        "n_objects": len(first.labels),
        "both_correct": int(np.sum(first_correct & second_correct)),
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "both_wrong": int(np.sum(~first_correct & ~second_correct)),
        "net_correct_objects": second_only - first_only,
        "accuracy_delta": float(second_correct.mean() - first_correct.mean()),
        "macro_recall_delta": float(second.metrics["macro_recall"] - first.metrics["macro_recall"]),
        "prediction_agreement": float(np.mean(first.predictions == second.predictions)),
        "mcnemar_continuity_chi2_p": _mcnemar_pvalue(first_only, second_only),
    }


def _cluster_bootstrap_pair(
    first: ConditionData,
    second: ConditionData,
    *,
    n_resamples: int = 10_000,
    seed: int = 20260717,
) -> dict[str, Any]:
    """按源图聚类重抽样，估计已训练 OOF 模型的配对差异区间。

    该区间只包含当前已训练模型在验证对象上的抽样不确定性，不包含
    重新训练时的 seed/优化方差。
    """

    if not np.array_equal(first.annotation_uids, second.annotation_uids):
        raise ValueError("配对 bootstrap 的 annotation_uid 不对齐")
    if not np.array_equal(first.source_image_ids, second.source_image_ids):
        raise ValueError("配对 bootstrap 的 source_image_id 不对齐")
    if not np.array_equal(first.labels, second.labels):
        raise ValueError("配对 bootstrap 的真值不对齐")
    source_ids, inverse = np.unique(first.source_image_ids, return_inverse=True)
    n_sources = len(source_ids)
    n_classes = first.logits.shape[1]
    support = np.zeros((n_sources, n_classes), dtype=np.int32)
    first_correct = np.zeros_like(support)
    second_correct = np.zeros_like(support)
    for index, (source_index, class_id) in enumerate(zip(inverse, first.labels, strict=True)):
        support[source_index, class_id] += 1
        first_correct[source_index, class_id] += int(first.predictions[index] == class_id)
        second_correct[source_index, class_id] += int(second.predictions[index] == class_id)

    rng = np.random.default_rng(seed)
    macro_deltas: list[float] = []
    accuracy_deltas: list[float] = []
    batch_size = 100
    for start in range(0, n_resamples, batch_size):
        size = min(batch_size, n_resamples - start)
        sampled = rng.integers(0, n_sources, size=(size, n_sources))
        sampled_support = support[sampled].sum(axis=1)
        first_counts = first_correct[sampled].sum(axis=1)
        second_counts = second_correct[sampled].sum(axis=1)
        active = sampled_support > 0
        first_recall = np.divide(
            first_counts,
            sampled_support,
            out=np.zeros_like(first_counts, dtype=np.float64),
            where=active,
        )
        second_recall = np.divide(
            second_counts,
            sampled_support,
            out=np.zeros_like(second_counts, dtype=np.float64),
            where=active,
        )
        macro_deltas.extend(
            (
                (second_recall * active).sum(axis=1) / active.sum(axis=1)
                - (first_recall * active).sum(axis=1) / active.sum(axis=1)
            ).tolist()
        )
        accuracy_deltas.extend(
            (
                (second_counts.sum(axis=1) - first_counts.sum(axis=1)) / sampled_support.sum(axis=1)
            ).tolist()
        )

    result: dict[str, Any] = {
        "cluster_unit": "source_image_id",
        "n_source_clusters": n_sources,
        "n_resamples": n_resamples,
        "seed": seed,
        "scope": "conditional_on_fitted_models; excludes_retraining_variance",
    }
    for name, values in (
        ("macro_recall_delta", np.asarray(macro_deltas)),
        ("accuracy_delta", np.asarray(accuracy_deltas)),
    ):
        result[f"{name}_bootstrap_mean"] = float(values.mean())
        result[f"{name}_ci95_lower"] = float(np.quantile(values, 0.025))
        result[f"{name}_ci95_upper"] = float(np.quantile(values, 0.975))
        result[f"{name}_probability_gt_zero"] = float(np.mean(values > 0))
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"不允许写空 CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_manifest_geometry(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            policy = row["crop_policy"]
            if policy not in {"tight", "context_1p25"}:
                continue
            key = (row["annotation_uid"], policy)
            result[key] = {
                "major_class": row["major_class"],
                "source_edge_risk": row["source_edge_risk"].lower() == "true",
                "padding_fraction": float(row["padding_fraction"]),
                "gt_short_edge": float(row["gt_short_edge"]),
            }
    if len(result) != 2 * 20933:
        raise ValueError(f"manifest clean geometry 行数异常: {len(result)}")
    return result


def _subset_rows(
    condition: ConditionData,
    geometry: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    metadata = [geometry[(str(uid), condition.policy)] for uid in condition.annotation_uids]
    definitions: list[tuple[str, np.ndarray]] = [("all", np.ones(len(metadata), dtype=bool))]
    for major in ("ship", "aircraft", "vehicle"):
        definitions.append(
            (f"major_{major}", np.asarray([item["major_class"] == major for item in metadata]))
        )
    definitions.extend(
        [
            ("edge_clean", np.asarray([not item["source_edge_risk"] for item in metadata])),
            ("edge_risk", np.asarray([item["source_edge_risk"] for item in metadata])),
            ("padding_zero", np.asarray([item["padding_fraction"] == 0 for item in metadata])),
            (
                "padding_0_to_0p1",
                np.asarray([0 < item["padding_fraction"] <= 0.1 for item in metadata]),
            ),
            ("padding_gt_0p1", np.asarray([item["padding_fraction"] > 0.1 for item in metadata])),
            ("native_short_lt48", np.asarray([item["gt_short_edge"] < 48 for item in metadata])),
            (
                "native_short_48_to_96",
                np.asarray([48 <= item["gt_short_edge"] <= 96 for item in metadata]),
            ),
            ("native_short_gt96", np.asarray([item["gt_short_edge"] > 96 for item in metadata])),
        ]
    )
    rows: list[dict[str, Any]] = []
    label = CONDITION_LABELS[(condition.policy, condition.resolution)]
    for subset_name, mask in definitions:
        if not mask.any():
            continue
        metrics, _ = evaluate_classification(condition.labels[mask], condition.logits[mask])
        rows.append(
            {
                "condition": label,
                "subset": subset_name,
                "n_objects": int(mask.sum()),
                "n_active_classes": metrics["n_macro_classes"],
                "accuracy": metrics["accuracy"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    return rows


def _per_class_rows(conditions: Mapping[tuple[str, int], ConditionData]) -> list[dict[str, Any]]:
    reference = conditions[("tight", 224)]
    supports = np.bincount(reference.labels, minlength=25)
    ordered = np.argsort(supports, kind="stable")
    tier_by_class: dict[int, str] = {}
    for rank, class_id in enumerate(ordered):
        tier_by_class[int(class_id)] = ("tail", "middle", "head")[min(2, (3 * rank) // 25)]
    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(FINE_NAMES):
        row: dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "support": int(supports[class_id]),
            "frequency_tier": tier_by_class[class_id],
        }
        for key in EXPECTED_CONDITIONS:
            condition = conditions[key]
            values = condition.metrics["per_class"][class_id]
            label = CONDITION_LABELS[key]
            row[f"{label}_recall"] = values["recall"]
            row[f"{label}_f1"] = values["f1"]
        row["tight336_minus_tight224_recall"] = row["tight_336_recall"] - row["tight_224_recall"]
        first = conditions[("tight", 224)]
        second = conditions[("tight", 336)]
        class_mask = first.labels == class_id
        first_correct = first.predictions[class_mask] == first.labels[class_mask]
        second_correct = second.predictions[class_mask] == second.labels[class_mask]
        row["tight224_only_correct_count"] = int(np.sum(first_correct & ~second_correct))
        row["tight336_only_correct_count"] = int(np.sum(~first_correct & second_correct))
        row["tight336_net_correct_count"] = (
            row["tight336_only_correct_count"] - row["tight224_only_correct_count"]
        )
        if ("context_1p25", 224) in conditions:
            row["tight224_minus_context224_recall"] = (
                row["tight_224_recall"] - row["context_224_recall"]
            )
            row["tight336_minus_context336_recall"] = (
                row["tight_336_recall"] - row["context_336_recall"]
            )
        rows.append(row)
    return rows


def _top_confusion_rows(
    conditions: Mapping[tuple[str, int], ConditionData],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in EXPECTED_CONDITIONS:
        condition = conditions[key]
        candidates: list[tuple[int, int, int]] = []
        for true_id in range(25):
            for pred_id in range(25):
                if true_id != pred_id and condition.confusion[true_id, pred_id] > 0:
                    candidates.append(
                        (int(condition.confusion[true_id, pred_id]), true_id, pred_id)
                    )
        for rank, (count, true_id, pred_id) in enumerate(
            sorted(candidates, reverse=True)[:15], start=1
        ):
            rows.append(
                {
                    "condition": CONDITION_LABELS[key],
                    "rank": rank,
                    "count": count,
                    "true_class_id": true_id,
                    "true_class_name": FINE_NAMES[true_id],
                    "pred_class_id": pred_id,
                    "pred_class_name": FINE_NAMES[pred_id],
                }
            )
    return rows


def _frequency_tier_rows(per_class_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tier in ("tail", "middle", "head"):
        selected = [row for row in per_class_rows if row["frequency_tier"] == tier]
        for key in EXPECTED_CONDITIONS:
            label = CONDITION_LABELS[key]
            rows.append(
                {
                    "condition": label,
                    "frequency_tier": tier,
                    "n_classes": len(selected),
                    "n_objects": sum(int(row["support"]) for row in selected),
                    "macro_recall": mean(float(row[f"{label}_recall"]) for row in selected),
                    "macro_f1": mean(float(row[f"{label}_f1"]) for row in selected),
                }
            )
    return rows


def _fold_pairwise_rows(
    runs_by_condition: Mapping[tuple[str, int], Sequence[RunData]],
    pair_specs: Sequence[tuple[tuple[str, int], tuple[str, int]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for first_key, second_key in pair_specs:
        first_runs = {run.fold: run for run in runs_by_condition[first_key]}
        second_runs = {run.fold: run for run in runs_by_condition[second_key]}
        for fold in (0, 1, 2):
            first, second = first_runs[fold], second_runs[fold]
            rows.append(
                {
                    "first": CONDITION_LABELS[first_key],
                    "second": CONDITION_LABELS[second_key],
                    "fold": fold,
                    "delta_definition": "second_minus_first",
                    "macro_recall_delta": second.metrics["macro_recall"]
                    - first.metrics["macro_recall"],
                    "macro_f1_delta": second.metrics["macro_f1"] - first.metrics["macro_f1"],
                    "accuracy_delta": second.metrics["accuracy"] - first.metrics["accuracy"],
                }
            )
    return rows


def _object_stability_rows(
    conditions: Mapping[tuple[str, int], ConditionData],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    reference = conditions[("tight", 224)]
    rows: list[dict[str, Any]] = []
    n_conditions = len(EXPECTED_CONDITIONS)
    counts = {f"correct_{index}_of_{n_conditions}": 0 for index in range(n_conditions + 1)}
    for index, uid in enumerate(reference.annotation_uids):
        row: dict[str, Any] = {
            "annotation_uid": uid,
            "source_image_id": reference.source_image_ids[index],
            "true_class_id": int(reference.labels[index]),
            "true_class_name": FINE_NAMES[int(reference.labels[index])],
        }
        correct_count = 0
        for key in EXPECTED_CONDITIONS:
            condition = conditions[key]
            if (
                condition.annotation_uids[index] != uid
                or condition.labels[index] != reference.labels[index]
            ):
                raise ValueError("条件间 OOF 排序或标签不一致")
            label = CONDITION_LABELS[key]
            prediction = int(condition.predictions[index])
            correct = prediction == int(reference.labels[index])
            row[f"{label}_pred_class_id"] = prediction
            row[f"{label}_correct"] = correct
            correct_count += int(correct)
        row["correct_condition_count"] = correct_count
        counts[f"correct_{correct_count}_of_{n_conditions}"] += 1
        rows.append(row)
    return rows, counts


def _stability_per_class_rows(
    stability_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_conditions = len(EXPECTED_CONDITIONS)
    for class_id, class_name in enumerate(FINE_NAMES):
        selected = [row for row in stability_rows if int(row["true_class_id"]) == class_id]
        always_wrong = sum(int(row["correct_condition_count"]) == 0 for row in selected)
        always_correct = sum(
            int(row["correct_condition_count"]) == n_conditions for row in selected
        )
        variable = len(selected) - always_wrong - always_correct
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support": len(selected),
                "always_wrong_count": always_wrong,
                "always_wrong_rate": always_wrong / len(selected),
                "variable_count": variable,
                "variable_rate": variable / len(selected),
                "always_correct_count": always_correct,
                "always_correct_rate": always_correct / len(selected),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs_root = Path(args.runs_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    checksum_report = _verify_return_checksums(runs_root)
    if checksum_report.get("unexpected_mismatches"):
        raise ValueError(f"回传文件 checksum 异常: {checksum_report['unexpected_mismatches']}")
    runs_by_condition = _collect_runs(runs_root, args.stage)
    conditions = {key: _build_condition(runs) for key, runs in runs_by_condition.items()}
    reference_uids = conditions[("tight", 224)].annotation_uids
    reference_labels = conditions[("tight", 224)].labels
    for key, condition in conditions.items():
        if not np.array_equal(condition.annotation_uids, reference_uids):
            raise ValueError(f"{key} 与参考条件对象集不一致")
        if not np.array_equal(condition.labels, reference_labels):
            raise ValueError(f"{key} 与参考条件标签不一致")

    fold_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    for key in EXPECTED_CONDITIONS:
        runs = runs_by_condition[key]
        label = CONDITION_LABELS[key]
        for run in runs:
            fold_rows.append(
                {
                    "condition": label,
                    "fold": run.fold,
                    "n_val": len(run.labels),
                    "best_epoch": run.best_epoch,
                    "epochs_completed": run.epochs_completed,
                    "max_epochs": run.max_epochs,
                    "early_stopped": run.early_stopped,
                    "accuracy": run.metrics["accuracy"],
                    "macro_recall": run.metrics["macro_recall"],
                    "macro_f1": run.metrics["macro_f1"],
                    "aircraft20_macro_recall": run.metrics["subgroups"]["aircraft20"][
                        "macro_recall"
                    ],
                    "peak_cuda_memory_mib": run.peak_cuda_memory_bytes / 1024**2,
                    "train_samples_per_second_at_best_epoch": run.train_samples_per_second,
                    "val_samples_per_second": run.val_samples_per_second,
                }
            )
        condition = conditions[key]
        calibration = _calibration_metrics(condition.labels, condition.logits)
        macro_values = [float(run.metrics["macro_recall"]) for run in runs]
        f1_values = [float(run.metrics["macro_f1"]) for run in runs]
        accuracy_values = [float(run.metrics["accuracy"]) for run in runs]
        condition_rows.append(
            {
                "condition": label,
                "policy": key[0],
                "resolution": key[1],
                "n_oof_objects": len(condition.labels),
                "fold_macro_recall_mean": mean(macro_values),
                "fold_macro_recall_std": stdev(macro_values),
                "fold_macro_f1_mean": mean(f1_values),
                "fold_macro_f1_std": stdev(f1_values),
                "fold_accuracy_mean": mean(accuracy_values),
                "fold_accuracy_std": stdev(accuracy_values),
                "pooled_oof_macro_recall": condition.metrics["macro_recall"],
                "pooled_oof_macro_f1": condition.metrics["macro_f1"],
                "pooled_oof_accuracy": condition.metrics["accuracy"],
                "pooled_oof_top5_accuracy": condition.metrics["top5_accuracy"],
                "pooled_ship4_accuracy": condition.metrics["subgroups"]["ship4"]["accuracy"],
                "pooled_ship4_macro_recall": condition.metrics["subgroups"]["ship4"][
                    "macro_recall"
                ],
                "pooled_aircraft20_accuracy": condition.metrics["subgroups"]["aircraft20"][
                    "accuracy"
                ],
                "pooled_aircraft20_macro_recall": condition.metrics["subgroups"]["aircraft20"][
                    "macro_recall"
                ],
                "pooled_vehicle1_accuracy": condition.metrics["subgroups"]["vehicle1"]["accuracy"],
                "nll": calibration["nll"],
                "ece_15": calibration["ece_15"],
                "mean_confidence": calibration["mean_confidence"],
                "mean_confidence_correct": calibration["mean_confidence_correct"],
                "mean_confidence_wrong": calibration["mean_confidence_wrong"],
                "best_epoch_min": min(run.best_epoch for run in runs),
                "best_epoch_max": max(run.best_epoch for run in runs),
                "peak_cuda_memory_mib_max": max(run.peak_cuda_memory_bytes for run in runs)
                / 1024**2,
            }
        )

    if args.stage == "linear_probe":
        pair_specs = (
            (("tight", 224), ("tight", 336)),
            (("context_1p25", 224), ("context_1p25", 336)),
            (("context_1p25", 224), ("tight", 224)),
            (("context_1p25", 336), ("tight", 336)),
        )
    else:
        pair_specs = ((("tight", 224), ("tight", 336)),)
    pairwise_rows = []
    for first, second in pair_specs:
        row = _pairwise_row(conditions[first], conditions[second])
        row.update(_cluster_bootstrap_pair(conditions[first], conditions[second]))
        pairwise_rows.append(row)
    fold_pairwise_rows = _fold_pairwise_rows(runs_by_condition, pair_specs)
    geometry = _load_manifest_geometry(manifest_path)
    subset_rows = [
        row for key in EXPECTED_CONDITIONS for row in _subset_rows(conditions[key], geometry)
    ]
    per_class_rows = _per_class_rows(conditions)
    frequency_tier_rows = _frequency_tier_rows(per_class_rows)
    confusion_rows = _top_confusion_rows(conditions)
    stability_rows, stability_counts = _object_stability_rows(conditions)
    stability_per_class_rows = _stability_per_class_rows(stability_rows)

    _write_csv(output / "fold_metrics.csv", fold_rows)
    _write_csv(output / "condition_summary.csv", condition_rows)
    _write_csv(output / "pairwise_comparisons.csv", pairwise_rows)
    _write_csv(output / "fold_pairwise_deltas.csv", fold_pairwise_rows)
    _write_csv(output / "per_class_comparison.csv", per_class_rows)
    _write_csv(output / "frequency_tier_metrics.csv", frequency_tier_rows)
    _write_csv(output / "subset_metrics.csv", subset_rows)
    _write_csv(output / "top_confusions.csv", confusion_rows)
    _write_csv(output / "object_stability.csv", stability_rows)
    _write_csv(output / "stability_per_class.csv", stability_per_class_rows)
    integrity = {
        "status": "pass",
        "stage": args.stage,
        "return_checksums": checksum_report,
        "n_runs": sum(len(runs) for runs in runs_by_condition.values()),
        "conditions": [CONDITION_LABELS[key] for key in EXPECTED_CONDITIONS],
        "folds_per_condition": 3,
        "n_oof_objects_per_condition": 20933,
        "raw_logits_recomputed": True,
        "stored_metrics_matched": True,
        "stored_confusions_matched": True,
        "prediction_csv_matched_logits": True,
        "condition_object_universe_matched": True,
        "manifest_sha256": _sha256(manifest_path),
        "stability_counts": stability_counts,
    }
    (output / "integrity_report.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "integrity": integrity,
        "conditions": condition_rows,
        "pairwise": pairwise_rows,
        "stability_counts": stability_counts,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
