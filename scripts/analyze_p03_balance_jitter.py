#!/usr/bin/env python3
"""P03-3 类别均衡与 clean→jitter 的独立配对分析。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tarfile
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from analyze_p03_results import (
    ConditionData,
    RunData,
    _build_condition,
    _calibration_metrics,
    _cluster_bootstrap_pair,
    _load_run,
    _mcnemar_pvalue,
    _sha256,
    _softmax,
    _verify_return_checksums,
    _write_csv,
)

from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.evaluation.classification import evaluate_classification

CONDITION_ORDER = (
    "natural_clean",
    "sqrt_clean",
    "natural_jitter",
    "sqrt_jitter",
)
PAIR_SPECS = (
    ("natural_clean", "sqrt_clean", "clean_sampler"),
    ("natural_jitter", "sqrt_jitter", "jitter_sampler"),
    ("natural_clean", "natural_jitter", "natural_clean_to_jitter"),
    ("sqrt_clean", "sqrt_jitter", "sqrt_clean_to_jitter"),
)
EXPECTED_NATURAL_CHECKPOINTS = {
    0: (
        111_426_262,
        "66aaa514420e4c1454222e8e80f3e0f614541e5e0a512b543d0135dbd367248e",
    ),
    1: (
        111_426_262,
        "79d70665ce4777a4e54bce117393fb635d39ead5b241a01a4644430d1c8b565a",
    ),
    2: (
        111_426_262,
        "2e395a2291b4c21566d7af6c3eb7e1f970be3cb2773a1292b7d029aa7d80b9d1",
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立验收 P03-TASK-03")
    parser.add_argument("--natural-root", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-archive", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args(argv)


def _verify_code_hashes(task_root: Path, repo_root: Path) -> dict[str, Any]:
    path = task_root / "CODE_SHA256.txt"
    entries = matched = 0
    mismatches: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.strip().lstrip("*")
        entries += 1
        local = repo_root / relative
        if not local.is_file() or _sha256(local) != expected:
            mismatches.append(relative)
        else:
            matched += 1
    return {"entries": entries, "matched": matched, "mismatches": mismatches}


def _verify_checkpoint_archive(path: Path) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    with tarfile.open(path, mode="r:") as archive:
        names = set(archive.getnames())
        for fold, (expected_size, expected_sha) in EXPECTED_NATURAL_CHECKPOINTS.items():
            name = f"P03-TASK-02/ft-tight-224-fold{fold}/best_checkpoint.pt"
            if name not in names:
                raise ValueError(f"checkpoint 归档缺少 {name}")
            info = archive.getmember(name)
            stream = archive.extractfile(info)
            if stream is None:
                raise ValueError(f"无法读取归档成员 {name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            actual_sha = digest.hexdigest()
            if info.size != expected_size or actual_sha != expected_sha:
                raise ValueError(f"{name} 大小或 SHA-256 不一致")
            members.append(
                {
                    "fold": fold,
                    "member": name,
                    "size_bytes": info.size,
                    "sha256": actual_sha,
                }
            )
    return {
        "archive": str(path),
        "archive_size_bytes": path.stat().st_size,
        "archive_sha256": _sha256(path),
        "members": members,
    }


def _load_group(paths: Sequence[Path], label: str) -> list[RunData]:
    if len(paths) != 3:
        raise ValueError(f"{label} 预期 3 个 run，实际 {len(paths)}")
    runs = [_load_run(path) for path in paths]
    runs.sort(key=lambda run: run.fold)
    if [run.fold for run in runs] != [0, 1, 2]:
        raise ValueError(f"{label} fold 不完整")
    return runs


def _load_run_groups(natural_root: Path, task_root: Path) -> dict[str, list[RunData]]:
    groups = {
        "natural_clean": _load_group(
            sorted(natural_root.glob("ft-tight-224-fold*")), "natural_clean"
        ),
        "sqrt_clean": _load_group(
            sorted(task_root.glob("ft-sqrtinv-tight-224-fold*")), "sqrt_clean"
        ),
        "natural_jitter": _load_group(
            sorted(task_root.glob("eval-jitter-natural-tight-224-fold*")),
            "natural_jitter",
        ),
        "sqrt_jitter": _load_group(
            sorted(task_root.glob("eval-jitter-sqrtinv-tight-224-fold*")),
            "sqrt_jitter",
        ),
    }
    expected = {
        "natural_clean": ("tight", "natural", False),
        "sqrt_clean": ("tight", "sqrt_inverse", False),
        "natural_jitter": ("jitter_light", "natural", True),
        "sqrt_jitter": ("jitter_light", "sqrt_inverse", True),
    }
    for label, runs in groups.items():
        policy, sampler, eval_only = expected[label]
        for run in runs:
            actual = (
                run.policy,
                run.sampler,
                run.eval_only,
                run.resolution,
                run.regime,
                run.seed,
            )
            wanted = (policy, sampler, eval_only, 224, "fine_tune", 42)
            if actual != wanted:
                raise ValueError(f"{run.path} 条件错误: {actual} != {wanted}")
            if eval_only:
                source = run.checkpoint_source_condition or {}
                for key, value in {
                    "fold": run.fold,
                    "policy": "tight",
                    "resolution": 224,
                    "regime": "fine_tune",
                    "sampler": sampler,
                    "seed": 42,
                }.items():
                    if source.get(key) != value:
                        raise ValueError(f"{run.path} checkpoint 来源 {key} 不一致")

    for sampler in ("natural", "sqrt"):
        clean_runs = groups[f"{sampler}_clean"]
        jitter_runs = groups[f"{sampler}_jitter"]
        for clean, jitter in zip(clean_runs, jitter_runs, strict=True):
            clean_uids = [row["annotation_uid"] for row in clean.records]
            jitter_uids = [row["annotation_uid"] for row in jitter.records]
            if clean_uids != jitter_uids or not np.array_equal(clean.labels, jitter.labels):
                raise ValueError(f"{sampler} fold{clean.fold} clean/jitter 对象不对齐")
            jitter_meta = json.loads((jitter.path / "meta.json").read_text(encoding="utf-8"))
            expected_sha = json.loads(
                (clean.path / "run_summary.json").read_text(encoding="utf-8")
            )["best_checkpoint_sha256"]
            if jitter_meta["loaded_checkpoint_sha256"] != expected_sha:
                raise ValueError(f"{jitter.path} 加载的 checkpoint SHA-256 错误")
    return groups


def _condition_summary_rows(
    groups: Mapping[str, Sequence[RunData]],
    conditions: Mapping[str, ConditionData],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in CONDITION_ORDER:
        runs = groups[label]
        condition = conditions[label]
        calibration = _calibration_metrics(condition.labels, condition.logits)
        values = {
            metric: [float(run.metrics[metric]) for run in runs]
            for metric in ("macro_recall", "macro_f1", "accuracy")
        }
        row: dict[str, Any] = {
            "condition": label,
            "n_oof_objects": len(condition.labels),
        }
        for metric, numbers in values.items():
            row[f"fold_{metric}_mean"] = mean(numbers)
            row[f"fold_{metric}_std"] = stdev(numbers)
            row[f"pooled_{metric}"] = condition.metrics[metric]
        row["pooled_top5_accuracy"] = condition.metrics["top5_accuracy"]
        for subgroup in ("ship4", "aircraft20", "vehicle1"):
            row[f"pooled_{subgroup}_accuracy"] = condition.metrics["subgroups"][subgroup][
                "accuracy"
            ]
            row[f"pooled_{subgroup}_macro_recall"] = condition.metrics["subgroups"][subgroup][
                "macro_recall"
            ]
        row.update(calibration)
        rows.append(row)
    return rows


def _fold_rows(groups: Mapping[str, Sequence[RunData]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in CONDITION_ORDER:
        for run in groups[label]:
            rows.append(
                {
                    "condition": label,
                    "fold": run.fold,
                    "n_val": len(run.labels),
                    "best_epoch": run.best_epoch,
                    "epochs_completed": run.epochs_completed or None,
                    "max_epochs": run.max_epochs or None,
                    "early_stopped": run.early_stopped if not run.eval_only else None,
                    "accuracy": run.metrics["accuracy"],
                    "macro_recall": run.metrics["macro_recall"],
                    "macro_f1": run.metrics["macro_f1"],
                    "aircraft20_macro_recall": run.metrics["subgroups"]["aircraft20"][
                        "macro_recall"
                    ],
                    "val_samples_per_second": run.val_samples_per_second,
                    "peak_cuda_memory_mib": run.peak_cuda_memory_bytes / 1024**2,
                }
            )
    return rows


def _paired_row(
    first_label: str,
    second_label: str,
    comparison: str,
    first: ConditionData,
    second: ConditionData,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    if not np.array_equal(first.annotation_uids, second.annotation_uids):
        raise ValueError(f"{comparison} annotation_uid 不对齐")
    if not np.array_equal(first.labels, second.labels):
        raise ValueError(f"{comparison} 真值不对齐")
    first_correct = first.predictions == first.labels
    second_correct = second.predictions == second.labels
    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    row = {
        "comparison": comparison,
        "first": first_label,
        "second": second_label,
        "delta_definition": "second_minus_first",
        "n_objects": len(first.labels),
        "both_correct": int(np.sum(first_correct & second_correct)),
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "both_wrong": int(np.sum(~first_correct & ~second_correct)),
        "net_correct_objects": second_only - first_only,
        "accuracy_delta": float(second_correct.mean() - first_correct.mean()),
        "macro_recall_delta": float(second.metrics["macro_recall"] - first.metrics["macro_recall"]),
        "macro_f1_delta": float(second.metrics["macro_f1"] - first.metrics["macro_f1"]),
        "prediction_agreement": float(np.mean(first.predictions == second.predictions)),
        "mcnemar_continuity_chi2_p": _mcnemar_pvalue(first_only, second_only),
    }
    row.update(
        _cluster_bootstrap_pair(
            first,
            second,
            n_resamples=bootstrap_resamples,
            seed=20260717,
        )
    )
    return row


def _tier_map(labels: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    support = np.bincount(labels, minlength=len(FINE_NAMES))
    ordered = np.argsort(support, kind="stable")
    tiers = {
        int(class_id): ("tail", "middle", "head")[min(2, (3 * rank) // 25)]
        for rank, class_id in enumerate(ordered)
    }
    return support, tiers


def _per_class_rows(
    conditions: Mapping[str, ConditionData],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = conditions["natural_clean"]
    support, tiers = _tier_map(reference.labels)
    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(FINE_NAMES):
        row: dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "support": int(support[class_id]),
            "frequency_tier": tiers[class_id],
        }
        for label in CONDITION_ORDER:
            values = conditions[label].metrics["per_class"][class_id]
            row[f"{label}_recall"] = values["recall"]
            row[f"{label}_f1"] = values["f1"]
        row["sqrt_minus_natural_clean_recall"] = (
            row["sqrt_clean_recall"] - row["natural_clean_recall"]
        )
        row["sqrt_minus_natural_jitter_recall"] = (
            row["sqrt_jitter_recall"] - row["natural_jitter_recall"]
        )
        row["natural_jitter_minus_clean_recall"] = (
            row["natural_jitter_recall"] - row["natural_clean_recall"]
        )
        row["sqrt_jitter_minus_clean_recall"] = row["sqrt_jitter_recall"] - row["sqrt_clean_recall"]
        class_mask = reference.labels == class_id
        for sampler in ("natural", "sqrt"):
            clean = conditions[f"{sampler}_clean"]
            jitter = conditions[f"{sampler}_jitter"]
            clean_correct = clean.predictions[class_mask] == clean.labels[class_mask]
            jitter_correct = jitter.predictions[class_mask] == jitter.labels[class_mask]
            row[f"{sampler}_clean_only_correct"] = int(np.sum(clean_correct & ~jitter_correct))
            row[f"{sampler}_jitter_only_correct"] = int(np.sum(~clean_correct & jitter_correct))
        rows.append(row)

    tier_rows: list[dict[str, Any]] = []
    for tier in ("tail", "middle", "head"):
        ids = [class_id for class_id in range(25) if tiers[class_id] == tier]
        for label in CONDITION_ORDER:
            tier_rows.append(
                {
                    "condition": label,
                    "frequency_tier": tier,
                    "n_classes": len(ids),
                    "n_objects": int(support[ids].sum()),
                    "macro_recall": mean(
                        float(conditions[label].metrics["per_class"][class_id]["recall"])
                        for class_id in ids
                    ),
                    "macro_f1": mean(
                        float(conditions[label].metrics["per_class"][class_id]["f1"])
                        for class_id in ids
                    ),
                }
            )
    return rows, tier_rows


def _load_jitter_geometry(path: Path, annotation_uids: np.ndarray) -> dict[str, np.ndarray]:
    values: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["crop_policy"] != "jitter_light":
                continue
            uid = row["annotation_uid"]
            dx = float(row["jitter_dx_fraction"])
            dy = float(row["jitter_dy_fraction"])
            width_scale = float(row["jitter_width_scale"])
            height_scale = float(row["jitter_height_scale"])
            values[uid] = {
                "major_class": row["major_class"],
                "source_edge_risk": row["source_edge_risk"].lower() == "true",
                "padding_fraction": float(row["padding_fraction"]),
                "gt_coverage_fraction": float(row["gt_coverage_fraction"]),
                "gt_short_edge": float(row["gt_short_edge"]),
                "jitter_dx_fraction": dx,
                "jitter_dy_fraction": dy,
                "jitter_center_magnitude": math.hypot(dx, dy),
                "jitter_width_scale": width_scale,
                "jitter_height_scale": height_scale,
                "jitter_scale_log_magnitude": max(
                    abs(math.log(width_scale)), abs(math.log(height_scale))
                ),
            }
    if len(values) != 20_933:
        raise ValueError(f"jitter manifest 对象数异常: {len(values)}")
    ordered = [values[str(uid)] for uid in annotation_uids]
    fields = list(ordered[0])
    return {field: np.asarray([row[field] for row in ordered]) for field in fields}


def _geometry_definitions(geometry: Mapping[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    coverage = geometry["gt_coverage_fraction"].astype(float)
    center = geometry["jitter_center_magnitude"].astype(float)
    scale = geometry["jitter_scale_log_magnitude"].astype(float)
    padding = geometry["padding_fraction"].astype(float)
    short = geometry["gt_short_edge"].astype(float)
    major = geometry["major_class"]
    edge = geometry["source_edge_risk"].astype(bool)
    return [
        ("all", np.ones(len(coverage), dtype=bool)),
        ("major_ship", major == "ship"),
        ("major_aircraft", major == "aircraft"),
        ("major_vehicle", major == "vehicle"),
        ("edge_clean", ~edge),
        ("edge_risk", edge),
        ("padding_zero", padding == 0),
        ("padding_positive", padding > 0),
        ("coverage_lt0p90", coverage < 0.90),
        ("coverage_0p90_to_0p95", (coverage >= 0.90) & (coverage < 0.95)),
        ("coverage_0p95_to_0p99", (coverage >= 0.95) & (coverage < 0.99)),
        ("coverage_ge0p99", coverage >= 0.99),
        ("center_le0p04", center <= 0.04),
        ("center_0p04_to_0p08", (center > 0.04) & (center <= 0.08)),
        ("center_gt0p08", center > 0.08),
        ("scale_log_le0p035", scale <= 0.035),
        ("scale_log_0p035_to_0p07", (scale > 0.035) & (scale <= 0.07)),
        ("scale_log_gt0p07", scale > 0.07),
        ("native_short_lt48", short < 48),
        ("native_short_48_to_96", (short >= 48) & (short <= 96)),
        ("native_short_gt96", short > 96),
    ]


def _jitter_subset_rows(
    conditions: Mapping[str, ConditionData], geometry: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sampler in ("natural", "sqrt"):
        clean = conditions[f"{sampler}_clean"]
        jitter = conditions[f"{sampler}_jitter"]
        clean_probabilities = _softmax(clean.logits)
        jitter_probabilities = _softmax(jitter.logits)
        clean_confidence = clean_probabilities.max(axis=1)
        jitter_confidence = jitter_probabilities.max(axis=1)
        clean_true_probability = clean_probabilities[np.arange(len(clean.labels)), clean.labels]
        jitter_true_probability = jitter_probabilities[np.arange(len(jitter.labels)), jitter.labels]
        clean_correct = clean.predictions == clean.labels
        jitter_correct = jitter.predictions == jitter.labels
        for subset, mask in _geometry_definitions(geometry):
            clean_metrics, _ = evaluate_classification(clean.labels[mask], clean.logits[mask])
            jitter_metrics, _ = evaluate_classification(jitter.labels[mask], jitter.logits[mask])
            rows.append(
                {
                    "sampler": sampler,
                    "subset": subset,
                    "n_objects": int(mask.sum()),
                    "n_active_classes": clean_metrics["n_macro_classes"],
                    "clean_accuracy": clean_metrics["accuracy"],
                    "jitter_accuracy": jitter_metrics["accuracy"],
                    "accuracy_delta": jitter_metrics["accuracy"] - clean_metrics["accuracy"],
                    "clean_macro_recall": clean_metrics["macro_recall"],
                    "jitter_macro_recall": jitter_metrics["macro_recall"],
                    "macro_recall_delta": jitter_metrics["macro_recall"]
                    - clean_metrics["macro_recall"],
                    "clean_only_correct": int(np.sum(mask & clean_correct & ~jitter_correct)),
                    "jitter_only_correct": int(np.sum(mask & ~clean_correct & jitter_correct)),
                    "net_correct_objects": int(
                        np.sum(mask & jitter_correct) - np.sum(mask & clean_correct)
                    ),
                    "mean_max_confidence_delta": float(
                        np.mean(jitter_confidence[mask] - clean_confidence[mask])
                    ),
                    "mean_true_class_probability_delta": float(
                        np.mean(jitter_true_probability[mask] - clean_true_probability[mask])
                    ),
                }
            )
    return rows


def _object_rows(
    conditions: Mapping[str, ConditionData], geometry: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    reference = conditions["natural_clean"]
    probabilities = {label: _softmax(condition.logits) for label, condition in conditions.items()}
    rows: list[dict[str, Any]] = []
    for index, uid in enumerate(reference.annotation_uids):
        class_id = int(reference.labels[index])
        row: dict[str, Any] = {
            "annotation_uid": uid,
            "source_image_id": reference.source_image_ids[index],
            "true_class_id": class_id,
            "true_class_name": FINE_NAMES[class_id],
        }
        for field, values in geometry.items():
            value = values[index]
            row[field] = value.item() if isinstance(value, np.generic) else value
        for label in CONDITION_ORDER:
            condition = conditions[label]
            prediction = int(condition.predictions[index])
            row[f"{label}_pred_class_id"] = prediction
            row[f"{label}_correct"] = prediction == class_id
            row[f"{label}_confidence"] = float(probabilities[label][index].max())
            row[f"{label}_true_probability"] = float(probabilities[label][index, class_id])
        rows.append(row)
    return rows


def _sampler_audit_rows(groups: Mapping[str, Sequence[RunData]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in groups["sqrt_clean"]:
        config = yaml.safe_load((run.path / "resolved_config.yaml").read_text(encoding="utf-8"))
        audit = config["sampler_audit"]
        counts = {int(key): int(value) for key, value in config["training_class_counts"].items()}
        weights = {int(key): float(value) for key, value in audit["class_weights"].items()}
        probabilities = {
            int(key): float(value) for key, value in audit["expected_class_probability"].items()
        }
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-12):
            raise ValueError(f"fold{run.fold} sampler probability 之和不为 1")
        for class_id, class_name in enumerate(FINE_NAMES):
            rows.append(
                {
                    "fold": run.fold,
                    "class_id": class_id,
                    "class_name": class_name,
                    "training_count": counts[class_id],
                    "class_weight": weights[class_id],
                    "natural_probability": counts[class_id] / sum(counts.values()),
                    "sqrt_inverse_expected_probability": probabilities[class_id],
                }
            )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    natural_root = Path(args.natural_root).expanduser().resolve()
    task_root = Path(args.task_root).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    checkpoint_archive = Path(args.checkpoint_archive).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    task_checksums = _verify_return_checksums(task_root)
    natural_checksums = _verify_return_checksums(natural_root)
    if task_checksums.get("unexpected_mismatches") or natural_checksums.get(
        "unexpected_mismatches"
    ):
        raise ValueError("服务器回传文件 checksum 异常")
    code_hashes = _verify_code_hashes(task_root, repo_root)
    if code_hashes["mismatches"]:
        raise ValueError(f"本地/服务器代码不一致: {code_hashes['mismatches']}")
    archive_report = _verify_checkpoint_archive(checkpoint_archive)
    groups = _load_run_groups(natural_root, task_root)
    conditions = {label: _build_condition(groups[label]) for label in CONDITION_ORDER}
    reference = conditions["natural_clean"]
    for label in CONDITION_ORDER[1:]:
        condition = conditions[label]
        if not np.array_equal(condition.annotation_uids, reference.annotation_uids):
            raise ValueError(f"{label} OOF 对象集不一致")
        if not np.array_equal(condition.labels, reference.labels):
            raise ValueError(f"{label} OOF 标签不一致")

    condition_rows = _condition_summary_rows(groups, conditions)
    fold_rows = _fold_rows(groups)
    pairwise_rows = [
        _paired_row(
            first,
            second,
            comparison,
            conditions[first],
            conditions[second],
            args.bootstrap_resamples,
        )
        for first, second, comparison in PAIR_SPECS
    ]
    per_class_rows, tier_rows = _per_class_rows(conditions)
    geometry = _load_jitter_geometry(manifest, reference.annotation_uids)
    subset_rows = _jitter_subset_rows(conditions, geometry)
    object_rows = _object_rows(conditions, geometry)
    sampler_rows = _sampler_audit_rows(groups)

    _write_csv(output / "condition_summary.csv", condition_rows)
    _write_csv(output / "fold_metrics.csv", fold_rows)
    _write_csv(output / "pairwise_comparisons.csv", pairwise_rows)
    _write_csv(output / "per_class_comparison.csv", per_class_rows)
    _write_csv(output / "frequency_tier_metrics.csv", tier_rows)
    _write_csv(output / "jitter_subset_metrics.csv", subset_rows)
    _write_csv(output / "object_comparisons.csv", object_rows)
    _write_csv(output / "sampler_audit.csv", sampler_rows)

    integrity = {
        "status": "pass",
        "task_return_checksums": task_checksums,
        "natural_return_checksums": natural_checksums,
        "code_hashes": code_hashes,
        "checkpoint_archive": archive_report,
        "manifest_sha256": _sha256(manifest),
        "n_runs": sum(len(runs) for runs in groups.values()),
        "n_training_runs": 6,
        "n_eval_only_runs": 6,
        "n_new_task_training_runs": 3,
        "n_new_task_eval_only_runs": 6,
        "n_oof_objects_per_condition": len(reference.labels),
        "raw_logits_recomputed": True,
        "stored_metrics_matched": True,
        "stored_confusions_matched": True,
        "prediction_csv_matched_logits": True,
        "condition_object_universe_matched": True,
        "checkpoint_source_sha256_matched": True,
    }
    (output / "integrity_report.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "integrity": integrity,
        "conditions": condition_rows,
        "pairwise": pairwise_rows,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
