#!/usr/bin/env python3
"""P03-5 三随机种子 clean/jitter 稳定性的独立复算与配对分析。"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np
from analyze_p03_balance_jitter import (
    _geometry_definitions,
    _load_jitter_geometry,
    _tier_map,
    _verify_code_hashes,
)
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

SEEDS = (42, 3407, 202625)
CONDITIONS = ("clean", "jitter")
EXPECTED_TASK_ARCHIVE_SHA256 = "f15658202d388d6539f9037bf322b98afe0babf4683d48e536206d69c37d5923"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立验收 P03-TASK-04 多 seed 稳定性")
    parser.add_argument("--seed42-clean-root", required=True)
    parser.add_argument("--seed42-jitter-root", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--task-archive", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args(argv)


def _load_group(paths: Sequence[Path], label: str) -> list[RunData]:
    if len(paths) != 3:
        raise ValueError(f"{label} 预期 3 个 run，实际 {len(paths)}")
    runs = sorted((_load_run(path) for path in paths), key=lambda run: run.fold)
    if [run.fold for run in runs] != [0, 1, 2]:
        raise ValueError(f"{label} fold 不完整")
    return runs


def _load_groups(
    seed42_clean_root: Path,
    seed42_jitter_root: Path,
    task_root: Path,
) -> dict[tuple[int, str], list[RunData]]:
    groups = {
        (42, "clean"): _load_group(
            sorted(seed42_clean_root.glob("ft-tight-224-fold*")), "seed42-clean"
        ),
        (42, "jitter"): _load_group(
            sorted(seed42_jitter_root.glob("eval-jitter-natural-tight-224-fold*")),
            "seed42-jitter",
        ),
    }
    for seed in SEEDS[1:]:
        groups[(seed, "clean")] = _load_group(
            sorted(task_root.glob(f"ft-natural-tight-224-seed{seed}-fold*")),
            f"seed{seed}-clean",
        )
        groups[(seed, "jitter")] = _load_group(
            sorted(task_root.glob(f"eval-jitter-natural-tight-224-seed{seed}-fold*")),
            f"seed{seed}-jitter",
        )

    for (seed, condition), runs in groups.items():
        policy = "tight" if condition == "clean" else "jitter_light"
        for run in runs:
            actual = (
                run.policy,
                run.resolution,
                run.regime,
                run.sampler,
                run.seed,
                run.eval_only,
            )
            expected = (policy, 224, "fine_tune", "natural", seed, condition == "jitter")
            if actual != expected:
                raise ValueError(f"{run.path} 条件错误: {actual} != {expected}")
            if condition == "jitter":
                source = run.checkpoint_source_condition or {}
                source_expected = {
                    "fold": run.fold,
                    "policy": "tight",
                    "resolution": 224,
                    "regime": "fine_tune",
                    "sampler": "natural",
                    "seed": seed,
                }
                for key, value in source_expected.items():
                    if source.get(key) != value:
                        raise ValueError(f"{run.path} checkpoint 来源 {key} 不一致")

        clean = groups[(seed, "clean")]
        jitter = groups[(seed, "jitter")]
        for clean_run, jitter_run in zip(clean, jitter, strict=True):
            if [row["annotation_uid"] for row in clean_run.records] != [
                row["annotation_uid"] for row in jitter_run.records
            ] or not np.array_equal(clean_run.labels, jitter_run.labels):
                raise ValueError(f"seed{seed}/fold{clean_run.fold} clean/jitter 对象不对齐")
            jitter_meta = json.loads((jitter_run.path / "meta.json").read_text(encoding="utf-8"))
            clean_summary = json.loads(
                (clean_run.path / "run_summary.json").read_text(encoding="utf-8")
            )
            if jitter_meta["loaded_checkpoint_sha256"] != clean_summary["best_checkpoint_sha256"]:
                raise ValueError(f"seed{seed}/fold{clean_run.fold} checkpoint SHA-256 不一致")
    return groups


def _parse_checkpoint_inventory(task_root: Path) -> dict[str, Any]:
    sha_rows: dict[tuple[int, int], str] = {}
    pattern = re.compile(r"seed(\d+)-fold(\d+)/best_checkpoint\.pt$")
    for line in (task_root / "CHECKPOINTS_SHA256.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, path = line.split(maxsplit=1)
        match = pattern.search(path)
        if not match:
            raise ValueError(f"无法解析 checkpoint 路径: {path}")
        sha_rows[(int(match.group(1)), int(match.group(2)))] = sha

    size_rows: dict[tuple[int, int], int] = {}
    for line in (task_root / "CHECKPOINTS_SIZES.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        size, path = line.split(maxsplit=1)
        match = pattern.search(path)
        if not match:
            raise ValueError(f"无法解析 checkpoint 大小路径: {path}")
        size_rows[(int(match.group(1)), int(match.group(2)))] = int(size)
    expected_keys = {(seed, fold) for seed in SEEDS[1:] for fold in range(3)}
    if set(sha_rows) != expected_keys or set(size_rows) != expected_keys:
        raise ValueError("TASK-04 checkpoint 清单条件不完整")
    return {
        "count": len(expected_keys),
        "all_sizes_equal": len(set(size_rows.values())) == 1,
        "size_bytes": sorted(set(size_rows.values())),
        "sha256": {
            f"seed{seed}_fold{fold}": sha_rows[(seed, fold)] for seed, fold in sorted(expected_keys)
        },
        "sizes": {
            f"seed{seed}_fold{fold}": size_rows[(seed, fold)]
            for seed, fold in sorted(expected_keys)
        },
    }


def _verify_inventory_against_runs(
    inventory: Mapping[str, Any], groups: Mapping[tuple[int, str], Sequence[RunData]]
) -> None:
    for seed in SEEDS[1:]:
        for run in groups[(seed, "clean")]:
            summary = json.loads((run.path / "run_summary.json").read_text(encoding="utf-8"))
            key = f"seed{seed}_fold{run.fold}"
            if summary["best_checkpoint_sha256"] != inventory["sha256"][key]:
                raise ValueError(f"{key} run_summary/checkpoint 清单 SHA-256 不一致")


def _condition_rows(
    groups: Mapping[tuple[int, str], Sequence[RunData]],
    conditions: Mapping[tuple[int, str], ConditionData],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for condition_name in CONDITIONS:
            runs = groups[(seed, condition_name)]
            condition = conditions[(seed, condition_name)]
            row: dict[str, Any] = {
                "seed": seed,
                "condition": condition_name,
                "n_oof_objects": len(condition.labels),
            }
            for metric in ("macro_recall", "macro_f1", "accuracy"):
                values = [float(run.metrics[metric]) for run in runs]
                row[f"fold_{metric}_mean"] = mean(values)
                row[f"fold_{metric}_std"] = stdev(values)
                row[f"pooled_{metric}"] = condition.metrics[metric]
            row["pooled_top5_accuracy"] = condition.metrics["top5_accuracy"]
            for subgroup in ("ship4", "aircraft20", "vehicle1"):
                values = condition.metrics["subgroups"][subgroup]
                row[f"pooled_{subgroup}_accuracy"] = values["accuracy"]
                row[f"pooled_{subgroup}_macro_recall"] = values["macro_recall"]
            row.update(_calibration_metrics(condition.labels, condition.logits))
            rows.append(row)
    return rows


def _fold_rows(groups: Mapping[tuple[int, str], Sequence[RunData]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for condition_name in CONDITIONS:
            for run in groups[(seed, condition_name)]:
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition_name,
                        "fold": run.fold,
                        "n_val": len(run.labels),
                        "best_epoch": run.best_epoch,
                        "epochs_completed": run.epochs_completed or None,
                        "max_epochs": run.max_epochs or None,
                        "early_stopped": (run.early_stopped if condition_name == "clean" else None),
                        "accuracy": run.metrics["accuracy"],
                        "macro_recall": run.metrics["macro_recall"],
                        "macro_f1": run.metrics["macro_f1"],
                        "aircraft20_macro_recall": run.metrics["subgroups"]["aircraft20"][
                            "macro_recall"
                        ],
                        "samples_per_second": run.val_samples_per_second,
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
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not np.array_equal(first.annotation_uids, second.annotation_uids):
        raise ValueError(f"{comparison} annotation_uid 不对齐")
    if not np.array_equal(first.labels, second.labels):
        raise ValueError(f"{comparison} 标签不对齐")
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
            seed=bootstrap_seed,
        )
    )
    return row


def _two_way_rows(groups: Mapping[tuple[int, str], Sequence[RunData]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_name in CONDITIONS:
        for metric in ("macro_recall", "macro_f1", "accuracy"):
            matrix = np.asarray(
                [
                    [float(run.metrics[metric]) for run in groups[(seed, condition_name)]]
                    for seed in SEEDS
                ]
            )
            grand = float(matrix.mean())
            seed_means = matrix.mean(axis=1)
            fold_means = matrix.mean(axis=0)
            residual = matrix - seed_means[:, None] - fold_means[None, :] + grand
            ss_seed = float(matrix.shape[1] * np.sum((seed_means - grand) ** 2))
            ss_fold = float(matrix.shape[0] * np.sum((fold_means - grand) ** 2))
            ss_residual = float(np.sum(residual**2))
            ss_total = ss_seed + ss_fold + ss_residual
            rows.append(
                {
                    "condition": condition_name,
                    "metric": metric,
                    "grand_mean_9_runs": grand,
                    "seed42_mean": seed_means[0],
                    "seed3407_mean": seed_means[1],
                    "seed202625_mean": seed_means[2],
                    "seed_mean_range": float(seed_means.max() - seed_means.min()),
                    "seed_mean_sample_std": float(np.std(seed_means, ddof=1)),
                    "fold0_mean": fold_means[0],
                    "fold1_mean": fold_means[1],
                    "fold2_mean": fold_means[2],
                    "fold_mean_range": float(fold_means.max() - fold_means.min()),
                    "fold_mean_sample_std": float(np.std(fold_means, ddof=1)),
                    "residual_rmse": float(np.sqrt(np.mean(residual**2))),
                    "descriptive_ss_seed_share": ss_seed / ss_total if ss_total else 0.0,
                    "descriptive_ss_fold_share": ss_fold / ss_total if ss_total else 0.0,
                    "descriptive_ss_interaction_share": ss_residual / ss_total if ss_total else 0.0,
                }
            )
    return rows


def _object_stability(
    conditions: Mapping[tuple[int, str], ConditionData],
    geometry: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    reference = conditions[(42, "clean")]
    object_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for condition_name in CONDITIONS:
        condition_values = [conditions[(seed, condition_name)] for seed in SEEDS]
        predictions = np.stack([condition.predictions for condition in condition_values])
        correct = predictions == reference.labels[None, :]
        probabilities = np.stack([_softmax(condition.logits) for condition in condition_values])
        unanimous = np.all(predictions == predictions[0:1], axis=0)
        all_distinct = (
            (predictions[0] != predictions[1])
            & (predictions[0] != predictions[2])
            & (predictions[1] != predictions[2])
        )
        two_agree = ~unanimous & ~all_distinct
        correct_count = correct.sum(axis=0)
        ensemble_probabilities = probabilities.mean(axis=0)
        ensemble_logits = np.log(np.clip(ensemble_probabilities, 1e-12, 1.0))
        ensemble_metrics, _ = evaluate_classification(reference.labels, ensemble_logits)
        summary_rows.append(
            {
                "condition": condition_name,
                "n_objects": len(reference.labels),
                "unanimous_prediction": int(unanimous.sum()),
                "unanimous_prediction_rate": float(unanimous.mean()),
                "exactly_two_seeds_agree": int(two_agree.sum()),
                "all_three_predictions_distinct": int(all_distinct.sum()),
                "all_three_correct": int(np.sum(correct_count == 3)),
                "exactly_two_correct": int(np.sum(correct_count == 2)),
                "exactly_one_correct": int(np.sum(correct_count == 1)),
                "all_three_wrong": int(np.sum(correct_count == 0)),
                "mean_probability_ensemble_macro_recall": ensemble_metrics["macro_recall"],
                "mean_probability_ensemble_macro_f1": ensemble_metrics["macro_f1"],
                "mean_probability_ensemble_accuracy": ensemble_metrics["accuracy"],
            }
        )

        _, tiers = _tier_map(reference.labels)
        support = np.bincount(reference.labels, minlength=len(FINE_NAMES))
        for class_id, class_name in enumerate(FINE_NAMES):
            mask = reference.labels == class_id
            recalls = [
                float(condition.metrics["per_class"][class_id]["recall"])
                for condition in condition_values
            ]
            ensemble_class = ensemble_metrics["per_class"][class_id]
            per_class_rows.append(
                {
                    "condition": condition_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "support": int(support[class_id]),
                    "frequency_tier": tiers[class_id],
                    "seed42_recall": recalls[0],
                    "seed3407_recall": recalls[1],
                    "seed202625_recall": recalls[2],
                    "seed_recall_mean": mean(recalls),
                    "seed_recall_sample_std": stdev(recalls),
                    "seed_recall_range": max(recalls) - min(recalls),
                    "unanimous_prediction_rate": float(unanimous[mask].mean()),
                    "all_three_correct_rate": float(np.mean(correct_count[mask] == 3)),
                    "all_three_wrong_rate": float(np.mean(correct_count[mask] == 0)),
                    "ensemble_recall": ensemble_class["recall"],
                    "ensemble_f1": ensemble_class["f1"],
                }
            )

    for index, uid in enumerate(reference.annotation_uids):
        class_id = int(reference.labels[index])
        row: dict[str, Any] = {
            "annotation_uid": uid,
            "source_image_id": reference.source_image_ids[index],
            "fold": int(reference.folds[index]),
            "true_class_id": class_id,
            "true_class_name": FINE_NAMES[class_id],
        }
        for field, values in geometry.items():
            value = values[index]
            row[field] = value.item() if isinstance(value, np.generic) else value
        for condition_name in CONDITIONS:
            predictions = []
            correctness = []
            for seed in SEEDS:
                condition = conditions[(seed, condition_name)]
                prediction = int(condition.predictions[index])
                predictions.append(prediction)
                correctness.append(prediction == class_id)
                row[f"{condition_name}_seed{seed}_prediction"] = prediction
                row[f"{condition_name}_seed{seed}_correct"] = prediction == class_id
            row[f"{condition_name}_prediction_unanimous"] = len(set(predictions)) == 1
            row[f"{condition_name}_correct_seed_count"] = sum(correctness)
        object_rows.append(row)
    return summary_rows, per_class_rows, object_rows


def _frequency_tier_rows(
    conditions: Mapping[tuple[int, str], ConditionData],
) -> list[dict[str, Any]]:
    reference = conditions[(42, "clean")]
    support, tiers = _tier_map(reference.labels)
    rows: list[dict[str, Any]] = []
    for condition_name in CONDITIONS:
        for tier in ("tail", "middle", "head"):
            ids = [class_id for class_id in range(25) if tiers[class_id] == tier]
            values = []
            for seed in SEEDS:
                condition = conditions[(seed, condition_name)]
                macro_recall = mean(
                    float(condition.metrics["per_class"][class_id]["recall"]) for class_id in ids
                )
                values.append(macro_recall)
                rows.append(
                    {
                        "condition": condition_name,
                        "frequency_tier": tier,
                        "seed": seed,
                        "n_classes": len(ids),
                        "n_objects": int(support[ids].sum()),
                        "macro_recall": macro_recall,
                    }
                )
            rows.append(
                {
                    "condition": condition_name,
                    "frequency_tier": tier,
                    "seed": "mean_across_seeds",
                    "n_classes": len(ids),
                    "n_objects": int(support[ids].sum()),
                    "macro_recall": mean(values),
                    "macro_recall_seed_std": stdev(values),
                }
            )
    return rows


def _jitter_subset_rows(
    conditions: Mapping[tuple[int, str], ConditionData], geometry: Mapping[str, np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        clean = conditions[(seed, "clean")]
        jitter = conditions[(seed, "jitter")]
        clean_correct = clean.predictions == clean.labels
        jitter_correct = jitter.predictions == jitter.labels
        for subset, mask in _geometry_definitions(geometry):
            clean_metrics, _ = evaluate_classification(clean.labels[mask], clean.logits[mask])
            jitter_metrics, _ = evaluate_classification(jitter.labels[mask], jitter.logits[mask])
            rows.append(
                {
                    "seed": seed,
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
                }
            )
    summary_rows: list[dict[str, Any]] = []
    for subset in [name for name, _ in _geometry_definitions(geometry)]:
        values = [row for row in rows if row["subset"] == subset]
        macro = [float(row["macro_recall_delta"]) for row in values]
        accuracy = [float(row["accuracy_delta"]) for row in values]
        summary_rows.append(
            {
                "subset": subset,
                "n_objects": values[0]["n_objects"],
                "macro_recall_delta_seed_mean": mean(macro),
                "macro_recall_delta_seed_std": stdev(macro),
                "macro_recall_delta_min": min(macro),
                "macro_recall_delta_max": max(macro),
                "macro_recall_negative_seed_count": sum(value < 0 for value in macro),
                "accuracy_delta_seed_mean": mean(accuracy),
                "accuracy_delta_seed_std": stdev(accuracy),
                "accuracy_negative_seed_count": sum(value < 0 for value in accuracy),
            }
        )
    return rows, summary_rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seed42_clean_root = Path(args.seed42_clean_root).expanduser().resolve()
    seed42_jitter_root = Path(args.seed42_jitter_root).expanduser().resolve()
    task_root = Path(args.task_root).expanduser().resolve()
    task_archive = Path(args.task_archive).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    return_checksums = {
        "task02": _verify_return_checksums(seed42_clean_root),
        "task03": _verify_return_checksums(seed42_jitter_root),
        "task04": _verify_return_checksums(task_root),
    }
    if any(value.get("unexpected_mismatches") for value in return_checksums.values()):
        raise ValueError("服务器回传文件 checksum 异常")
    code_hashes = _verify_code_hashes(task_root, repo_root)
    if code_hashes["mismatches"]:
        raise ValueError(f"本地/服务器代码不一致: {code_hashes['mismatches']}")
    archive_sha = _sha256(task_archive)
    if archive_sha != EXPECTED_TASK_ARCHIVE_SHA256:
        raise ValueError("TASK-04 回传包 SHA-256 不一致")
    inventory = _parse_checkpoint_inventory(task_root)
    groups = _load_groups(seed42_clean_root, seed42_jitter_root, task_root)
    _verify_inventory_against_runs(inventory, groups)

    conditions = {key: _build_condition(runs) for key, runs in groups.items()}
    reference = conditions[(42, "clean")]
    for key, condition in conditions.items():
        if not np.array_equal(condition.annotation_uids, reference.annotation_uids):
            raise ValueError(f"{key} OOF 对象集不一致")
        if not np.array_equal(condition.source_image_ids, reference.source_image_ids):
            raise ValueError(f"{key} source_image_id 不一致")
        if not np.array_equal(condition.labels, reference.labels):
            raise ValueError(f"{key} 标签不一致")
        if not np.array_equal(condition.folds, reference.folds):
            raise ValueError(f"{key} fold 不一致")

    condition_rows = _condition_rows(groups, conditions)
    fold_rows = _fold_rows(groups)
    clean_to_jitter_rows = [
        _paired_row(
            f"seed{seed}_clean",
            f"seed{seed}_jitter",
            f"seed{seed}_clean_to_jitter",
            conditions[(seed, "clean")],
            conditions[(seed, "jitter")],
            args.bootstrap_resamples,
            20260717 + index,
        )
        for index, seed in enumerate(SEEDS)
    ]
    cross_seed_rows = []
    pair_index = 0
    for condition_name in CONDITIONS:
        for first_seed, second_seed in itertools.combinations(SEEDS, 2):
            cross_seed_rows.append(
                _paired_row(
                    f"seed{first_seed}_{condition_name}",
                    f"seed{second_seed}_{condition_name}",
                    f"{condition_name}_seed{first_seed}_to_seed{second_seed}",
                    conditions[(first_seed, condition_name)],
                    conditions[(second_seed, condition_name)],
                    args.bootstrap_resamples,
                    20260800 + pair_index,
                )
            )
            pair_index += 1
    two_way_rows = _two_way_rows(groups)
    geometry = _load_jitter_geometry(manifest, reference.annotation_uids)
    object_summary, per_class_rows, object_rows = _object_stability(conditions, geometry)
    tier_rows = _frequency_tier_rows(conditions)
    jitter_subset_rows, jitter_subset_summary = _jitter_subset_rows(conditions, geometry)

    _write_csv(output / "condition_summary.csv", condition_rows)
    _write_csv(output / "fold_metrics.csv", fold_rows)
    _write_csv(output / "clean_to_jitter_pairwise.csv", clean_to_jitter_rows)
    _write_csv(output / "cross_seed_pairwise.csv", cross_seed_rows)
    _write_csv(output / "two_way_decomposition.csv", two_way_rows)
    _write_csv(output / "object_stability_summary.csv", object_summary)
    _write_csv(output / "per_class_seed_stability.csv", per_class_rows)
    _write_csv(output / "frequency_tier_seed_stability.csv", tier_rows)
    _write_csv(output / "jitter_subset_metrics.csv", jitter_subset_rows)
    _write_csv(output / "jitter_subset_summary.csv", jitter_subset_summary)
    _write_csv(output / "object_seed_stability.csv", object_rows)

    integrity = {
        "status": "pass",
        "return_checksums": return_checksums,
        "code_hashes": code_hashes,
        "task_archive": {
            "path": str(task_archive),
            "size_bytes": task_archive.stat().st_size,
            "sha256": archive_sha,
        },
        "checkpoint_inventory": inventory,
        "manifest_sha256": _sha256(manifest),
        "n_training_runs": 9,
        "n_eval_only_runs": 9,
        "n_new_training_runs": 6,
        "n_new_eval_only_runs": 6,
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
        "clean_to_jitter": clean_to_jitter_rows,
        "cross_seed": cross_seed_rows,
        "two_way": two_way_rows,
        "object_stability": object_summary,
        "jitter_subset_summary": jitter_subset_summary,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
