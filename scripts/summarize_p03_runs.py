#!/usr/bin/env python3
"""P0-3 三折运行汇总和下一阶段候选条件选择。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 P0-3 run_summary.json 为三折条件表")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--regime", default="linear_probe", choices=("linear_probe", "fine_tune"))
    parser.add_argument("--sampler", default="natural", choices=("natural", "sqrt_inverse"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def _read_summaries(root: Path, regime: str, sampler: str, seed: int) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("run_summary.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        condition = value["condition"]
        if condition.get("smoke") or condition.get("eval_only"):
            continue
        if (
            condition["regime"] != regime
            or condition["sampler"] != sampler
            or int(condition["seed"]) != seed
        ):
            continue
        value["summary_path"] = str(path.resolve())
        summaries.append(value)
    return summaries


def _aggregate(summaries: Sequence[dict[str, Any]], allow_incomplete: bool) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        condition = summary["condition"]
        key = (
            str(condition["policy"]),
            int(condition["resolution"]),
            str(condition["regime"]),
            str(condition["sampler"]),
            int(condition["seed"]),
        )
        grouped[key].append(summary)
    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        folds = sorted(int(item["condition"]["fold"]) for item in values)
        if len(folds) != len(set(folds)):
            raise ValueError(f"{key} 存在重复 fold: {folds}")
        if folds != [0, 1, 2] and not allow_incomplete:
            raise ValueError(f"{key} 折数不完整: {folds}")

        def values_for(metric: str) -> list[float]:
            return [float(item["final_metrics"][metric]) for item in values]

        def aggregate_metric(metric: str) -> tuple[float, float]:
            numbers = values_for(metric)
            return mean(numbers), stdev(numbers) if len(numbers) > 1 else 0.0

        macro_recall_mean, macro_recall_std = aggregate_metric("macro_recall")
        macro_f1_mean, macro_f1_std = aggregate_metric("macro_f1")
        accuracy_mean, accuracy_std = aggregate_metric("accuracy")
        aircraft = [float(item["aircraft20"]["macro_recall"]) for item in values]
        rows.append(
            {
                "policy": key[0],
                "resolution": key[1],
                "regime": key[2],
                "sampler": key[3],
                "seed": key[4],
                "folds": ";".join(str(fold) for fold in folds),
                "n_folds": len(folds),
                "macro_recall_mean": macro_recall_mean,
                "macro_recall_std": macro_recall_std,
                "macro_f1_mean": macro_f1_mean,
                "macro_f1_std": macro_f1_std,
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "aircraft20_macro_recall_mean": mean(aircraft),
                "aircraft20_macro_recall_std": stdev(aircraft) if len(aircraft) > 1 else 0.0,
                "n_val_total": sum(int(item["n_val"]) for item in values),
            }
        )
    return rows


def _rank_key(row: dict[str, Any], best_score: float) -> tuple[Any, ...]:
    # 0.5 个百分点内视为工程近似并列，再用 F1、较低分辨率和 tight 打破并列。
    equivalent = best_score - float(row["macro_recall_mean"]) <= 0.005
    return (
        0 if equivalent else 1,
        -float(row["macro_f1_mean"]) if equivalent else -float(row["macro_recall_mean"]),
        int(row["resolution"]),
        0 if row["policy"] == "tight" else 1,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.runs_root).expanduser().resolve()
    summaries = _read_summaries(root, args.regime, args.sampler, args.seed)
    if not summaries:
        raise ValueError(f"{root} 未找到符合条件的非 smoke run_summary.json")
    rows = _aggregate(summaries, args.allow_incomplete)
    if not rows:
        raise ValueError("无可汇总条件")
    best_score = max(float(row["macro_recall_mean"]) for row in rows)
    ranked = sorted(rows, key=lambda row: _rank_key(row, best_score))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = ["rank", *[key for key in ranked[0] if key != "rank"]]
    with (output / "aggregate.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ranked)
    selection = {
        "selection_rule": {
            "primary": "3-fold mean macro_recall",
            "practical_equivalence_tolerance_absolute": 0.005,
            "tie_breakers": ["macro_f1", "lower_resolution", "tight_policy"],
        },
        "selected_for_fine_tune": [
            {"policy": row["policy"], "resolution": row["resolution"]}
            for row in ranked[: min(2, len(ranked))]
        ],
        "ranking": ranked,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
