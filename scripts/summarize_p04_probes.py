#!/usr/bin/env python3
"""P04 线性探针三折汇总与 P03 等价门禁。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 P04 feature probe runs")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-fold-macro-recall")
    parser.add_argument("--equivalence-tolerance", type=float, default=0.003)
    return parser.parse_args(argv)


def _group_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        summary["feature_name"],
        summary["cache_fingerprint"],
        summary["manifest_sha256"],
        summary["seed"],
        summary["normalization"],
        summary["pca_dim"],
        summary.get("head_init", "p04_default"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.runs_root).expanduser().resolve()
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("run_summary.json"))
    ]
    if not summaries:
        raise FileNotFoundError(f"{root} 下无 run_summary.json")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for summary in summaries:
        groups.setdefault(_group_key(summary), []).append(summary)
    rows: list[dict[str, Any]] = []
    for key, values in groups.items():
        folds = sorted(int(value["fold"]) for value in values)
        if folds != [0, 1, 2]:
            raise ValueError(f"分组 {key} folds 不完整: {folds}")
        ordered = sorted(values, key=lambda value: int(value["fold"]))
        metric_names = ("macro_recall", "macro_f1", "accuracy", "aircraft20_macro_recall")
        row: dict[str, Any] = {
            "feature_name": key[0],
            "cache_fingerprint": key[1],
            "manifest_sha256": key[2],
            "seed": key[3],
            "normalization": key[4],
            "pca_dim": key[5],
            "head_init": key[6],
            "folds": folds,
        }
        for metric in metric_names:
            scores = np.asarray([value["metrics"][metric] for value in ordered], dtype=float)
            row[f"{metric}_fold_values"] = scores.tolist()
            row[f"{metric}_mean"] = float(scores.mean())
            row[f"{metric}_std"] = float(scores.std(ddof=1))
        rows.append(row)

    result: dict[str, Any] = {"status": "complete", "groups": rows}
    if args.expected_fold_macro_recall:
        if len(rows) != 1:
            raise ValueError("等价门禁要求 runs-root 只包含一组三折")
        expected = np.asarray(
            [float(value) for value in args.expected_fold_macro_recall.split(",")],
            dtype=float,
        )
        if expected.shape != (3,):
            raise ValueError("expected-fold-macro-recall 必须有 3 个逗号分隔数")
        actual = np.asarray(rows[0]["macro_recall_fold_values"], dtype=float)
        mean_delta = float(actual.mean() - expected.mean())
        per_fold_delta = actual - expected
        passed = bool(abs(mean_delta) <= args.equivalence_tolerance)
        result["equivalence_gate"] = {
            "status": "pass" if passed else "fail",
            "expected_fold_macro_recall": expected.tolist(),
            "actual_fold_macro_recall": actual.tolist(),
            "per_fold_delta": per_fold_delta.tolist(),
            "mean_delta": mean_delta,
            "absolute_mean_delta": abs(mean_delta),
            "tolerance": args.equivalence_tolerance,
            "note": "均值是硬门禁；单折差异必须在报告中单独解释",
        }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    if result.get("equivalence_gate", {}).get("status") == "fail":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
