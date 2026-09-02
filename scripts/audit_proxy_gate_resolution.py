#!/usr/bin/env python3
"""Audit whether fixed recall-delta gates are resolvable on pseudo benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def coarse(category: int) -> str:
    return "ship" if category <= 3 else "aircraft" if category <= 23 else "vehicle"


def audit(name: str, path: Path, fixed_delta: float) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    image_fold: dict[int, int] = {}
    for image in payload["images"]:
        image_id = int(image["id"])
        file_name = str(image.get("file_name", ""))
        fold = next((value for value in (0, 1, 2) if file_name.startswith(f"fold{value}_")), None)
        if fold is None:
            raise ValueError(f"cannot infer fold from {file_name}")
        image_fold[image_id] = fold
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in payload["annotations"]:
        counts[image_fold[int(row["image_id"])]][coarse(int(row["category_id"]))] += 1
    folds = {}
    for fold in (0, 1, 2):
        folds[str(fold)] = {}
        for group in ("ship", "aircraft", "vehicle"):
            total = counts[fold][group]
            if total <= 0:
                raise ValueError(f"{name} fold {fold} has no {group} GT")
            recall_se_at_085 = math.sqrt(0.85 * 0.15 / total)
            folds[str(fold)][group] = {
                "gt": total,
                "one_object_pp": 100.0 / total,
                "fixed_gate_object_budget": fixed_delta * total,
                "fixed_gate_allows_one_miss": fixed_delta * total >= 1.0,
                "binomial_recall_se_pp_at_r085": 100.0 * recall_se_at_085,
                "discovery_tolerance_two_objects": max(fixed_delta, 2.0 / total),
            }
    return {"name": name, "ground_truth": str(path.resolve()), "folds": folds}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", action="append", required=True, help="NAME=ground_truth.json")
    parser.add_argument("--fixed-delta", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proxies = []
    for raw in args.proxy:
        name, separator, path = raw.partition("=")
        if not separator:
            raise ValueError("--proxy must be NAME=PATH")
        proxies.append(audit(name, Path(path), args.fixed_delta))
    result = {
        "schema_version": "proxy_gate_resolution_audit_v1",
        "fixed_delta": args.fixed_delta,
        "interpretation": (
            "If fixed_gate_allows_one_miss is false, a 0.5pp non-degradation veto "
            "permits zero additional misses and is unsuitable as a discovery gate."
        ),
        "proxies": proxies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
