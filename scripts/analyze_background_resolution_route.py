#!/usr/bin/env python3
"""Evaluate class-disjoint dual-resolution routes on Background-100MP."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rsdet.evaluation.background_stress import evaluate_background_stress
from rsdet.evaluation.coco import load_coco_predictions
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

OFFICIAL_LABELS = frozenset(range(25))


def parse_labels(text: str) -> frozenset[int]:
    values: set[int] = set()
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            first, last = token.split("-", 1)
            values.update(range(int(first), int(last) + 1))
        else:
            values.add(int(token))
    result = frozenset(values)
    if not result or result - OFFICIAL_LABELS:
        raise ValueError("labels must be a non-empty subset of 0..24")
    return result


def filter_predictions(
    predictions: dict[int, list[dict[str, Any]]],
    *,
    labels: frozenset[int],
    threshold: float,
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            row
            for row in rows
            if int(row["category_id"]) in labels and float(row["score"]) >= threshold
        ]
        for image_id, rows in predictions.items()
    }


def analyze(
    *,
    manifest: list[dict[str, Any]],
    primary: dict[int, list[dict[str, Any]]],
    expert: dict[int, list[dict[str, Any]]],
    primary_labels: frozenset[int],
    expert_labels: frozenset[int],
    primary_threshold: float,
    expert_threshold: float,
    category_mapping: dict[int, str],
) -> dict[str, Any]:
    if primary_labels & expert_labels or primary_labels | expert_labels != OFFICIAL_LABELS:
        raise ValueError("route label ownership must be a disjoint cover of 0..24")
    if not 0.0 <= primary_threshold <= 1.0 or not 0.0 <= expert_threshold <= 1.0:
        raise ValueError("thresholds must be in [0, 1]")
    primary_all = filter_predictions(
        primary, labels=OFFICIAL_LABELS, threshold=primary_threshold
    )
    expert_all = filter_predictions(expert, labels=OFFICIAL_LABELS, threshold=expert_threshold)
    routed_primary = filter_predictions(
        primary, labels=primary_labels, threshold=primary_threshold
    )
    routed_expert = filter_predictions(expert, labels=expert_labels, threshold=expert_threshold)
    route = {
        int(row["image_id"]): [
            *routed_primary.get(int(row["image_id"]), []),
            *routed_expert.get(int(row["image_id"]), []),
        ]
        for row in manifest
    }
    return {
        "schema_version": "background_100mp_resolution_route_v1",
        "primary_labels": sorted(primary_labels),
        "expert_labels": sorted(expert_labels),
        "primary_threshold": primary_threshold,
        "expert_threshold": expert_threshold,
        "primary": asdict(
            evaluate_background_stress(
                manifest, primary_all, category_mapping=category_mapping
            )
        ),
        "expert": asdict(
            evaluate_background_stress(
                manifest, expert_all, category_mapping=category_mapping
            )
        ),
        "route": asdict(
            evaluate_background_stress(manifest, route, category_mapping=category_mapping)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-pred", type=Path, required=True)
    parser.add_argument("--expert-pred", type=Path, required=True)
    parser.add_argument("--primary-labels", required=True)
    parser.add_argument("--expert-labels", required=True)
    parser.add_argument("--primary-threshold", type=float, required=True)
    parser.add_argument("--expert-threshold", type=float, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    result = analyze(
        manifest=manifest,
        primary=load_coco_predictions(args.primary_pred),
        expert=load_coco_predictions(args.expert_pred),
        primary_labels=parse_labels(args.primary_labels),
        expert_labels=parse_labels(args.expert_labels),
        primary_threshold=args.primary_threshold,
        expert_threshold=args.expert_threshold,
        category_mapping=dict(protocol.category_mapping),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"BACKGROUND_ROUTE_PASS fp_per_100mp={result['route']['false_positives_per_100mp']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
