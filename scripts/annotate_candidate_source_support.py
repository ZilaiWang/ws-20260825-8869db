#!/usr/bin/env python3
"""Attach deployable multi-source agreement evidence to merged candidates.

The operation never uses ground truth.  For every retained proposal it records
the maximum same-fine IoU and the best score above a fixed IoU threshold from
each detector/view source.  These fields allow the fold-heldout risk head to
distinguish a single-model hallucination from a proposal supported by another
scale or model family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    if not label:
        raise argparse.ArgumentTypeError("source label cannot be empty")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"source path does not exist: {path}")
    return label, path


def _xywh_array(rows: Iterable[dict[str, Any]]) -> np.ndarray:
    values = []
    for item in rows:
        x, y, width, height = (float(value) for value in item["bbox"])
        if width <= 0.0 or height <= 0.0:
            raise ValueError("source candidate has non-positive bbox")
        values.append((x, y, x + width, y + height))
    return np.asarray(values, dtype=np.float64).reshape((-1, 4))


def pairwise_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.ndim != 2 or boxes_a.shape[1:] != (4,):
        raise ValueError("boxes_a must have shape (N, 4)")
    if boxes_b.ndim != 2 or boxes_b.shape[1:] != (4,):
        raise ValueError("boxes_b must have shape (M, 4)")
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=2)
    area_a = np.prod(boxes_a[:, 2:] - boxes_a[:, :2], axis=1)[:, None]
    area_b = np.prod(boxes_b[:, 2:] - boxes_b[:, :2], axis=1)[None, :]
    union = area_a + area_b - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)


def annotate_source_support(
    candidates: list[dict[str, Any]],
    sources: list[tuple[str, list[dict[str, Any]]]],
    *,
    support_iou: float,
) -> list[dict[str, Any]]:
    if not 0.0 < support_iou <= 1.0:
        raise ValueError("support_iou must be in (0, 1]")
    labels = [label for label, _ in sources]
    if len(set(labels)) != len(labels):
        raise ValueError("source labels must be unique")
    source_groups: dict[str, dict[tuple[int, int], list[dict[str, Any]]]] = {}
    for label, rows in sources:
        groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for item in rows:
            score = float(item["score"])
            if not math.isfinite(score):
                raise ValueError("source score contains NaN/Inf")
            groups[(int(item["image_id"]), int(item["category_id"]))].append(item)
        source_groups[label] = groups

    candidate_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, item in enumerate(candidates):
        candidate_groups[(int(item["image_id"]), int(item["category_id"]))].append(index)
    output = [dict(item) for item in candidates]
    for key, candidate_indices in candidate_groups.items():
        candidate_boxes = _xywh_array(candidates[index] for index in candidate_indices)
        per_label_support: dict[str, np.ndarray] = {}
        per_label_iou: dict[str, np.ndarray] = {}
        per_label_score: dict[str, np.ndarray] = {}
        for label in labels:
            rows = source_groups[label].get(key, [])
            if rows:
                ious = pairwise_iou(candidate_boxes, _xywh_array(rows))
                scores = np.asarray([float(item["score"]) for item in rows], dtype=np.float64)
                max_iou = ious.max(axis=1)
                matched_scores = np.where(ious >= support_iou, scores[None, :], 0.0).max(axis=1)
            else:
                max_iou = np.zeros(len(candidate_indices), dtype=np.float64)
                matched_scores = np.zeros(len(candidate_indices), dtype=np.float64)
            per_label_iou[label] = max_iou
            per_label_score[label] = matched_scores
            per_label_support[label] = matched_scores > 0.0
        for local_index, candidate_index in enumerate(candidate_indices):
            row = output[candidate_index]
            support_count = 0
            support_score_sum = 0.0
            for label in labels:
                max_iou = float(per_label_iou[label][local_index])
                matched_score = float(per_label_score[label][local_index])
                row[f"support_{label}_max_iou"] = max_iou
                row[f"support_{label}_score"] = matched_score
                support_count += int(per_label_support[label][local_index])
                support_score_sum += matched_score
            has_m3 = "m3_id" in per_label_support and bool(
                per_label_support["m3_id"][local_index]
            )
            has_y5 = any(
                bool(per_label_support[label][local_index])
                for label in labels
                if label != "m3_id"
            )
            row["source_support_count"] = support_count
            row["source_support_score_sum"] = support_score_sum
            row["heterogeneous_support"] = int(has_m3 and has_y5)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source", type=parse_source, action="append", required=True)
    parser.add_argument("--support-iou", type=float, default=0.50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    loaded = [
        (label, json.loads(path.read_text(encoding="utf-8")))
        for label, path in args.source
    ]
    output = annotate_source_support(candidates, loaded, support_iou=args.support_iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
    counts: dict[str, int] = defaultdict(int)
    for item in output:
        counts[str(int(item["source_support_count"]))] += 1
    summary = {
        "status": "complete",
        "protocol": "deployable_same_fine_multi_source_support_v1",
        "warning": "No ground-truth data are read by this annotation step.",
        "support_iou": args.support_iou,
        "candidates": {
            "path": str(args.candidates.resolve()),
            "sha256": _sha256(args.candidates),
            "count": len(candidates),
        },
        "sources": [
            {"label": label, "path": str(path), "sha256": _sha256(path)}
            for label, path in args.source
        ],
        "support_count_distribution": dict(sorted(counts.items())),
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
