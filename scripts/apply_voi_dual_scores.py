#!/usr/bin/env python3
"""Apply seven-channel scores only to a frozen VOI budget per 10K image."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.hera_guard.metric_risk import align_anchor_scores
from rsdet.hera_guard.voi import select_voi_budget


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--evidence-pred", type=Path, required=True)
    parser.add_argument("--anchor-pred", type=Path, required=True)
    parser.add_argument("--dual-pred", type=Path, required=True)
    parser.add_argument("--baseline-frontier", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets-per-image", type=int, nargs="+", default=(32, 64, 128, 256))
    parser.add_argument("--fdr-level", type=float, default=0.15)
    args = parser.parse_args()
    if any(value < 0 for value in args.budgets_per_image):
        raise ValueError("VOI budgets must be non-negative")

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    image_sizes = {
        int(row["id"]): (int(row["width"]), int(row["height"])) for row in gt["images"]
    }
    evidence: list[dict[str, Any]] = json.loads(args.evidence_pred.read_text(encoding="utf-8"))
    anchor: list[dict[str, Any]] = json.loads(args.anchor_pred.read_text(encoding="utf-8"))
    dual: list[dict[str, Any]] = json.loads(args.dual_pred.read_text(encoding="utf-8"))
    anchor_scores = align_anchor_scores(evidence, anchor)
    dual_scores = align_anchor_scores(evidence, dual)
    frontier = json.loads(args.baseline_frontier.read_text(encoding="utf-8"))
    thresholds = {
        int(key): float(value)
        for key, value in frontier["frontiers"][f"{args.fdr_level:.3f}"][
            "crossfit_thresholds"
        ].items()
    }
    by_image: dict[int, list[int]] = defaultdict(list)
    priority_records = []
    for index, (row, score) in enumerate(zip(evidence, anchor_scores, strict=True)):
        item = {**row, "score": float(score)}
        priority_records.append(item)
        by_image[int(row["image_id"])].append(index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for budget in sorted(set(args.budgets_per_image)):
        selected_global: set[int] = set()
        reason_counts: Counter[str] = Counter()
        for image_id, global_indices in sorted(by_image.items()):
            fold = int(evidence[global_indices[0]]["source_fold"])
            local = [priority_records[index] for index in global_indices]
            selected = select_voi_budget(
                local,
                image_sizes,
                budget=budget,
                decision_threshold=thresholds[fold],
            )
            for decision in selected:
                selected_global.add(global_indices[decision.candidate_index])
                reason_counts.update(decision.reasons)
        predictions = []
        for index, row in enumerate(evidence):
            predictions.append(
                {
                    "image_id": int(row["image_id"]),
                    "category_id": int(row["category_id"]),
                    "bbox": [float(value) for value in row["bbox"]],
                    "score": float(
                        dual_scores[index] if index in selected_global else anchor_scores[index]
                    ),
                    "source_fold": int(row["source_fold"]),
                    "source_model": str(row.get("source_model", "")),
                }
            )
        path = args.output_dir / f"voi_budget_{budget:04d}_predictions.json"
        path.write_text(json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8")
        outputs[str(budget)] = {
            "selected": len(selected_global),
            "selected_per_image_max": budget,
            "reason_counts": dict(sorted(reason_counts.items())),
            "predictions": path.name,
            "predictions_sha256": _sha256(path),
        }
    summary = {
        "status": "voi_dual_score_budgets_complete",
        "protocol": "frozen_baseline_threshold_deployable_voi_v1",
        "fdr_level": args.fdr_level,
        "thresholds": thresholds,
        "input_candidates": len(evidence),
        "outputs": outputs,
        "input_sha256": {
            "gt": _sha256(args.gt),
            "evidence_pred": _sha256(args.evidence_pred),
            "anchor_pred": _sha256(args.anchor_pred),
            "dual_pred": _sha256(args.dual_pred),
            "baseline_frontier": _sha256(args.baseline_frontier),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
