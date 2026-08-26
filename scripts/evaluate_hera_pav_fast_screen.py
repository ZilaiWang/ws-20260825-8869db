#!/usr/bin/env python3
"""Label-blind fixed-fusion PAV screen on one held-out formal CV3 fold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_frontier import official_fixed_risk_frontier
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.hera_guard.resolver import resolve_fine_category
from rsdet.utils.config import load_config

FUSIONS = {
    "baseline": (0.0, 0.0, 0.0, 0.0),
    "guard_soft": (0.50, 0.50, 0.25, 0.25),
    "guard_balanced": (0.75, 0.50, 0.25, 0.25),
    "guard_strong": (1.00, 0.50, 0.25, 0.25),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - value.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pav-logits", type=Path, required=True)
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--held-out-fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        manifest = {int(row["candidate_id"]): row for row in csv.DictReader(handle)}
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    logits = np.load(args.pav_logits, allow_pickle=False)
    candidate_ids = logits["candidate_id"].astype(np.int64)
    if len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError("PAV logits contain duplicate candidate IDs")
    expected_ids = {
        candidate_id
        for candidate_id, row in manifest.items()
        if int(row["fold"]) == args.held_out_fold
    }
    if set(candidate_ids.tolist()) != expected_ids:
        raise ValueError("PAV held-out coverage differs from manifest")
    image_ids = {
        int(obj.image_id)
        for obj in formal.objects.values()
        if int(obj.fold) == args.held_out_fold
    }
    fg = logits["foreground_logit"].astype(np.float64)
    quality = logits["quality_logit"].astype(np.float64)
    protect = logits["protect_logit"].astype(np.float64)
    fine_probability = _softmax(logits["fine_logits"].astype(np.float64))
    coarse_of_fine = [
        {"aircraft": 0, "ship": 1, "vehicle": 2}[protocol.category_mapping[index]]
        for index in range(25)
    ]
    by_candidate = {
        int(candidate_id): index for index, candidate_id in enumerate(candidate_ids.tolist())
    }

    def summarize(frontier) -> dict:
        fixed = {
            str(level): {
                "recall": frontier.points[level].recall,
                "fdr": frontier.points[level].fdr,
                "tp": frontier.points[level].tp,
                "fp": frontier.points[level].fp,
            }
            for level in levels
        }
        selected_ids = set(frontier.selected_candidate_ids[0.12])
        selected_by_image = {
            image_id: [
                row
                for row in records
                if int(row["source_prediction_index"]) in selected_ids
            ]
            for image_id, records in frontier.kept_predictions.items()
        }
        scoped_gt = {image_id: formal.boxes.get(image_id, []) for image_id in image_ids}
        ranking = evaluate_ranking_metrics(
            scoped_gt,
            selected_by_image,
            class_names=list(dict.fromkeys(protocol.category_mapping.values())),
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
            require_complete_taxonomy=True,
        )
        six = []
        coarse = {}
        for coarse_name in ("ship", "aircraft", "vehicle"):
            metric = ranking.per_coarse[coarse_name]
            coarse[coarse_name] = {
                "macro_recall": metric.macro_recall,
                "macro_fdr": metric.macro_fdr,
                "pooled_recall": metric.pooled_recall,
                "pooled_fdr": metric.pooled_fdr,
            }
            six.extend((metric.macro_recall, 1.0 - metric.macro_fdr))
        return {
            "frontier": fixed,
            "ranking_at_fdr_0.12": {
                "per_coarse": coarse,
                "six_metric_min": min(six),
                "six_metric_mean": sum(six) / len(six),
            },
        }

    def build_variant(
        *,
        rho: float,
        fg_weight: float,
        quality_weight: float,
        protect_weight: float,
        relabel: bool,
    ) -> tuple[list[dict], int]:
        result = []
        changed = 0
        for candidate_id, raw in enumerate(predictions):
            if candidate_id not in by_candidate:
                continue
            local = by_candidate[candidate_id]
            record = dict(raw)
            base_score = min(max(float(record["score"]), 1e-7), 1 - 1e-7)
            base_logit = math.log(base_score / (1 - base_score))
            evidence = (
                fg_weight * fg[local]
                + quality_weight * quality[local]
                + protect_weight * protect[local]
            )
            record["score"] = 1 / (1 + math.exp(-(base_logit + rho * math.tanh(evidence))))
            record["source_prediction_index"] = candidate_id
            if relabel:
                resolution = resolve_fine_category(
                    detector_category_id=int(record["category_id"]),
                    fine_probabilities=fine_probability[local],
                    coarse_of_fine=coarse_of_fine,
                    minimum_probability=0.85,
                    minimum_margin=0.40,
                    protect_probability=float(_sigmoid(protect[local : local + 1])[0]),
                    maximum_protect_for_change=0.40,
                )
                record["category_id"] = resolution.category_id
                changed += int(resolution.changed)
            result.append(record)
        return result, changed

    rows = {}
    levels = (0.15, 0.12, 0.11, 0.10)
    for name, (rho, fg_weight, quality_weight, protect_weight) in FUSIONS.items():
        variant, changed = build_variant(
            rho=rho,
            fg_weight=fg_weight,
            quality_weight=quality_weight,
            protect_weight=protect_weight,
            relabel=False,
        )
        frontier = official_fixed_risk_frontier(
            gt_boxes=formal.boxes,
            predictions=variant,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
            image_ids=image_ids,
            fdr_levels=levels,
            nms_iou=0.50,
        )
        rows[name] = {
            "rho": rho,
            "fg_weight": fg_weight,
            "quality_weight": quality_weight,
            "protect_weight": protect_weight,
            "relabel_count": changed,
        }
        rows[name].update(summarize(frontier))
    relabel_variant, changed = build_variant(
        rho=0.50,
        fg_weight=0.50,
        quality_weight=0.25,
        protect_weight=0.25,
        relabel=True,
    )
    relabel_frontier = official_fixed_risk_frontier(
        gt_boxes=formal.boxes,
        predictions=relabel_variant,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        image_ids=image_ids,
        fdr_levels=levels,
        nms_iou=0.50,
    )
    rows["guard_soft_plus_safe_relabel"] = {
        "rho": 0.50,
        "relabel_count": changed,
    }
    rows["guard_soft_plus_safe_relabel"].update(summarize(relabel_frontier))
    baseline_012 = rows["baseline"]["frontier"]["0.12"]["recall"]
    baseline_010 = rows["baseline"]["frontier"]["0.1"]["recall"]
    baseline_min_six = rows["baseline"]["ranking_at_fdr_0.12"]["six_metric_min"]
    decisions = []
    for name, row in rows.items():
        if name == "baseline":
            continue
        delta_012 = row["frontier"]["0.12"]["recall"] - baseline_012
        delta_010 = row["frontier"]["0.1"]["recall"] - baseline_010
        delta_min_six = (
            row["ranking_at_fdr_0.12"]["six_metric_min"] - baseline_min_six
        )
        decisions.append(
            {
                "variant": name,
                "delta_recall_at_fdr_0.12": delta_012,
                "delta_recall_at_fdr_0.10": delta_010,
                "delta_six_metric_min_at_fdr_0.12": delta_min_six,
                "passes_exploratory_gate": (
                    delta_012 >= 0.002
                    and delta_010 >= -0.001
                    and delta_min_six >= -0.005
                ),
            }
        )
    output = {
        "status": "complete_exploratory_fast_screen",
        "formal_admission": False,
        "held_out_fold": args.held_out_fold,
        "n_candidates": len(candidate_ids),
        "variants": rows,
        "decisions": decisions,
        "next_action": (
            "expand_pav_to_three_outer_folds"
            if any(row["passes_exploratory_gate"] for row in decisions)
            else "stop_or_redesign_pav"
        ),
        "input_sha256": {
            "predictions": _sha256(args.predictions),
            "manifest": _sha256(args.manifest),
            "pav_logits": _sha256(args.pav_logits),
            "formal_crop_manifest": _sha256(args.formal_crop_manifest),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
