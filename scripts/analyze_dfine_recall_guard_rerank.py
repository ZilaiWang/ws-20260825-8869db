#!/usr/bin/env python3
"""Audit D-FINE agreement reranking under an incumbent-recall guard.

The Y5 baseline stays at its deployed threshold.  On the two training folds,
the product-score threshold is chosen to minimize FDR while retaining at least
``baseline recall - allowed loss``.  The held-out fold is evaluated once with
the exact repository official matcher.  D-FINE never contributes a new box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_dfine_agreement_recovery import (  # noqa: E402
    _annotate_support,
    _coarse_mapping,
    _filter_coarse,
    _metrics,
)

from rsdet.evaluation.coco import (  # noqa: E402
    load_coco_ground_truth,
    load_coco_predictions,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol  # noqa: E402
from rsdet.postprocess.calibration import build_threshold_grid  # noqa: E402
from rsdet.utils.config import load_config  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reranked(
    rows: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            dict(item, score=float(item["agreement_product"])) for item in items
        ]
        for image_id, items in rows.items()
    }


def _select_threshold(
    gt: dict[int, list[dict[str, Any]]],
    reranked: dict[int, list[dict[str, Any]]],
    *,
    image_ids: set[int],
    baseline_recall: float,
    allowed_recall_loss: float,
    thresholds: list[float],
    coarse: str,
    mapping: dict[int, str],
    iou_threshold: float,
) -> dict[str, Any]:
    recall_floor = max(0.0, baseline_recall - allowed_recall_loss)
    feasible: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = _metrics(
            gt,
            _filter_coarse(
                reranked,
                mapping=mapping,
                threshold=threshold,
                image_ids=image_ids,
            ),
            coarse=coarse,
            mapping=mapping,
            iou_threshold=iou_threshold,
        )
        if metrics["recall"] >= recall_floor:
            feasible.append({"threshold": threshold, "metrics": metrics})
    if not feasible:
        raise RuntimeError("no product threshold satisfies the recall guard")
    selected = min(
        feasible,
        key=lambda row: (
            row["metrics"]["fdr"],
            -row["metrics"]["recall"],
            -row["threshold"],
        ),
    )
    selected["recall_floor"] = recall_floor
    return selected


def _select_each_fold_threshold(
    fold_payloads: list[dict[str, Any]],
    *,
    allowed_recall_loss: float,
    thresholds: list[float],
    coarse: str,
    mapping: dict[int, str],
    iou_threshold: float,
) -> dict[str, Any]:
    pooled_gt: dict[int, list[dict[str, Any]]] = {}
    pooled_reranked: dict[int, list[dict[str, Any]]] = {}
    feasible: list[dict[str, Any]] = []
    for payload in fold_payloads:
        pooled_gt.update(payload["gt"])
        pooled_reranked.update(payload["reranked"])
    for threshold in thresholds:
        per_fold: dict[str, dict[str, Any]] = {}
        passes = True
        for payload in fold_payloads:
            metrics = _metrics(
                payload["gt"],
                _filter_coarse(
                    payload["reranked"],
                    mapping=mapping,
                    threshold=threshold,
                    image_ids=payload["image_ids"],
                ),
                coarse=coarse,
                mapping=mapping,
                iou_threshold=iou_threshold,
            )
            recall_floor = max(
                0.0,
                float(payload["baseline"]["recall"]) - allowed_recall_loss,
            )
            metrics["recall_floor"] = recall_floor
            per_fold[str(payload["fold"])] = metrics
            passes &= metrics["recall"] >= recall_floor
        if not passes:
            continue
        pooled = _metrics(
            pooled_gt,
            _filter_coarse(
                pooled_reranked,
                mapping=mapping,
                threshold=threshold,
                image_ids=set(pooled_gt),
            ),
            coarse=coarse,
            mapping=mapping,
            iou_threshold=iou_threshold,
        )
        feasible.append(
            {"threshold": threshold, "metrics": pooled, "per_fold": per_fold}
        )
    if not feasible:
        raise RuntimeError("no product threshold satisfies every-fold recall guard")
    return min(
        feasible,
        key=lambda row: (
            row["metrics"]["fdr"],
            max(item["fdr"] for item in row["per_fold"].values()),
            -row["metrics"]["recall"],
            -row["threshold"],
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-threshold", type=float, default=0.15)
    parser.add_argument(
        "--allowed-recall-losses",
        type=float,
        nargs="+",
        default=(0.0, 0.005, 0.01),
    )
    parser.add_argument(
        "--guard-scope",
        choices=("pooled", "each_fold"),
        default="each_fold",
    )
    parser.add_argument("--coarse", default="vehicle")
    parser.add_argument("--agreement-iou", type=float, default=0.35)
    parser.add_argument("--threshold-step", type=float, default=0.001)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not 0.0 <= args.primary_threshold <= 1.0:
        raise ValueError("primary-threshold must be in [0, 1]")
    if any(value < 0.0 or value > 1.0 for value in args.allowed_recall_losses):
        raise ValueError("allowed recall losses must be in [0, 1]")
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    mapping = _coarse_mapping(protocol, args.coarse)
    official_iou = float(protocol.iou_thresholds[args.coarse])
    grid = build_threshold_grid(0.001, 1.0, args.threshold_step)

    folds: dict[int, dict[str, Any]] = {}
    input_sha: dict[str, dict[str, str]] = {}
    seen_images: set[int] = set()
    for fold in (0, 1, 2):
        folder = args.fold_root / f"fold_{fold}"
        paths = {
            "gt": folder / "instances_val.json",
            "primary": folder / "y5_predictions.json",
            "specialist": folder / "dfine_predictions.json",
        }
        raw_gt = json.loads(paths["gt"].read_text(encoding="utf-8"))
        image_ids = {int(item["id"]) for item in raw_gt["images"]}
        if seen_images & image_ids:
            raise ValueError("image IDs overlap across folds")
        seen_images |= image_ids
        gt = _filter_coarse(
            load_coco_ground_truth(paths["gt"]),
            mapping=mapping,
            image_ids=image_ids,
        )
        primary = load_coco_predictions(paths["primary"])
        supported = _annotate_support(
            primary,
            load_coco_predictions(paths["specialist"]),
            image_ids=image_ids,
            mapping=mapping,
            iou_threshold=args.agreement_iou,
        )
        folds[fold] = {
            "image_ids": image_ids,
            "gt": gt,
            "primary": primary,
            "reranked": _reranked(supported),
        }
        input_sha[str(fold)] = {name: _sha256(path) for name, path in paths.items()}

    output: dict[str, Any] = {
        "status": "complete_diagnostic_recall_guard_rerank",
        "protocol": "cv3_product_rerank_incumbent_recall_guard_v1",
        "coarse": args.coarse,
        "primary_threshold": args.primary_threshold,
        "agreement_iou": args.agreement_iou,
        "input_sha256": input_sha,
        "levels": {},
    }
    for allowed_loss in args.allowed_recall_losses:
        pooled_gt: dict[int, list[dict[str, Any]]] = {}
        pooled_baseline: dict[int, list[dict[str, Any]]] = {}
        pooled_candidate: dict[int, list[dict[str, Any]]] = {}
        heldout_payload: dict[str, Any] = {}
        for held_out in (0, 1, 2):
            train_folds = [fold for fold in (0, 1, 2) if fold != held_out]
            train_images = set().union(*(folds[fold]["image_ids"] for fold in train_folds))
            train_gt: dict[int, list[dict[str, Any]]] = {}
            train_primary: dict[int, list[dict[str, Any]]] = {}
            train_reranked: dict[int, list[dict[str, Any]]] = {}
            for fold in train_folds:
                train_gt.update(folds[fold]["gt"])
                train_primary.update(folds[fold]["primary"])
                train_reranked.update(folds[fold]["reranked"])
            train_baseline = _metrics(
                train_gt,
                _filter_coarse(
                    train_primary,
                    mapping=mapping,
                    threshold=args.primary_threshold,
                    image_ids=train_images,
                ),
                coarse=args.coarse,
                mapping=mapping,
                iou_threshold=official_iou,
            )
            if args.guard_scope == "pooled":
                selected = _select_threshold(
                    train_gt,
                    train_reranked,
                    image_ids=train_images,
                    baseline_recall=float(train_baseline["recall"]),
                    allowed_recall_loss=allowed_loss,
                    thresholds=grid,
                    coarse=args.coarse,
                    mapping=mapping,
                    iou_threshold=official_iou,
                )
            else:
                guard_payloads: list[dict[str, Any]] = []
                for fold in train_folds:
                    fold_baseline = _metrics(
                        folds[fold]["gt"],
                        _filter_coarse(
                            folds[fold]["primary"],
                            mapping=mapping,
                            threshold=args.primary_threshold,
                            image_ids=folds[fold]["image_ids"],
                        ),
                        coarse=args.coarse,
                        mapping=mapping,
                        iou_threshold=official_iou,
                    )
                    guard_payloads.append(
                        {
                            "fold": fold,
                            "image_ids": folds[fold]["image_ids"],
                            "gt": folds[fold]["gt"],
                            "reranked": folds[fold]["reranked"],
                            "baseline": fold_baseline,
                        }
                    )
                selected = _select_each_fold_threshold(
                    guard_payloads,
                    allowed_recall_loss=allowed_loss,
                    thresholds=grid,
                    coarse=args.coarse,
                    mapping=mapping,
                    iou_threshold=official_iou,
                )
            heldout_images = folds[held_out]["image_ids"]
            baseline_pred = _filter_coarse(
                folds[held_out]["primary"],
                mapping=mapping,
                threshold=args.primary_threshold,
                image_ids=heldout_images,
            )
            candidate_pred = _filter_coarse(
                folds[held_out]["reranked"],
                mapping=mapping,
                threshold=float(selected["threshold"]),
                image_ids=heldout_images,
            )
            baseline = _metrics(
                folds[held_out]["gt"],
                baseline_pred,
                coarse=args.coarse,
                mapping=mapping,
                iou_threshold=official_iou,
            )
            candidate = _metrics(
                folds[held_out]["gt"],
                candidate_pred,
                coarse=args.coarse,
                mapping=mapping,
                iou_threshold=official_iou,
            )
            heldout_payload[str(held_out)] = {
                "train_baseline": train_baseline,
                "train_selection": selected,
                "heldout_baseline": baseline,
                "heldout_candidate": candidate,
                "heldout_delta": {
                    "tp": candidate["tp"] - baseline["tp"],
                    "fp": candidate["fp"] - baseline["fp"],
                    "recall": candidate["recall"] - baseline["recall"],
                    "fdr": candidate["fdr"] - baseline["fdr"],
                },
            }
            pooled_gt.update(folds[held_out]["gt"])
            pooled_baseline.update(baseline_pred)
            pooled_candidate.update(candidate_pred)
        baseline = _metrics(
            pooled_gt,
            pooled_baseline,
            coarse=args.coarse,
            mapping=mapping,
            iou_threshold=official_iou,
        )
        candidate = _metrics(
            pooled_gt,
            pooled_candidate,
            coarse=args.coarse,
            mapping=mapping,
            iou_threshold=official_iou,
        )
        output["levels"][f"{allowed_loss:.3f}"] = {
            "heldout": heldout_payload,
            "pooled_baseline": baseline,
            "pooled_candidate": candidate,
            "pooled_delta": {
                "tp": candidate["tp"] - baseline["tp"],
                "fp": candidate["fp"] - baseline["fp"],
                "recall": candidate["recall"] - baseline["recall"],
                "fdr": candidate["fdr"] - baseline["fdr"],
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
