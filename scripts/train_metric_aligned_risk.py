#!/usr/bin/env python3
"""Train HERA-Guard V3 bounded residual risk heads with outer-fold OOF.

The four cumulative stages are frozen by contract:

1. official one-winner BCE;
2. BCE + same-object RankNet;
3. stage 2 + differentiable FDR constraint and recall reward;
4. stage 3 + explicit canonical-vs-duplicate one-winner ranking.

No held-out fold is used for fitting, early stopping, threshold choice, or
normalization.  The resulting stage predictions are intended for the exact
official cross-fit frontier script, not for in-script model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.hera_guard.metric_aligned import build_metric_aligned_roles
from rsdet.hera_guard.metric_risk import (
    FEATURE_COLUMNS,
    AnchoredResidualRisk,
    align_anchor_scores,
    broad_rank_roles,
    build_metric_features,
    deterministic_rank_pairs,
    metric_risk_loss,
    one_winner_roles,
)
from rsdet.utils.config import load_config

STAGES = ("bce", "rank", "soft_fdr", "one_winner")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_threshold(logits: np.ndarray, labels: np.ndarray, target_fdr: float) -> float:
    order = np.argsort(-logits, kind="stable")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    count = np.arange(1, len(order) + 1)
    fdr = (count - tp) / count
    feasible = np.flatnonzero(fdr <= target_fdr)
    if not feasible.size:
        return float(logits[order[0]])
    best_tp = tp[feasible].max()
    candidates = feasible[tp[feasible] == best_tp]
    selected = int(candidates[np.argmin(fdr[candidates])])
    return float(logits[order[selected]])


def _to_coco(
    records: list[dict[str, Any]], probabilities: np.ndarray
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item, probability in zip(records, probabilities, strict=True):
        output.append(
            {
                "image_id": int(item["image_id"]),
                "category_id": int(item["category_id"]),
                "bbox": [float(value) for value in item["bbox"]],
                "score": float(probability),
                "source_fold": int(item["source_fold"]),
                "source_model": str(item.get("source_model", "")),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--evidence-pred", type=Path, required=True)
    parser.add_argument("--anchor-pred", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--epochs-per-stage", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--residual-limit", type=float, default=2.5)
    parser.add_argument("--max-rank-pairs", type=int, default=100_000)
    parser.add_argument("--target-fdr", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.epochs_per_stage <= 0 or args.max_rank_pairs <= 0:
        raise ValueError("training epochs and max rank pairs must be positive")

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    evidence: list[dict[str, Any]] = json.loads(
        args.evidence_pred.read_text(encoding="utf-8")
    )
    anchor: list[dict[str, Any]] = json.loads(args.anchor_pred.read_text(encoding="utf-8"))
    anchor_scores = align_anchor_scores(evidence, anchor)
    folds = np.asarray([int(item["source_fold"]) for item in evidence], dtype=np.int64)
    if set(folds.tolist()) != {0, 1, 2}:
        raise ValueError("evidence ledger must contain source folds 0, 1 and 2")
    gt_raw = json.loads(args.gt.read_text(encoding="utf-8"))
    image_folds = {int(item["id"]): int(item["fold"]) for item in gt_raw["images"]}
    if any(image_folds[int(item["image_id"])] != int(item["source_fold"]) for item in evidence):
        raise ValueError("proposal source_fold does not match pseudo image fold")

    role_predictions = [
        {**row, "score": float(score)}
        for row, score in zip(evidence, anchor_scores, strict=True)
    ]
    role_result = build_metric_aligned_roles(
        gt_boxes=load_coco_ground_truth(args.gt),
        predictions=role_predictions,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    labels = np.asarray([row.target for row in role_result.roles], dtype=np.float32)
    features = np.asarray(
        build_metric_features(
            evidence,
            anchor_scores=anchor_scores,
            category_mapping=protocol.category_mapping,
        ),
        dtype=np.float32,
    )
    anchor_logits = features[:, 0].astype(np.float64)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_logits = {
        stage: np.full(len(evidence), np.nan, dtype=np.float64) for stage in STAGES
    }
    fold_records: list[dict[str, Any]] = []

    for held_out in (0, 1, 2):
        train_indices = np.flatnonzero(folds != held_out)
        val_indices = np.flatnonzero(folds == held_out)
        train_features_np = features[train_indices]
        feature_mean = train_features_np.mean(axis=0)
        feature_std = train_features_np.std(axis=0)
        feature_std[feature_std < 1e-5] = 1.0
        model = AnchoredResidualRisk(
            feature_mean=feature_mean,
            feature_std=feature_std,
            hidden_dim=args.hidden_dim,
            residual_limit=args.residual_limit,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        train_x = torch.from_numpy(train_features_np).to(device)
        train_y = torch.from_numpy(labels[train_indices]).to(device)
        val_x = torch.from_numpy(features[val_indices]).to(device)
        train_roles = [role_result.roles[index] for index in train_indices]
        rank_pairs = torch.as_tensor(
            deterministic_rank_pairs(
                train_roles,
                include_roles=broad_rank_roles(),
                max_pairs=args.max_rank_pairs,
            ),
            dtype=torch.long,
            device=device,
        ).reshape(-1, 2)
        winner_pairs = torch.as_tensor(
            deterministic_rank_pairs(
                train_roles,
                include_roles=one_winner_roles(),
                max_pairs=args.max_rank_pairs,
            ),
            dtype=torch.long,
            device=device,
        ).reshape(-1, 2)
        threshold = _training_threshold(
            anchor_logits[train_indices], labels[train_indices], args.target_fdr
        )
        positive_weight = min(
            12.0,
            max(1.0, float((train_y.numel() - train_y.sum()).item() / train_y.sum().item())),
        )
        for stage_index, stage in enumerate(STAGES):
            history = []
            model.train()
            for epoch in range(args.epochs_per_stage):
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_x)
                losses = metric_risk_loss(
                    logits,
                    train_y,
                    stage=stage,
                    rank_pairs=rank_pairs,
                    one_winner_pairs=winner_pairs,
                    soft_threshold=threshold,
                    target_fdr=args.target_fdr,
                    positive_weight=positive_weight,
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                if epoch in {0, args.epochs_per_stage - 1}:
                    history.append(
                        {
                            "epoch": epoch + 1,
                            **{
                                key: float(value.detach().cpu())
                                for key, value in losses.items()
                            },
                        }
                    )
            model.eval()
            with torch.inference_mode():
                heldout_logits = model(val_x).cpu().numpy().astype(np.float64)
            oof_logits[stage][val_indices] = heldout_logits
            checkpoint = args.output_dir / f"{stage}_fold{held_out}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_columns": list(FEATURE_COLUMNS),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "held_out_fold": held_out,
                    "stage": stage,
                    "residual_limit": args.residual_limit,
                },
                checkpoint,
            )
            model.cpu().eval()
            scripted = torch.jit.trace(
                model,
                torch.zeros((1, len(FEATURE_COLUMNS)), dtype=torch.float32),
            )
            torch.jit.save(scripted, args.output_dir / f"{stage}_fold{held_out}.torchscript.pt")
            model.to(device)
            fold_records.append(
                {
                    "held_out_fold": held_out,
                    "stage": stage,
                    "stage_index": stage_index,
                    "n_train": len(train_indices),
                    "n_validation": len(val_indices),
                    "train_positive": int(labels[train_indices].sum()),
                    "validation_positive": int(labels[val_indices].sum()),
                    "rank_pairs": int(rank_pairs.shape[0]),
                    "one_winner_pairs": int(winner_pairs.shape[0]),
                    "soft_threshold_logit": threshold,
                    "positive_weight": positive_weight,
                    "history": history,
                    "checkpoint_sha256": _sha256(checkpoint),
                }
            )

    stage_outputs: dict[str, Any] = {}
    for stage in STAGES:
        if not np.isfinite(oof_logits[stage]).all():
            raise RuntimeError(f"incomplete OOF logits for stage {stage}")
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(oof_logits[stage], -40.0, 40.0)))
        path = args.output_dir / f"{stage}_oof_predictions.json"
        path.write_text(
            json.dumps(_to_coco(evidence, probabilities), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        stage_outputs[stage] = {
            "predictions": path.name,
            "predictions_sha256": _sha256(path),
            "score_min": float(probabilities.min()),
            "score_max": float(probabilities.max()),
            "score_mean": float(probabilities.mean()),
        }

    summary = {
        "status": "metric_aligned_risk_oof_complete",
        "protocol": "hera_guard_v3_metric_aligned_residual_outer_cv3_v1",
        "warning": "pseudo-10K deployment-domain proxy; not hidden-test evidence",
        "stages": list(STAGES),
        "feature_columns": list(FEATURE_COLUMNS),
        "input_candidates": len(evidence),
        "canonical_positive": int(labels.sum()),
        "role_counts": role_result.role_counts,
        "cross_coarse_foreground_pollution": role_result.cross_coarse_foreground_pollution,
        "residual_limit": args.residual_limit,
        "epochs_per_stage": args.epochs_per_stage,
        "target_fdr": args.target_fdr,
        "fold_records": fold_records,
        "stage_outputs": stage_outputs,
        "input_sha256": {
            "gt": _sha256(args.gt),
            "evidence_pred": _sha256(args.evidence_pred),
            "anchor_pred": _sha256(args.anchor_pred),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
