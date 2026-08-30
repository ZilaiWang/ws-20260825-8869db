#!/usr/bin/env python3
"""Train a fold-heldout official-match quality head and export TorchScript.

Expected NPZ arrays:

features [N,D], detector_score [N], best_same_fine_iou [N], coarse_id [N],
protected_tp [N], active_fp [N], active_mask [N], group_id [N], fold [N].

Every feature must be deployable.  The script fits normalization and the model
on the two non-held-out folds only.  It does not select a threshold on the held-
out fold; the exported scores must later be evaluated with the repository's
nested official-frontier pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-out-fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--residual-limit", type=float, default=1.75)
    parser.add_argument("--group-dro-step", type=float, default=0.05)
    parser.add_argument(
        "--sampling",
        choices=("uniform", "group_balanced"),
        default="uniform",
        help="D1/D3 use group_balanced; D0/D2 use uniform.",
    )
    parser.add_argument(
        "--robustness",
        choices=("erm", "group_dro"),
        default="erm",
        help="D2/D3 use group_dro; D0/D1 use erm.",
    )
    parser.add_argument(
        "--rank-enabled",
        action="store_true",
        help="Enable Q2 active-frontier ranking; Q0/Q1 leave it disabled.",
    )
    parser.add_argument("--rank-relax-group-if-empty", action="store_true")
    parser.add_argument("--rank-max-pairs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _required_arrays(payload: Any) -> dict[str, np.ndarray]:
    names = (
        "features",
        "detector_score",
        "best_same_fine_iou",
        "coarse_id",
        "protected_tp",
        "active_fp",
        "active_mask",
        "group_id",
        "fold",
    )
    arrays = {name: np.asarray(payload[name]) for name in names}
    n = int(arrays["features"].shape[0])
    if arrays["features"].ndim != 2:
        raise ValueError("features must have shape [N, D]")
    for name, value in arrays.items():
        if value.shape[0] != n:
            raise ValueError(f"array {name} is not aligned with features")
    if not np.isfinite(arrays["features"]).all():
        raise ValueError("features contain NaN/Inf")
    if not np.isfinite(arrays["detector_score"]).all():
        raise ValueError("detector_score contains NaN/Inf")
    return arrays


def _pair_accuracy(positive: np.ndarray, negative: np.ndarray) -> float:
    """Exact pair accuracy in O(N log N), with ties worth one half."""

    if not len(positive) or not len(negative):
        return float("nan")
    ordered = np.sort(negative.astype(np.float64, copy=False))
    less = np.searchsorted(ordered, positive, side="left")
    less_equal = np.searchsorted(ordered, positive, side="right")
    wins = less + 0.5 * (less_equal - less)
    return float(wins.sum() / (len(positive) * len(negative)))


def main() -> int:
    args = _parse_args()
    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    from rsdet.innovation.group_dro import GroupDROController
    from rsdet.innovation.official_quality import (
        NormalizedQualityRuntime,
        OfficialMatchQualityHead,
        QualityLossWeights,
        official_quality_loss,
        soft_official_match_target,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    with np.load(args.data, allow_pickle=False) as payload:
        arrays = _required_arrays(payload)
    train_mask = arrays["fold"].astype(np.int64) != args.held_out_fold
    validation_mask = ~train_mask
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("both training and held-out rows are required")

    train_features = arrays["features"][train_mask].astype(np.float32)
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    normalized = ((arrays["features"].astype(np.float32) - feature_mean) / feature_std).astype(
        np.float32
    )

    # Group IDs are remapped using training folds only.  Held-out unseen groups
    # receive -1 and are never used for GroupDRO updates.
    raw_train_groups = arrays["group_id"][train_mask].astype(np.int64)
    unique_groups = sorted(set(int(value) for value in raw_train_groups.tolist()))
    group_map = {value: index for index, value in enumerate(unique_groups)}
    remapped_group = np.asarray(
        [group_map.get(int(value), -1) for value in arrays["group_id"]], dtype=np.int64
    )

    tensors = {
        "features": torch.from_numpy(normalized),
        "detector_score": torch.from_numpy(arrays["detector_score"].astype(np.float32)),
        "best_iou": torch.from_numpy(arrays["best_same_fine_iou"].astype(np.float32)),
        "coarse": torch.from_numpy(arrays["coarse_id"].astype(np.int64)),
        "protected": torch.from_numpy(arrays["protected_tp"].astype(np.float32)),
        "active_fp": torch.from_numpy(arrays["active_fp"].astype(np.float32)),
        "active": torch.from_numpy(arrays["active_mask"].astype(np.float32)),
        "group": torch.from_numpy(remapped_group),
    }
    train_indices = np.flatnonzero(train_mask).astype(np.int64)
    validation_indices = np.flatnonzero(validation_mask).astype(np.int64)

    dataset = TensorDataset(torch.from_numpy(train_indices))
    generator = torch.Generator().manual_seed(args.seed)
    if args.sampling == "group_balanced":
        group_counts = np.bincount(
            remapped_group[train_mask], minlength=len(unique_groups)
        ).astype(np.float64)
        sample_weights = 1.0 / np.maximum(group_counts[remapped_group[train_mask]], 1.0)
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights),
            num_samples=len(train_indices),
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            drop_last=False,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=False,
        )

    model = OfficialMatchQualityHead(
        int(normalized.shape[1]),
        hidden_dim=args.hidden_dim,
        residual_limit=args.residual_limit,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    dro = (
        GroupDROController(len(unique_groups), step_size=args.group_dro_step, device=device)
        if args.robustness == "group_dro"
        else None
    )
    weights = QualityLossWeights()
    history: list[dict[str, float]] = []
    validation_outputs: dict[str, np.ndarray] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "rank": 0.0, "batches": 0.0}
        for (batch_index_cpu,) in loader:
            index = batch_index_cpu.long()
            features = tensors["features"][index].to(device)
            detector_score = tensors["detector_score"][index].to(device)
            best_iou = tensors["best_iou"][index].to(device)
            coarse_id = tensors["coarse"][index].to(device)
            protected_tp = tensors["protected"][index].to(device)
            active_fp_batch = tensors["active_fp"][index].to(device)
            active_mask = tensors["active"][index].to(device)
            group_id = tensors["group"][index].to(device)
            if bool((group_id < 0).any()):
                raise RuntimeError("training batch contains an unmapped group")
            soft_match = soft_official_match_target(best_iou, coarse_id)
            output = model(features, detector_score)
            losses = official_quality_loss(
                output,
                detector_score=detector_score,
                soft_match_target=soft_match,
                protected_tp=protected_tp,
                active_fp=active_fp_batch,
                active_mask=active_mask,
                coarse_ids=coarse_id,
                group_ids=group_id,
                weights=weights,
                rank_enabled=args.rank_enabled,
                rank_relax_group_if_empty=args.rank_relax_group_if_empty,
                rank_max_pairs=args.rank_max_pairs,
            )
            if dro is None:
                total = losses["total"]
            else:
                robust = dro(losses["per_sample"], group_id, update=True)
                total = robust.loss + weights.rank * losses["rank"]
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running["loss"] += float(total.detach())
            running["rank"] += float(losses["rank"].detach())
            running["batches"] += 1.0

        model.eval()
        with torch.no_grad():
            index = torch.from_numpy(validation_indices)
            features = tensors["features"][index].to(device)
            detector_score = tensors["detector_score"][index].to(device)
            output = model(features, detector_score)
            final_score = torch.sigmoid(output.final_logit).cpu().numpy()
            validation_outputs = {
                "candidate_index": validation_indices.astype(np.int64),
                "score": final_score.astype(np.float32),
                "match_quality": torch.sigmoid(output.match_logit).cpu().numpy().astype(np.float32),
                "protect_quality": torch.sigmoid(output.protect_logit).cpu().numpy().astype(np.float32),
                "active_fp_probability": torch.sigmoid(output.active_fp_logit).cpu().numpy().astype(np.float32),
                "residual": output.residual.cpu().numpy().astype(np.float32),
            }
            validation_protected = arrays["protected_tp"][validation_mask].astype(bool)
            validation_active_fp = arrays["active_fp"][validation_mask].astype(bool)
            protected_mean = (
                float(final_score[validation_protected].mean())
                if validation_protected.any()
                else float("nan")
            )
            active_fp_mean = (
                float(final_score[validation_active_fp].mean())
                if validation_active_fp.any()
                else float("nan")
            )
            pair_accuracy = _pair_accuracy(
                final_score[validation_protected], final_score[validation_active_fp]
            )
        row = {
            "epoch": float(epoch),
            "train_loss": running["loss"] / max(1.0, running["batches"]),
            "train_rank_loss": running["rank"] / max(1.0, running["batches"]),
            "heldout_protected_mean_score": protected_mean,
            "heldout_active_fp_mean_score": active_fp_mean,
            "heldout_pair_accuracy": pair_accuracy,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"quality_fold{args.held_out_fold}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(normalized.shape[1]),
            "hidden_dim": args.hidden_dim,
            "residual_limit": args.residual_limit,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "group_map": group_map,
            "held_out_fold": args.held_out_fold,
            "sampling": args.sampling,
            "robustness": args.robustness,
            "rank_enabled": args.rank_enabled,
            "group_dro_state": dro.state_dict() if dro is not None else None,
        },
        checkpoint,
    )
    runtime = NormalizedQualityRuntime(
        model.cpu(), torch.from_numpy(feature_mean), torch.from_numpy(feature_std)
    ).eval()
    example_features = torch.zeros(2, normalized.shape[1], dtype=torch.float32)
    example_score = torch.full((2,), 0.15, dtype=torch.float32)
    scripted = torch.jit.trace(runtime, (example_features, example_score), strict=True)
    runtime_path = args.output_dir / f"quality_fold{args.held_out_fold}.torchscript.pt"
    scripted.save(str(runtime_path))
    if validation_outputs is None:
        raise RuntimeError("held-out score export is missing")
    score_path = args.output_dir / f"quality_fold{args.held_out_fold}.scores.npz"
    np.savez_compressed(score_path, **validation_outputs)

    summary = {
        "status": "complete",
        "held_out_fold": args.held_out_fold,
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "input_dim": int(normalized.shape[1]),
        "groups": len(unique_groups),
        "sampling": args.sampling,
        "robustness": args.robustness,
        "rank_enabled": args.rank_enabled,
        "rank_relax_group_if_empty": args.rank_relax_group_if_empty,
        "checkpoint": str(checkpoint),
        "torchscript": str(runtime_path),
        "held_out_scores": str(score_path),
        "history": history,
        "warning": (
            "This script exports scores only. Select thresholds with nested train-fold "
            "calibration and evaluate the held-out fold using the official matcher."
        ),
    }
    (args.output_dir / f"quality_fold{args.held_out_fold}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
