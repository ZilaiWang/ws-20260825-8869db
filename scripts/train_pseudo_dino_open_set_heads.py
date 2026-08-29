#!/usr/bin/env python3
"""Train fixed-epoch, fold-heldout DINOv2 open-set linear heads.

The frozen candidate features are extracted once.  For held-out fold *k*, the
linear head is fitted only from proposal rows belonging to the other two
formal source folds.  No held-out metric is used for checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

BACKGROUND_CLASS_ID = 25
NUM_CLASSES = 26


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_class(row: Mapping[str, str]) -> int:
    if str(row["is_foreground"]).strip() not in {"1", "true", "True"}:
        return BACKGROUND_CLASS_ID
    value = str(row["support_gt_category_id"]).strip()
    if not value:
        raise ValueError("foreground proposal lacks support_gt_category_id")
    category_id = int(value)
    if not 0 <= category_id < BACKGROUND_CLASS_ID:
        raise ValueError(f"invalid foreground category_id={category_id}")
    return category_id


def balanced_batches(
    rows: Sequence[Mapping[str, str]],
    *,
    batch_size: int,
    batches_per_epoch: int,
    seed: int,
) -> list[list[int]]:
    """Draw 50% natural hard background and 50% coarse/fine-balanced foreground."""

    if batch_size < 2 or batches_per_epoch < 1:
        raise ValueError("invalid batch contract")
    background: list[int] = []
    positive_by_fine: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        target = target_class(row)
        if target == BACKGROUND_CLASS_ID:
            background.append(index)
        else:
            positive_by_fine[target].append(index)
    coarse_to_fine = {
        "ship": [value for value in positive_by_fine if value < 4],
        "aircraft": [value for value in positive_by_fine if 4 <= value < 24],
        "vehicle": [value for value in positive_by_fine if value == 24],
    }
    if not background or any(not values for values in coarse_to_fine.values()):
        raise ValueError("training rows lack background or a coarse foreground class")
    rng = random.Random(seed)
    positive_count = batch_size // 2
    batches: list[list[int]] = []
    for _ in range(batches_per_epoch):
        indices: list[int] = []
        for _ in range(positive_count):
            coarse = rng.choice(("ship", "aircraft", "vehicle"))
            fine = rng.choice(coarse_to_fine[coarse])
            indices.append(rng.choice(positive_by_fine[fine]))
        indices.extend(rng.choice(background) for _ in range(batch_size - positive_count))
        rng.shuffle(indices)
        batches.append(indices)
    return batches


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty foreground manifest")
    proposal_indices = [int(row["proposal_index"]) for row in rows]
    if len(set(proposal_indices)) != len(proposal_indices):
        raise ValueError("proposal_index is not unique")
    for row in rows:
        target_class(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches-per-epoch", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")

    import torch

    with np.load(args.features) as archive:
        features = np.asarray(archive["dino_cls_patchmean"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 1536 or not np.isfinite(features).all():
        raise ValueError(f"invalid feature tensor: {features.shape}")
    rows = _load_rows(args.manifest)
    if max(int(row["proposal_index"]) for row in rows) >= len(features):
        raise ValueError("manifest proposal_index exceeds feature rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    fold_summaries: list[dict[str, object]] = []
    for held_out_fold in (0, 1, 2):
        train_rows = [row for row in rows if int(row["fold"]) != held_out_fold]
        train_feature_indices = np.asarray(
            [int(row["proposal_index"]) for row in train_rows], dtype=np.int64
        )
        train_rms = float(
            np.sqrt(np.mean(np.square(features[train_feature_indices]), dtype=np.float64))
        )
        if not math.isfinite(train_rms) or train_rms <= 1e-12:
            raise ValueError("invalid train-only RMS")
        torch.manual_seed(args.seed + held_out_fold)
        torch.cuda.manual_seed_all(args.seed + held_out_fold)
        model = torch.nn.Linear(features.shape[1], NUM_CLASSES).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.02)
        history: list[dict[str, float | int]] = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            correct = 0
            seen = 0
            batches = balanced_batches(
                train_rows,
                batch_size=args.batch_size,
                batches_per_epoch=args.batches_per_epoch,
                seed=args.seed + held_out_fold * 1000 + epoch,
            )
            for batch in batches:
                proposal_index = np.asarray(
                    [int(train_rows[index]["proposal_index"]) for index in batch],
                    dtype=np.int64,
                )
                x = torch.from_numpy(features[proposal_index] / np.float32(train_rms)).to(
                    device, non_blocking=True
                )
                y = torch.tensor(
                    [target_class(train_rows[index]) for index in batch],
                    dtype=torch.long,
                    device=device,
                )
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch)
                correct += int((logits.argmax(1) == y).sum())
                seen += len(batch)
            row = {
                "epoch": epoch,
                "sampled_loss": loss_sum / seen,
                "sampled_accuracy": correct / seen,
            }
            history.append(row)
            print(json.dumps({"fold": held_out_fold, **row}), flush=True)
        checkpoint = args.output_dir / f"dino_open_set_fold{held_out_fold}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "held_out_fold": held_out_fold,
                "feature_dim": features.shape[1],
                "num_classes": NUM_CLASSES,
                "train_rms": train_rms,
            },
            checkpoint,
        )
        fold_summaries.append(
            {
                "held_out_fold": held_out_fold,
                "training_rows": len(train_rows),
                "train_rms": train_rms,
                "history": history,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            }
        )
        del model
    summary = {
        "status": "complete",
        "protocol": "pseudo10k_fold_heldout_dinov2b_open_set_linear_v1",
        "selection": "fixed_epoch_last_no_heldout_selection",
        "feature_sha256": _sha256(args.features),
        "manifest_sha256": _sha256(args.manifest),
        "feature_shape": list(features.shape),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "batches_per_epoch": args.batches_per_epoch,
        "folds": fold_summaries,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "folds": len(fold_summaries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
