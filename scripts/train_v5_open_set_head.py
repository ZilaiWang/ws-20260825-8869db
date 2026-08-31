#!/usr/bin/env python3
"""Train fold-heldout V5 three-way open-set heads on frozen crop embeddings."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--max-sample-weight-ratio", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from torch.nn import functional as functional
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    from rsdet.innovation.proposal_open_set import OPEN_LABEL_NAMES, ProposalOpenSetHead

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with np.load(args.features, allow_pickle=False) as payload:
        tight = payload["tight_embedding"].astype(np.float32)
        context = payload["context_embedding"].astype(np.float32)
        labels = payload["open_set_label"].astype(np.int64)
        coarse = payload["coarse_id"].astype(np.int64)
        folds = payload["fold"].astype(np.int64)
        candidate_index = payload["candidate_index"].astype(np.int64)
        image_id = payload["image_id"].astype(np.int64)
        category_id = payload["category_id"].astype(np.int64)
        bbox_xyxy = payload["bbox_xyxy"].astype(np.float32)
    n, embedding_dim = tight.shape
    if context.shape != tight.shape or set(folds.tolist()) != {0, 1, 2}:
        raise ValueError("feature cache violates shape/fold contract")
    all_scores = np.full((n, len(OPEN_LABEL_NAMES)), np.nan, dtype=np.float32)
    summaries: dict[str, object] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for held_out in range(3):
        train = np.flatnonzero((folds != held_out) & (labels >= 0))
        validation = np.flatnonzero(folds == held_out)
        group = coarse[train] * len(OPEN_LABEL_NAMES) + labels[train]
        counts = np.bincount(group, minlength=3 * len(OPEN_LABEL_NAMES)).astype(np.float64)
        maximum = counts.max()
        weights = np.minimum(
            maximum / np.maximum(counts[group], 1.0),
            args.max_sample_weight_ratio,
        )
        dataset = TensorDataset(torch.from_numpy(train))
        generator = torch.Generator().manual_seed(args.seed + held_out)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights),
            num_samples=len(train),
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
        model = ProposalOpenSetHead(
            embedding_dim,
            hidden_dim=args.hidden_dim,
        ).to(args.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            seen = 0
            for (index_cpu,) in loader:
                index = index_cpu.numpy()
                tight_batch = torch.from_numpy(tight[index]).to(args.device)
                context_batch = torch.from_numpy(context[index]).to(args.device)
                coarse_batch = functional.one_hot(
                    torch.from_numpy(coarse[index]).to(args.device), num_classes=3
                ).float()
                target = torch.from_numpy(labels[index]).to(args.device)
                logits = model(tight_batch, context_batch, coarse_batch)
                loss = functional.cross_entropy(logits, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(index)
                seen += len(index)
            history.append({"epoch": epoch, "train_loss": total / max(seen, 1)})

        model.eval()
        with torch.inference_mode():
            for start in range(0, len(validation), args.batch_size):
                index = validation[start : start + args.batch_size]
                logits = model(
                    torch.from_numpy(tight[index]).to(args.device),
                    torch.from_numpy(context[index]).to(args.device),
                    functional.one_hot(
                        torch.from_numpy(coarse[index]).to(args.device), num_classes=3
                    ).float(),
                )
                all_scores[index] = torch.softmax(logits, dim=1).cpu().numpy()
        labeled_validation = validation[labels[validation] >= 0]
        predicted = all_scores[labeled_validation].argmax(axis=1)
        confusion = np.zeros((3, 3), dtype=np.int64)
        for target, prediction in zip(labels[labeled_validation], predicted, strict=True):
            confusion[int(target), int(prediction)] += 1
        recall = np.divide(
            confusion.diagonal(),
            confusion.sum(axis=1),
            out=np.zeros(3, dtype=np.float64),
            where=confusion.sum(axis=1) > 0,
        )
        checkpoint = args.output_dir / f"open_set_fold{held_out}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "embedding_dim": embedding_dim,
                "hidden_dim": args.hidden_dim,
                "labels": OPEN_LABEL_NAMES,
            },
            checkpoint,
        )
        summaries[str(held_out)] = {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "labeled_validation_rows": len(labeled_validation),
            "confusion_matrix": confusion.tolist(),
            "per_label_recall": {
                name: float(recall[index]) for index, name in enumerate(OPEN_LABEL_NAMES)
            },
            "balanced_accuracy": float(recall.mean()),
            "history": history,
            "max_sample_weight_ratio": args.max_sample_weight_ratio,
        }
    if not np.isfinite(all_scores).all():
        raise RuntimeError("OOF open-set score coverage is incomplete")
    score_path = args.output_dir / "open_set_oof_scores.npz"
    np.savez_compressed(
        score_path,
        candidate_index=candidate_index,
        probabilities=all_scores,
        fold=folds,
        image_id=image_id,
        category_id=category_id,
        bbox_xyxy=bbox_xyxy,
    )
    summary = {
        "status": "complete",
        "protocol": "v5_three_way_open_set_crossfit_head_v1",
        "rows": n,
        "folds": summaries,
        "score_path": str(score_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
