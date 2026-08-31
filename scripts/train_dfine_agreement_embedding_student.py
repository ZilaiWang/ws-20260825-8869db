#!/usr/bin/env python3
"""Cross-fit visual students for offline D-FINE vehicle agreement evidence.

The frozen ConvNeXt tight/context embeddings are already deployable through the
selective crop-verifier path.  Ground truth is not used by the loss: each arm
regresses D-FINE support independently of the Y5 score.  The exported evidence
is then ``Y5 score * predicted support``, matching the frozen agreement rule.
This remains diagnostic until the teacher ledger is nested for the outer CV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or float(first.std()) < 1e-12 or float(second.std()) < 1e-12:
        return 0.0
    return float(np.corrcoef(first.astype(np.float64), second.astype(np.float64))[0, 1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("tight", "context", "tight_context"),
        default=("tight", "context", "tight_context"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    with np.load(args.teacher_cache, allow_pickle=False) as payload:
        teacher = {
            name: np.asarray(payload[name])
            for name in (
                "candidate_index",
                "fold",
                "category_id",
                "detector_score",
                "teacher_support_score",
                "teacher_agreement_product",
            )
        }
    with np.load(args.embedding_cache, allow_pickle=False) as payload:
        embeddings = {
            name: np.asarray(payload[name])
            for name in (
                "candidate_index",
                "fold",
                "category_id",
                "detector_score",
                "tight_embedding",
                "context_embedding",
            )
        }
    row_count = len(teacher["candidate_index"])
    if any(len(value) != row_count for value in (*teacher.values(), *embeddings.values())):
        raise ValueError("teacher and embedding cache arrays are not aligned")
    for name in ("candidate_index", "fold", "category_id"):
        if not np.array_equal(teacher[name], embeddings[name]):
            raise ValueError(f"teacher and embedding cache disagree on {name}")
    if not np.allclose(
        teacher["detector_score"],
        embeddings["detector_score"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("teacher and embedding detector scores disagree")
    tight = embeddings["tight_embedding"].astype(np.float32)
    context = embeddings["context_embedding"].astype(np.float32)
    if tight.shape != (row_count, 768) or context.shape != (row_count, 768):
        raise ValueError("expected aligned 768-D tight/context embeddings")
    if not np.isfinite(tight).all() or not np.isfinite(context).all():
        raise ValueError("embedding cache contains NaN/Inf")
    folds = teacher["fold"].astype(np.int64)
    vehicle = teacher["category_id"].astype(np.int64) == 24
    detector_score = teacher["detector_score"].astype(np.float32)
    teacher_support = teacher["teacher_support_score"].astype(np.float32)
    teacher_product = teacher["teacher_agreement_product"].astype(np.float32)
    target = teacher_support
    if set(folds[vehicle].tolist()) != {0, 1, 2}:
        raise ValueError("vehicle rows must cover all three folds")

    class Student(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.20),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(64, 1),
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.network(value).squeeze(1))

    variant_features = {
        "tight": tight,
        "context": context,
        "tight_context": np.concatenate((tight, context), axis=1),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "status": "complete_diagnostic_visual_student",
        "protocol": "outer_cv3_visual_student_of_rowwise_oof_dfine_agreement_v1",
        "warning": (
            "Teacher rows are OOF for themselves but are not nested for the student outer CV. "
            "Formal admission requires nested teacher inference."
        ),
        "teacher_cache_sha256": _sha256(args.teacher_cache),
        "embedding_cache_sha256": _sha256(args.embedding_cache),
        "rows": row_count,
        "vehicle_rows": int(vehicle.sum()),
        "variants": {},
    }
    score_column = detector_score[:, None]
    for variant_index, variant in enumerate(args.variants):
        raw = np.concatenate((variant_features[variant], score_column), axis=1)
        input_dim = raw.shape[1]
        oof_support = np.full(row_count, np.nan, dtype=np.float32)
        fold_payload: list[dict[str, Any]] = []
        for held_out in (0, 1, 2):
            fold_seed = args.seed + variant_index * 10 + held_out
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(fold_seed)
            train = vehicle & (folds != held_out)
            validation = vehicle & (folds == held_out)
            mean = raw[train].mean(axis=0)
            std = raw[train].std(axis=0)
            std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
            normalized = ((raw - mean) / std).astype(np.float32)
            loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(normalized[train]),
                    torch.from_numpy(target[train]),
                ),
                batch_size=args.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(fold_seed),
                num_workers=0,
            )
            model = Student(input_dim).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            history: list[float] = []
            for _epoch in range(args.epochs):
                model.train()
                total = count = 0
                for batch_x, batch_y in loader:
                    prediction = model(batch_x.to(device))
                    loss = F.smooth_l1_loss(
                        prediction,
                        batch_y.to(device),
                        beta=0.05,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    total += float(loss.detach()) * len(batch_x)
                    count += len(batch_x)
                history.append(total / max(1, count))
            model.eval()
            with torch.no_grad():
                prediction = (
                    model(torch.from_numpy(normalized[validation]).to(device))
                    .cpu()
                    .numpy()
                )
            oof_support[validation] = prediction.astype(np.float32)
            positive = teacher_support[validation] > 0
            fold_payload.append(
                {
                    "held_out_fold": held_out,
                    "training_rows": int(train.sum()),
                    "heldout_rows": int(validation.sum()),
                    "final_train_loss": history[-1],
                    "heldout_mae": float(np.abs(prediction - target[validation]).mean()),
                    "heldout_pearson": _correlation(prediction, target[validation]),
                    "teacher_positive_score_mean": float(prediction[positive].mean()),
                    "teacher_zero_score_mean": float(prediction[~positive].mean()),
                }
            )
        if not np.isfinite(oof_support[vehicle]).all() or np.isfinite(
            oof_support[~vehicle]
        ).any():
            raise RuntimeError(f"invalid OOF coverage for {variant}")
        oof_evidence = detector_score * oof_support
        path = args.output_dir / f"{variant}_oof_scores.npz"
        np.savez_compressed(
            path,
            candidate_index=np.flatnonzero(vehicle).astype(np.int64),
            score=oof_evidence[vehicle],
            predicted_support=oof_support[vehicle],
            fold=folds[vehicle],
        )
        summary["variants"][variant] = {
            "input_dim": input_dim,
            "folds": fold_payload,
            "overall_support_mae": float(
                np.abs(oof_support[vehicle] - target[vehicle]).mean()
            ),
            "overall_support_pearson": _correlation(
                oof_support[vehicle], target[vehicle]
            ),
            "overall_evidence_pearson": _correlation(
                oof_evidence[vehicle], teacher_product[vehicle]
            ),
            "score_path": str(path),
            "score_sha256": _sha256(path),
        }
    summary_path = args.output_dir / "student_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
