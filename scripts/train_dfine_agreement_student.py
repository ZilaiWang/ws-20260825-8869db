#!/usr/bin/env python3
"""Cross-fit a small student to regress offline D-FINE agreement evidence.

Ground truth is never used by the loss.  The target is the square root of the
OOF ``Y5 score * D-FINE support score`` product.  Two fixed observability arms
are allowed: deployable 34-D metadata and the existing 63-D metadata+crop
schema.  The output remains diagnostic until a nested teacher ledger is made.
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
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=("base34", "base_crop63"), default=("base34", "base_crop63")
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cpu")
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

    with np.load(args.data, allow_pickle=False) as payload:
        required = (
            "features",
            "fold",
            "category_id",
            "teacher_agreement_product",
        )
        arrays = {name: np.asarray(payload[name]) for name in required}
    features = arrays["features"].astype(np.float32)
    folds = arrays["fold"].astype(np.int64)
    category_ids = arrays["category_id"].astype(np.int64)
    teacher_product = arrays["teacher_agreement_product"].astype(np.float32)
    n = int(features.shape[0])
    if features.ndim != 2 or features.shape[1] < 63:
        raise ValueError("teacher cache must expose at least the 63-D base_crop schema")
    if any(value.shape[0] != n for value in arrays.values()):
        raise ValueError("teacher cache arrays are not aligned")
    if not np.isfinite(features).all() or not np.isfinite(teacher_product).all():
        raise ValueError("teacher cache contains NaN/Inf")
    if np.any((teacher_product < 0.0) | (teacher_product > 1.0)):
        raise ValueError("teacher agreement product must be in [0, 1]")
    vehicle = category_ids == 24
    if set(folds[vehicle].tolist()) != {0, 1, 2}:
        raise ValueError("vehicle teacher rows must cover folds 0, 1 and 2")
    target = np.sqrt(teacher_product).astype(np.float32)

    class Student(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.network(value).squeeze(1))

    dimensions = {"base34": 34, "base_crop63": 63}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "status": "complete_diagnostic_student",
        "protocol": "outer_cv3_student_of_rowwise_oof_dfine_agreement_v1",
        "warning": (
            "Each teacher row is OOF for itself, but teacher models for the two student-training "
            "folds may have seen the held-out student fold. Formal admission requires nested "
            "teacher inference."
        ),
        "rows": n,
        "vehicle_rows": int(vehicle.sum()),
        "input_sha256": _sha256(args.data),
        "variants": {},
    }
    for variant_index, variant in enumerate(args.variants):
        input_dim = dimensions[variant]
        raw = features[:, :input_dim]
        oof = np.full(n, np.nan, dtype=np.float32)
        fold_payload: list[dict[str, Any]] = []
        for held_out in (0, 1, 2):
            fold_seed = args.seed + variant_index * 10 + held_out
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            train = vehicle & (folds != held_out)
            validation = vehicle & (folds == held_out)
            mean = raw[train].mean(axis=0)
            std = raw[train].std(axis=0)
            std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
            normalized = ((raw - mean) / std).astype(np.float32)
            x_train = torch.from_numpy(normalized[train])
            y_train = torch.from_numpy(target[train])
            loader = DataLoader(
                TensorDataset(x_train, y_train),
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
                heldout_prediction = (
                    model(torch.from_numpy(normalized[validation]).to(device)).cpu().numpy()
                )
            oof[validation] = heldout_prediction.astype(np.float32)
            fold_payload.append(
                {
                    "held_out_fold": held_out,
                    "training_rows": int(train.sum()),
                    "heldout_rows": int(validation.sum()),
                    "final_train_loss": history[-1],
                    "heldout_mae": float(
                        np.abs(heldout_prediction - target[validation]).mean()
                    ),
                    "heldout_pearson": _correlation(
                        heldout_prediction,
                        target[validation],
                    ),
                    "heldout_teacher_positive_mean": float(
                        heldout_prediction[teacher_product[validation] > 0].mean()
                    ),
                    "heldout_teacher_zero_mean": float(
                        heldout_prediction[teacher_product[validation] == 0].mean()
                    ),
                }
            )
        if not np.isfinite(oof[vehicle]).all() or np.isfinite(oof[~vehicle]).any():
            raise RuntimeError(f"invalid OOF coverage for {variant}")
        path = args.output_dir / f"{variant}_oof_scores.npz"
        np.savez_compressed(
            path,
            candidate_index=np.flatnonzero(vehicle).astype(np.int64),
            score=oof[vehicle],
            fold=folds[vehicle],
        )
        summary["variants"][variant] = {
            "input_dim": input_dim,
            "folds": fold_payload,
            "overall_mae": float(np.abs(oof[vehicle] - target[vehicle]).mean()),
            "overall_pearson": _correlation(oof[vehicle], target[vehicle]),
            "score_sha256": _sha256(path),
            "score_path": str(path),
        }
    summary_path = args.output_dir / "student_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
