#!/usr/bin/env python3
"""Train one formal outer-fold Proposal-Aligned Verifier checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.hera_guard.dataset import (
    MetadataStandardizer,
    PAVManifestDataset,
    balanced_sampling_weights,
)
from rsdet.hera_guard.losses import PAVLossWeights, pav_multitask_loss
from rsdet.hera_guard.manifest import (
    PAV_METADATA_COLUMNS,
    SUPPORTED_PAV_MANIFEST_VERSIONS,
)
from rsdet.hera_guard.verifier import build_proposal_aligned_verifier


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    versions = {row["manifest_version"] for row in rows}
    if not rows or len(versions) != 1 or not versions.issubset(SUPPORTED_PAV_MANIFEST_VERSIONS):
        raise ValueError("PAV manifest version mismatch")
    return rows


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(labels)) < 2:
        return {"auc": None, "ap": None}
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "ap": float(average_precision_score(labels, probabilities)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--convnext-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-out-fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--freeze", default="freeze_first_stages")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--samples-per-epoch", type=int, default=24000)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=202625)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-weight-sha256", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = _load_rows(args.manifest)
    train_rows = [row for row in rows if int(row["fold"]) != args.held_out_fold]
    validation_rows = [row for row in rows if int(row["fold"]) == args.held_out_fold]
    train_groups = {row["group_id"] for row in train_rows}
    validation_groups = {row["group_id"] for row in validation_rows}
    if not train_rows or not validation_rows or train_groups & validation_groups:
        raise ValueError("formal outer-fold source-group contract failed")
    standardizer = MetadataStandardizer.fit(train_rows)
    train_dataset = PAVManifestDataset(
        train_rows,
        data_root=args.data_root,
        metadata_standardizer=standardizer,
        resolution=args.resolution,
        augment_d4=True,
    )
    validation_dataset = PAVManifestDataset(
        validation_rows,
        data_root=args.data_root,
        metadata_standardizer=standardizer,
        resolution=args.resolution,
        augment_d4=False,
    )
    sample_weights = torch.as_tensor(balanced_sampling_weights(train_rows), dtype=torch.double)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    model = build_proposal_aligned_verifier(
        convnext_weight_path=args.convnext_weights,
        freeze=args.freeze,
        metadata_dim=len(PAV_METADATA_COLUMNS),
        hidden_dim=args.hidden_dim,
        verify_weight_sha256=args.verify_weight_sha256,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in backbone_ids
    ]
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    parameter_groups = [{"params": head_parameters, "lr": args.head_learning_rate}]
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": args.backbone_learning_rate})
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    fine_counts_counter = Counter(
        int(row["target_fine"]) for row in train_rows if int(row["target_foreground"]) == 1
    )
    if set(fine_counts_counter) != set(range(25)):
        raise ValueError("outer training data does not cover all 25 fine classes")
    fine_counts = torch.tensor(
        [fine_counts_counter[index] for index in range(25)],
        dtype=torch.float32,
        device=device,
    )
    loss_weights = PAVLossWeights()
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Counter[str] = Counter()
        examples = 0
        for batch in train_loader:
            tight = batch["tight"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)
            targets = {
                key: batch[key].to(device, non_blocking=True)
                for key in (
                    "foreground",
                    "coarse",
                    "fine",
                    "quality",
                    "protect",
                    "active_fp",
                )
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(tight, context, metadata)
                losses = pav_multitask_loss(
                    output,
                    targets,
                    fine_class_counts=fine_counts,
                    weights=loss_weights,
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            size = tight.shape[0]
            examples += size
            for key in (
                "total",
                "foreground",
                "coarse",
                "fine",
                "quality",
                "protect",
                "active_fp",
            ):
                totals[key] += float(losses[key].detach().item()) * size
        record = {"epoch": epoch, "examples": examples}
        record.update({key: totals[key] / examples for key in totals})
        history.append(record)
        print(json.dumps(record), flush=True)

    model.eval()
    arrays: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "candidate_id",
            "foreground_logit",
            "coarse_logits",
            "fine_logits",
            "quality_logit",
            "protect_logit",
            "active_fp_logit",
            "foreground_target",
            "fine_target",
            "protect_target",
            "active_fp_target",
        )
    }
    with torch.inference_mode():
        for batch in validation_loader:
            tight = batch["tight"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(tight, context, metadata)
            arrays["candidate_id"].append(batch["candidate_id"].numpy())
            for key in (
                "foreground_logit",
                "coarse_logits",
                "fine_logits",
                "quality_logit",
                "protect_logit",
                "active_fp_logit",
            ):
                arrays[key].append(getattr(output, key).float().cpu().numpy())
            arrays["foreground_target"].append(batch["foreground"].numpy())
            arrays["fine_target"].append(batch["fine"].numpy())
            arrays["protect_target"].append(batch["protect"].numpy())
            arrays["active_fp_target"].append(batch["active_fp"].numpy())
    merged = {key: np.concatenate(values) for key, values in arrays.items()}
    if len(merged["candidate_id"]) != len(validation_rows):
        raise RuntimeError("held-out PAV coverage is incomplete")
    foreground_probability = 1 / (1 + np.exp(-merged["foreground_logit"]))
    protect_probability = 1 / (1 + np.exp(-merged["protect_logit"]))
    active_fp_probability = 1 / (1 + np.exp(-merged["active_fp_logit"]))
    fine_prediction = merged["fine_logits"].argmax(axis=1)
    foreground_mask = merged["foreground_target"] > 0.5
    scientific_metrics = {
        "foreground": _metrics(merged["foreground_target"], foreground_probability),
        "protect": _metrics(merged["protect_target"], protect_probability),
        "active_fp": _metrics(merged["active_fp_target"], active_fp_probability),
        "fine_accuracy_on_foreground": float(
            (fine_prediction[foreground_mask] == merged["fine_target"][foreground_mask]).mean()
        ),
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"pav_fold{args.held_out_fold}_final.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "held_out_fold": args.held_out_fold,
            "metadata_standardizer": {
                "columns": PAV_METADATA_COLUMNS,
                "mean": standardizer.mean,
                "std": standardizer.std,
            },
        },
        checkpoint_path,
    )
    logits_path = output_dir / f"pav_fold{args.held_out_fold}_oof_logits.npz"
    np.savez_compressed(logits_path, **merged)
    result = {
        "status": "complete",
        "manifest_version": rows[0]["manifest_version"],
        "held_out_fold": args.held_out_fold,
        "n_train": len(train_rows),
        "n_validation": len(validation_rows),
        "n_train_groups": len(train_groups),
        "n_validation_groups": len(validation_groups),
        "group_overlap": len(train_groups & validation_groups),
        "history": history,
        "scientific_metrics": scientific_metrics,
        "elapsed_seconds": time.time() - started,
        "peak_vram_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "logits_sha256": _sha256(logits_path),
        "manifest_sha256": _sha256(args.manifest),
    }
    (output_dir / f"pav_fold{args.held_out_fold}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
