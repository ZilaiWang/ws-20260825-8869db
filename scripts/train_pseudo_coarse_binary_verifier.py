#!/usr/bin/env python3
"""Train fold-heldout, coarse-specific foreground/background crop verifiers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.models.crop_classifier import (
    build_convnext_tiny_classifier,
    parameter_summary,
    sha256_file,
)

COARSE_CLASSES = ("ship", "aircraft", "vehicle")


def binary_target(row: Mapping[str, Any]) -> int:
    return int(str(row["is_foreground"]).strip() in {"1", "true", "True"})


def balanced_binary_batches(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    batches_per_epoch: int,
    seed: int,
    negative_sampling: str = "uniform",
) -> list[list[int]]:
    """Draw equal background/foreground batches with fine-balanced positives.

    ``score_sqrt`` is the one pre-registered hard-negative variant.  It samples
    background proposals in proportion to ``sqrt(detector_score)``.  This
    concentrates capacity on false positives that can actually survive an
    operating threshold while retaining non-zero support for low-score
    background.  Foreground sampling remains score-agnostic so that weak true
    candidates are not silently removed from training.
    """

    if batch_size < 2 or batches_per_epoch <= 0:
        raise ValueError("batch_size must be >=2 and batches_per_epoch must be positive")
    if negative_sampling not in {"uniform", "score_sqrt"}:
        raise ValueError("negative_sampling must be uniform or score_sqrt")
    negative: list[int] = []
    positive_by_fine: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if binary_target(row):
            value = str(row["support_gt_category_id"]).strip()
            if not value:
                raise ValueError("foreground row lacks support_gt_category_id")
            positive_by_fine[int(value)].append(index)
        else:
            negative.append(index)
    if not negative or not positive_by_fine:
        raise ValueError("binary training requires positive and negative rows")
    negative_weights: list[float] | None = None
    if negative_sampling == "score_sqrt":
        negative_weights = []
        for index in negative:
            score = float(rows[index]["score"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"invalid detector score={score}")
            negative_weights.append(math.sqrt(max(score, 1e-4)))
    fine_classes = sorted(positive_by_fine)
    rng = random.Random(seed)
    positive_count = batch_size // 2
    batches: list[list[int]] = []
    for _ in range(batches_per_epoch):
        batch = [
            rng.choice(positive_by_fine[rng.choice(fine_classes)])
            for _ in range(positive_count)
        ]
        if negative_weights is None:
            batch.extend(rng.choice(negative) for _ in range(batch_size - positive_count))
        else:
            batch.extend(
                rng.choices(
                    negative,
                    weights=negative_weights,
                    k=batch_size - positive_count,
                )
            )
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def _load_rows(path: Path, *, held_out_fold: int, coarse: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["fold"]) != held_out_fold and str(row["coarse"]) == coarse
        ]
    if not rows:
        raise ValueError("no coarse rows remain after held-out filtering")
    return rows


def _initialize_backbone(model: Any, checkpoint_path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(source, Mapping):
        raise ValueError("P03 initializer is not a state_dict")
    destination = model.state_dict()
    copied = 0
    for key, value in source.items():
        if key.startswith("classifier.2."):
            continue
        if key in destination and destination[key].shape == value.shape:
            destination[key] = value
            copied += 1
    model.load_state_dict(destination, strict=True)
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "copied_non_head_tensors": copied,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--imagenet-weight", type=Path, required=True)
    parser.add_argument("--p03-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-out-fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--coarse", choices=COARSE_CLASSES, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches-per-epoch", type=int, default=40)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--negative-sampling",
        choices=("uniform", "score_sqrt"),
        default="uniform",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import numpy as np
    import torch
    from PIL import Image

    from rsdet.data.crop_classification import render_crop

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows = _load_rows(args.manifest, held_out_fold=args.held_out_fold, coarse=args.coarse)
    device = torch.device(args.device)
    model = build_convnext_tiny_classifier(2, weight_path=args.imagenet_weight, regime="fine_tune")
    initialization = _initialize_backbone(model, args.p03_checkpoint)
    for child in list(model.features.children())[:-1]:
        for parameter in child.parameters():
            parameter.requires_grad = False
    model.to(device)
    head_ids = {id(parameter) for parameter in model.classifier.parameters()}
    groups = [
        {
            "params": [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad and id(parameter) not in head_ids
            ],
            "lr": args.backbone_lr,
        },
        {
            "params": [parameter for parameter in model.classifier.parameters() if parameter.requires_grad],
            "lr": args.head_lr,
        },
    ]
    optimizer = torch.optim.AdamW(
        [group for group in groups if group["params"]], weight_decay=args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    image_cache: dict[Path, Image.Image] = {}
    crop_cache: dict[str, torch.Tensor] = {}

    def tensor_for(row: Mapping[str, Any]) -> torch.Tensor:
        uid = str(row["proposal_uid"])
        cached = crop_cache.get(uid)
        if cached is None:
            path = (args.pseudo_root / str(row["source_relative_path"])).resolve()
            image = image_cache.get(path)
            if image is None:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                image_cache[path] = image
            box = tuple(
                float(row[name])
                for name in ("context_x0", "context_y0", "context_x1", "context_y1")
            )
            crop = render_crop(image, box, args.resolution)
            cached = torch.from_numpy(np.asarray(crop, dtype=np.uint8).copy()).permute(2, 0, 1)
            crop_cache[uid] = cached
        tensor = cached
        rotation = random.randrange(4)
        if rotation:
            tensor = torch.rot90(tensor, rotation, dims=(1, 2))
        if random.random() < 0.5:
            tensor = torch.flip(tensor, dims=(2,))
        if random.random() < 0.5:
            tensor = torch.flip(tensor, dims=(1,))
        return tensor

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        batches = balanced_binary_batches(
            rows,
            batch_size=args.batch_size,
            batches_per_epoch=args.batches_per_epoch,
            seed=args.seed + epoch,
            negative_sampling=args.negative_sampling,
        )
        loss_sum = 0.0
        correct = 0
        seen = 0
        for indices in batches:
            batch_rows = [rows[index] for index in indices]
            x = torch.stack([tensor_for(row) for row in batch_rows]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            x = (x / 255.0 - mean) / std
            labels = torch.tensor([binary_target(row) for row in batch_rows], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(indices)
        epoch_row = {
            "epoch": epoch,
            "loss": loss_sum / seen,
            "sampled_accuracy": correct / seen,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "held_out_fold": args.held_out_fold,
            "coarse": args.coarse,
            "num_classes": 2,
            "resolution": args.resolution,
        },
        args.output,
    )
    summary = {
        "status": "complete",
        "protocol": "pseudo10k_fold_heldout_coarse_binary_verifier_v1",
        "held_out_fold": args.held_out_fold,
        "coarse": args.coarse,
        "manifest_sha256": sha256_file(args.manifest),
        "training_rows": len(rows),
        "positive_rows": sum(binary_target(row) for row in rows),
        "negative_rows": sum(not binary_target(row) for row in rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "batches_per_epoch": args.batches_per_epoch,
        "negative_sampling": args.negative_sampling,
        "resolution": args.resolution,
        "parameters": parameter_summary(model),
        "initialization": initialization,
        "history": history,
        "checkpoint_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
