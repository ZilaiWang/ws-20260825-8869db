#!/usr/bin/env python3
"""Train a fold-heldout 25-class-plus-background proposal verifier.

Unlike the earlier binary foreground gate, this verifier predicts one of the
25 competition fine classes or an explicit background class.  It is trained
on geometrically mined pseudo-10K proposals.  Fold *k* is evaluated only by a
model trained on the other two formal source folds.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.models.crop_classifier import (
    build_convnext_tiny_classifier,
    parameter_summary,
    sha256_file,
)

BACKGROUND_CLASS_ID = len(FINE_NAMES)
NUM_OPEN_SET_CLASSES = BACKGROUND_CLASS_ID + 1


def target_class(row: Mapping[str, Any]) -> int:
    is_foreground = str(row["is_foreground"]).strip() in {"1", "true", "True"}
    if not is_foreground:
        return BACKGROUND_CLASS_ID
    value = str(row["support_gt_category_id"]).strip()
    if not value:
        raise ValueError("foreground proposal lacks support_gt_category_id")
    category_id = int(value)
    if not 0 <= category_id < BACKGROUND_CLASS_ID:
        raise ValueError(f"invalid foreground category_id={category_id}")
    return category_id


def balanced_batch_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    batches_per_epoch: int,
    seed: int,
    background_sampling: str = "natural",
) -> list[list[int]]:
    """Sample 50% background and 50% coarse/fine-balanced foreground."""

    if batch_size < 2 or batches_per_epoch <= 0:
        raise ValueError("batch_size must be >=2 and batches_per_epoch must be positive")
    if background_sampling not in {"natural", "coarse_balanced"}:
        raise ValueError("background_sampling must be natural or coarse_balanced")
    background: list[int] = []
    background_by_coarse: dict[str, list[int]] = defaultdict(list)
    fine: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        target = target_class(row)
        if target == BACKGROUND_CLASS_ID:
            background.append(index)
            coarse = str(row.get("coarse", ""))
            if background_sampling == "coarse_balanced":
                if coarse not in {"ship", "aircraft", "vehicle"}:
                    raise ValueError(f"invalid background coarse={coarse}")
                background_by_coarse[coarse].append(index)
        else:
            fine[target].append(index)
    if not background or not fine:
        raise ValueError("open-set training requires foreground and background proposals")
    by_coarse = {
        "ship": sorted(category for category in fine if category < 4),
        "aircraft": sorted(category for category in fine if 4 <= category < 24),
        "vehicle": sorted(category for category in fine if category == 24),
    }
    if any(not values for values in by_coarse.values()):
        raise ValueError("every coarse class must have foreground support")
    if background_sampling == "coarse_balanced" and any(
        not background_by_coarse[name] for name in ("ship", "aircraft", "vehicle")
    ):
        raise ValueError("coarse-balanced background sampling requires every coarse class")
    rng = random.Random(seed)
    foreground_count = batch_size // 2
    batches: list[list[int]] = []
    for _ in range(batches_per_epoch):
        batch: list[int] = []
        for _ in range(foreground_count):
            coarse = rng.choice(("ship", "aircraft", "vehicle"))
            category = rng.choice(by_coarse[coarse])
            batch.append(rng.choice(fine[category]))
        for _ in range(batch_size - foreground_count):
            if background_sampling == "coarse_balanced":
                coarse = rng.choice(("ship", "aircraft", "vehicle"))
                batch.append(rng.choice(background_by_coarse[coarse]))
            else:
                batch.append(rng.choice(background))
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def _load_rows(path: Path, held_out_fold: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["fold"]) != held_out_fold]
    if not rows:
        raise ValueError("no training rows remain after held-out fold filtering")
    for row in rows:
        target_class(row)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--imagenet-weight", type=Path, required=True)
    parser.add_argument("--p03-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-out-fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument(
        "--regime", choices=("linear_probe", "last_stage", "fine_tune"), default="last_stage"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches-per-epoch", type=int, default=120)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--background-sampling",
        choices=("natural", "coarse_balanced"),
        default="natural",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _initialize_from_p03(model: Any, checkpoint_path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(source, Mapping) or "classifier.2.weight" not in source:
        raise ValueError("P03 initializer is neither a training checkpoint nor a state_dict")
    destination = model.state_dict()
    copied = 0
    for key, value in source.items():
        if key.startswith("classifier.2."):
            continue
        if key in destination and destination[key].shape == value.shape:
            destination[key] = value
            copied += 1
    source_weight = source["classifier.2.weight"]
    source_bias = source["classifier.2.bias"]
    if source_weight.shape[0] < BACKGROUND_CLASS_ID:
        raise ValueError("P03 initializer does not contain all 25 fine classes")
    destination["classifier.2.weight"][:BACKGROUND_CLASS_ID] = source_weight[
        :BACKGROUND_CLASS_ID
    ]
    destination["classifier.2.bias"][:BACKGROUND_CLASS_ID] = source_bias[
        :BACKGROUND_CLASS_ID
    ]
    # A neutral background logit avoids injecting an arbitrary prior.
    destination["classifier.2.weight"][BACKGROUND_CLASS_ID].zero_()
    destination["classifier.2.bias"][BACKGROUND_CLASS_ID].zero_()
    model.load_state_dict(destination, strict=True)
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "copied_non_head_tensors": copied,
    }


def main() -> int:
    args = _parse_args()
    if args.epochs <= 0 or args.resolution <= 0:
        raise ValueError("epochs and resolution must be positive")
    import numpy as np
    import torch
    from PIL import Image

    from rsdet.data.crop_classification import render_crop

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    rows = _load_rows(args.manifest, args.held_out_fold)
    model = build_convnext_tiny_classifier(
        NUM_OPEN_SET_CLASSES,
        weight_path=args.imagenet_weight,
        regime="linear_probe" if args.regime == "linear_probe" else "fine_tune",
    )
    initialization = None
    if args.p03_checkpoint is not None:
        initialization = _initialize_from_p03(model, args.p03_checkpoint)
    if args.regime == "last_stage":
        children = list(model.features.children())
        for child in children[:-1]:
            for parameter in child.parameters():
                parameter.requires_grad = False
    elif args.regime == "linear_probe":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier[-1].parameters():
            parameter.requires_grad = True
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
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
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
            box = tuple(float(row[name]) for name in ("context_x0", "context_y0", "context_x1", "context_y1"))
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
        batches = balanced_batch_indices(
            rows,
            batch_size=args.batch_size,
            batches_per_epoch=args.batches_per_epoch,
            seed=args.seed + epoch,
            background_sampling=args.background_sampling,
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
            labels = torch.tensor([target_class(row) for row in batch_rows], device=device)
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
        row = {"epoch": epoch, "loss": loss_sum / seen, "sampled_accuracy": correct / seen}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "held_out_fold": args.held_out_fold,
        "num_classes": NUM_OPEN_SET_CLASSES,
        "background_class_id": BACKGROUND_CLASS_ID,
        "regime": args.regime,
        "resolution": args.resolution,
    }
    torch.save(payload, args.output)
    summary = {
        "status": "complete",
        "protocol": "pseudo10k_fold_heldout_open_set_verifier_v1",
        "held_out_fold": args.held_out_fold,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "pseudo_root": str(args.pseudo_root.resolve()),
        "training_rows": len(rows),
        "regime": args.regime,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "batches_per_epoch": args.batches_per_epoch,
        "background_sampling": args.background_sampling,
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
