#!/usr/bin/env python3
"""R1-1 aircraft-only proposal-domain refinement and D4 ablation."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

from rsdet.analysis.aircraft_refinement import (
    AIRCRAFT_CLASS_IDS,
    R11_CONTRACT_VERSION,
    audit_aircraft_training_rows,
    build_aircraft_variants,
    condition_probabilities,
    load_aircraft_training_rows,
)
from rsdet.analysis.crossfit_thresholds import (
    load_cv3_aggregate,
    load_gt_from_formal_crop_manifest,
)
from rsdet.analysis.proposal_reranking import (
    apply_frozen_c2,
    load_proposal_manifest,
    paired_transition,
    run_reranking_crossfit,
    sha256_file,
)
from rsdet.data.crop_classification import render_crop
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.features.p04_inputs import D4_VIEW_IDS, apply_d4_view
from rsdet.models.crop_classifier import build_convnext_tiny_classifier
from rsdet.utils.config import load_config


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract_version") != R11_CONTRACT_VERSION:
        raise ValueError("R1-1 config contract_version 非法")
    return payload


def _seed(torch: Any, value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _require_stack(torch: Any, torchvision: Any, config: Mapping[str, Any]) -> None:
    expected_torch = str(config["runtime"]["torch_version"])
    expected_torchvision = str(config["runtime"]["torchvision_version"])
    if torch.__version__.split("+")[0] != expected_torch:
        raise RuntimeError(f"torch 必须为 {expected_torch}，当前 {torch.__version__}")
    if torchvision.__version__.split("+")[0] != expected_torchvision:
        raise RuntimeError(
            f"torchvision 必须为 {expected_torchvision}，当前 {torchvision.__version__}"
        )


def _parse_xyxy(text: str) -> tuple[float, float, float, float]:
    values = tuple(float(item) for item in text.split(","))
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"crop_xyxy 非法: {text!r}")
    return values


def _normalize(image: Image.Image) -> Any:
    import torch
    from torchvision.transforms import functional

    tensor = functional.to_tensor(image)
    tensor = functional.normalize(
        tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return tensor.to(dtype=torch.float32)


class _ImageCache:
    def __init__(self, data_root: Path, capacity: int = 8) -> None:
        self.data_root = data_root.resolve()
        self.capacity = capacity
        self.values: OrderedDict[Path, Image.Image] = OrderedDict()

    def get(self, relative_path: str) -> Image.Image:
        source_path = (self.data_root / relative_path).resolve()
        try:
            source_path.relative_to(self.data_root)
        except ValueError as error:
            raise ValueError(f"source path 逃逸 data root: {source_path}") from error
        image = self.values.pop(source_path, None)
        if image is None:
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            with Image.open(source_path) as source:
                image = source.convert("RGB")
        self.values[source_path] = image
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return image


class _TrainDataset:
    def __init__(self, rows: Sequence[Mapping[str, str]], data_root: Path) -> None:
        self.rows = tuple(rows)
        self.cache = _ImageCache(data_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        row = self.rows[index]
        source = self.cache.get(row["source_relative_path"])
        crop = render_crop(source, _parse_xyxy(row["crop_xyxy"]), 224)
        crop = apply_d4_view(crop, random.choice(D4_VIEW_IDS))
        return _normalize(crop), int(row["class_id"]), row["proposal_uid"]


class _InferenceDataset:
    def __init__(self, records: Sequence[Any], data_root: Path) -> None:
        self.records = tuple(records)
        self.cache = _ImageCache(data_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, str]:
        import torch

        record = self.records[index]
        source = self.cache.get(record.relative_path)
        crop = render_crop(source, record.bbox_xyxy, 224)
        views = [_normalize(apply_d4_view(crop, view)) for view in D4_VIEW_IDS]
        return torch.stack(views, dim=0), record.proposal_uid


def _convnext_features_and_logits(model: Any, images: Any) -> tuple[Any, Any]:
    """Expose ConvNeXt penultimate features without changing its state dict.

    Torchvision ConvNeXt keeps normalization and flattening in ``classifier``.
    Calling those layers explicitly preserves bitwise-equivalent logits while
    making the 768-D feature available to a training-only auxiliary loss.
    """

    if len(model.classifier) != 3:
        raise ValueError("unexpected torchvision ConvNeXt classifier layout")
    feature_map = model.features(images)
    pooled = model.avgpool(feature_map)
    features = model.classifier[0](pooled)
    features = model.classifier[1](features)
    logits = model.classifier[2](features)
    return features, logits


class _ClassCenterMemory:
    """EMA class prototypes for an ECC-inspired angular margin loss.

    Centers are initialized from the already trained classifier weights and
    updated from detached training features.  They are never serialized into
    the deployed model and therefore add no inference parameter or latency.
    """

    def __init__(
        self,
        classifier_weight: Any,
        *,
        momentum: float,
        margin: float,
        negative_weight: float,
    ) -> None:
        import torch.nn.functional as functional

        if not 0.0 <= momentum < 1.0:
            raise ValueError("class center momentum 必须在 [0, 1) 内")
        if margin < 0.0 or negative_weight < 0.0:
            raise ValueError("class center margin/negative_weight 必须非负")
        self.centers = functional.normalize(classifier_weight.detach().float(), dim=1)
        self.momentum = momentum
        self.margin = margin
        self.negative_weight = negative_weight

    def loss(self, features: Any, labels: Any) -> tuple[Any, dict[str, float]]:
        import torch
        import torch.nn.functional as functional

        normalized = functional.normalize(features.float(), dim=1)
        centers = self.centers.to(device=normalized.device)
        similarities = normalized @ centers.transpose(0, 1)
        indices = torch.arange(len(labels), device=labels.device)
        positive = similarities[indices, labels]
        negatives = similarities.masked_fill(
            functional.one_hot(labels, num_classes=centers.shape[0]).bool(),
            float("-inf"),
        )
        hardest_negative = negatives.max(dim=1).values
        pull = (1.0 - positive).mean()
        push = functional.relu(self.margin + hardest_negative - positive).mean()
        total = pull + self.negative_weight * push
        diagnostics = {
            "center_pull": float(pull.detach()),
            "center_push": float(push.detach()),
            "positive_cosine": float(positive.detach().mean()),
            "hardest_negative_cosine": float(hardest_negative.detach().mean()),
        }
        return total.to(dtype=features.dtype), diagnostics

    def update(self, features: Any, labels: Any) -> None:
        import torch.nn.functional as functional

        normalized = functional.normalize(features.detach().float(), dim=1)
        centers = self.centers.to(device=normalized.device)
        for class_id in labels.unique(sorted=True).tolist():
            mask = labels.eq(int(class_id))
            batch_center = functional.normalize(
                normalized[mask].mean(dim=0, keepdim=True), dim=1
            ).squeeze(0)
            centers[int(class_id)] = functional.normalize(
                self.momentum * centers[int(class_id)]
                + (1.0 - self.momentum) * batch_center,
                dim=0,
            )
        self.centers = centers.detach()


def _load_p03_model(
    checkpoint_path: Path,
    weights_path: Path,
    *,
    fold: int,
    expected_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    import torch

    actual = sha256_file(checkpoint_path)
    if actual != expected_sha256:
        raise ValueError(f"fold{fold} P03 checkpoint SHA 不匹配: {actual}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resolved = payload.get("resolved_config", {})
    expected = {
        "fold": fold,
        "policy": "tight",
        "resolution": 224,
        "regime": "fine_tune",
        "sampler": "natural",
        "seed": 42,
        "checkpoint_selection": "fixed_epoch_last",
    }
    mismatches = {
        key: {"actual": resolved.get(key), "expected": value}
        for key, value in expected.items()
        if resolved.get(key) != value
    }
    if mismatches or int(payload.get("epoch", -1)) != 30:
        raise ValueError(f"P03 embedded contract 不匹配: {mismatches}")
    model = build_convnext_tiny_classifier(25, weight_path=weights_path, regime="fine_tune")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, {"checkpoint_sha256": actual, "embedded_config": resolved}


def _load_adapted_model(
    checkpoint_path: Path,
    weights_path: Path,
    *,
    fold: int,
    method: str,
) -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resolved = payload.get("resolved_config", {})
    expected = {
        "contract_version": R11_CONTRACT_VERSION,
        "held_out_fold": fold,
        "method": method,
        "checkpoint_selection": "fixed_epoch_last_no_heldout_evaluation",
    }
    mismatches = {
        key: {"actual": resolved.get(key), "expected": value}
        for key, value in expected.items()
        if resolved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"R1-1 adapted checkpoint contract 不匹配: {mismatches}")
    model = build_convnext_tiny_classifier(25, weight_path=weights_path, regime="fine_tune")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_p03_checkpoint_sha256": resolved["source_p03_checkpoint_sha256"],
        "embedded_config": resolved,
    }


def _audit(args: argparse.Namespace, config: dict[str, Any]) -> int:
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if sha256_file(args.training_manifest) != config["inputs"]["training_manifest_sha256"]:
        raise ValueError("N2-v2 training manifest SHA 不匹配")
    if sha256_file(args.inference_manifest) != config["inputs"]["inference_manifest_sha256"]:
        raise ValueError("R1 inference manifest SHA 不匹配")
    rows = load_aircraft_training_rows(args.training_manifest)
    row_audit = audit_aircraft_training_rows(rows)
    records = load_proposal_manifest(args.inference_manifest)
    inference_fold_counts = {
        fold: sum(
            item.fold == fold and item.category_id in AIRCRAFT_CLASS_IDS for item in records
        )
        for fold in (0, 1, 2)
    }
    payload = {
        "contract_version": R11_CONTRACT_VERSION,
        "status": "pass",
        "training": row_audit,
        "inference": {
            "all_proposals": len(records),
            "aircraft_proposals": sum(inference_fold_counts.values()),
            "fold_counts": inference_fold_counts,
        },
        "structural_bypass": {
            "ship_and_vehicle_relabel_allowed": False,
            "cross_coarse_relabel_allowed": False,
        },
    }
    expected = config["inputs"]["expected_counts"]
    if row_audit["row_count"] != int(expected["training_aircraft_rows"]):
        raise ValueError("aircraft training row 数不匹配")
    if payload["inference"]["aircraft_proposals"] != int(expected["inference_aircraft_rows"]):
        raise ValueError("aircraft inference proposal 数不匹配")
    _json(destination / "aircraft_data_audit.json", payload)
    return 0


def _train(args: argparse.Namespace, config: dict[str, Any]) -> int:
    import torch
    import torch.nn.functional as functional
    import torchvision

    _require_stack(torch, torchvision, config)
    if sha256_file(args.training_manifest) != config["inputs"]["training_manifest_sha256"]:
        raise ValueError("N2-v2 training manifest SHA 不匹配")
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    _seed(torch, int(config["training"]["seed"]) + args.fold)
    rows = load_aircraft_training_rows(args.training_manifest)
    train_rows = [row for row in rows if int(row["fold"]) != args.fold]
    heldout_rows_loaded = any(int(row["fold"]) == args.fold for row in train_rows)
    if heldout_rows_loaded or not train_rows:
        raise ValueError("训练折隔离失败")
    expected_p03 = config["inputs"]["p03_checkpoint_sha256"][str(args.fold)]
    student, source_meta = _load_p03_model(
        args.p03_checkpoint,
        args.weights,
        fold=args.fold,
        expected_sha256=expected_p03,
    )
    teacher = None
    if args.method == "selective_anchor_kd":
        teacher, _ = _load_p03_model(
            args.p03_checkpoint,
            args.weights,
            fold=args.fold,
            expected_sha256=expected_p03,
        )
        teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
    student.to(device)
    center_memory = None
    if args.method == "class_center":
        center_config = config["training"]["class_center"]
        classifier_weight = student.classifier[-1].weight[
            AIRCRAFT_CLASS_IDS[0] : AIRCRAFT_CLASS_IDS[-1] + 1
        ]
        center_memory = _ClassCenterMemory(
            classifier_weight,
            momentum=float(center_config["momentum"]),
            margin=float(center_config["margin"]),
            negative_weight=float(center_config["negative_weight"]),
        )
    dataset = _TrainDataset(train_rows, args.data_root.expanduser())
    generator = torch.Generator().manual_seed(int(config["training"]["seed"]) + args.fold)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size or config["runtime"]["train_batch_size"]),
        shuffle=True,
        num_workers=int(args.num_workers if args.num_workers is not None else config["runtime"]["num_workers"]),
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
        persistent_workers=False,
    )
    training = config["training"]
    head_ids = {id(item) for item in student.classifier.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [item for item in student.parameters() if id(item) not in head_ids],
                "lr": float(training["backbone_lr"]),
            },
            {
                "params": list(student.classifier.parameters()),
                "lr": float(training["head_lr"]),
            },
        ],
        weight_decay=float(training["weight_decay"]),
    )
    epochs = 1 if args.smoke else int(training["epochs"])
    total_steps = max(1, epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    temperature = float(training["kd_temperature"])
    kd_weight = float(training["kd_weight"])
    label_smoothing = float(training["label_smoothing"])
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        student.train()
        totals = {
            "samples": 0,
            "loss": 0.0,
            "ce": 0.0,
            "kd": 0.0,
            "center": 0.0,
            "center_pull": 0.0,
            "center_push": 0.0,
            "positive_cosine": 0.0,
            "hardest_negative_cosine": 0.0,
            "correct": 0,
            "anchor": 0,
        }
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels20 = labels.to(device, non_blocking=True) - AIRCRAFT_CLASS_IDS[0]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                if center_memory is None:
                    features = None
                    logits = student(images)
                else:
                    features, logits = _convnext_features_and_logits(student, images)
                logits20 = logits[:, AIRCRAFT_CLASS_IDS[0] : AIRCRAFT_CLASS_IDS[-1] + 1]
                ce = functional.cross_entropy(
                    logits20,
                    labels20,
                    label_smoothing=label_smoothing,
                )
                kd = logits20.new_zeros(())
                anchor_count = 0
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(images)[
                            :, AIRCRAFT_CLASS_IDS[0] : AIRCRAFT_CLASS_IDS[-1] + 1
                        ]
                        anchor_mask = teacher_logits.argmax(dim=1).eq(labels20)
                    anchor_count = int(anchor_mask.sum())
                    if anchor_count:
                        per_sample = functional.kl_div(
                            functional.log_softmax(logits20 / temperature, dim=1),
                            functional.softmax(teacher_logits / temperature, dim=1),
                            reduction="none",
                        ).sum(dim=1)
                        kd = per_sample[anchor_mask].mean() * (temperature**2)
                center = logits20.new_zeros(())
                center_diagnostics = {
                    "center_pull": 0.0,
                    "center_push": 0.0,
                    "positive_cosine": 0.0,
                    "hardest_negative_cosine": 0.0,
                }
                center_weight = 0.0
                if center_memory is not None:
                    if features is None:
                        raise RuntimeError("class center features 未生成")
                    center, center_diagnostics = center_memory.loss(features, labels20)
                    ramp = epoch / max(epochs, 1)
                    center_weight = float(training["class_center"]["loss_weight"]) * ramp
                loss = ce + kd_weight * kd + center_weight * center
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), float(training["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            if center_memory is not None:
                center_memory.update(features, labels20)
            scheduler.step()
            count = int(labels20.numel())
            totals["samples"] += count
            totals["loss"] += float(loss.detach()) * count
            totals["ce"] += float(ce.detach()) * count
            totals["kd"] += float(kd.detach()) * count
            totals["center"] += float(center.detach()) * count
            for key, value in center_diagnostics.items():
                totals[key] += value * count
            totals["correct"] += int(logits20.argmax(dim=1).eq(labels20).sum())
            totals["anchor"] += anchor_count
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / totals["samples"],
            "train_ce": totals["ce"] / totals["samples"],
            "train_kd": totals["kd"] / totals["samples"],
            "train_center": totals["center"] / totals["samples"],
            "center_pull": totals["center_pull"] / totals["samples"],
            "center_push": totals["center_push"] / totals["samples"],
            "positive_cosine": totals["positive_cosine"] / totals["samples"],
            "hardest_negative_cosine": totals["hardest_negative_cosine"]
            / totals["samples"],
            "train_accuracy": totals["correct"] / totals["samples"],
            "teacher_correct_anchor_fraction": totals["anchor"] / totals["samples"],
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    resolved = {
        "contract_version": R11_CONTRACT_VERSION,
        "experiment_id": config["experiment_id"],
        "held_out_fold": args.fold,
        "selection_folds": sorted({0, 1, 2} - {args.fold}),
        "method": args.method,
        "epochs": epochs,
        "source_p03_checkpoint_sha256": source_meta["checkpoint_sha256"],
        "training_manifest_sha256": sha256_file(args.training_manifest),
        "training_rows": len(train_rows),
        "heldout_rows_loaded_or_evaluated": False,
        "checkpoint_selection": "fixed_epoch_last_no_heldout_evaluation",
        "augmentation": "uniform_random_D4",
        "training_only_class_center": (
            dict(config["training"]["class_center"])
            if args.method == "class_center"
            else None
        ),
        "aircraft_class_ids": list(AIRCRAFT_CLASS_IDS),
    }
    checkpoint_path = destination / "final_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "resolved_config": resolved,
            "epoch": epochs,
            "selection_metric": "fixed_epoch_last_no_heldout_evaluation",
        },
        checkpoint_path,
    )
    summary = {
        "status": "smoke_pass" if args.smoke else "complete",
        "resolved_config": resolved,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    _json(destination / "run_summary.json", summary)
    return 0


def _infer(args: argparse.Namespace, config: dict[str, Any]) -> int:
    import torch
    import torchvision

    _require_stack(torch, torchvision, config)
    if sha256_file(args.inference_manifest) != config["inputs"]["inference_manifest_sha256"]:
        raise ValueError("R1 inference manifest SHA 不匹配")
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"fold_{args.fold}_aircraft_bundle.npz"
    if output_path.exists():
        raise FileExistsError(output_path)
    expected_p03 = config["inputs"]["p03_checkpoint_sha256"][str(args.fold)]
    if args.method == "p03":
        model, meta = _load_p03_model(
            args.checkpoint,
            args.weights,
            fold=args.fold,
            expected_sha256=expected_p03,
        )
    else:
        model, meta = _load_adapted_model(
            args.checkpoint,
            args.weights,
            fold=args.fold,
            method=args.method,
        )
        if meta["source_p03_checkpoint_sha256"] != expected_p03:
            raise ValueError("adapted checkpoint 的 source P03 SHA 不匹配")
    device = torch.device(args.device)
    model.to(device).eval()
    records = [
        item
        for item in load_proposal_manifest(args.inference_manifest)
        if item.fold == args.fold and item.category_id in AIRCRAFT_CLASS_IDS
    ]
    if args.smoke:
        records = records[: min(16, len(records))]
    loader = torch.utils.data.DataLoader(
        _InferenceDataset(records, args.data_root.expanduser()),
        batch_size=int(args.batch_size or config["runtime"]["inference_object_batch_size"]),
        shuffle=False,
        num_workers=int(args.num_workers if args.num_workers is not None else config["runtime"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    identity_values: list[np.ndarray] = []
    d4_values: list[np.ndarray] = []
    uids: list[str] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for stacks, batch_uids in loader:
            batch, views = stacks.shape[:2]
            images = stacks.reshape(batch * views, *stacks.shape[2:]).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images).reshape(batch, views, 25)
            logits = logits.float()
            identity_values.append(logits[:, 0, :].cpu().numpy())
            probabilities = logits[:, :, AIRCRAFT_CLASS_IDS[0] : AIRCRAFT_CLASS_IDS[-1] + 1].softmax(dim=2).mean(dim=1)
            d4_values.append(probabilities.cpu().numpy())
            uids.extend(str(uid) for uid in batch_uids)
    identity = np.concatenate(identity_values).astype(np.float32, copy=False)
    d4 = np.concatenate(d4_values).astype(np.float32, copy=False)
    if identity.shape != (len(records), 25) or d4.shape != (len(records), 20):
        raise ValueError("inference bundle shape 非法")
    np.savez_compressed(
        output_path,
        proposal_uids=np.asarray(uids, dtype=f"<U{max(map(len, uids))}"),
        identity_logits=identity,
        d4_probabilities=d4,
    )
    _json(
        destination / f"fold_{args.fold}_runtime.json",
        {
            "contract_version": R11_CONTRACT_VERSION,
            "status": "smoke_pass" if args.smoke else "complete",
            "method": args.method,
            "fold": args.fold,
            "proposal_count": len(records),
            "view_count": len(D4_VIEW_IDS),
            "checkpoint": meta,
            "bundle_sha256": sha256_file(output_path),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
        },
    )
    return 0


def _delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in ("recall", "fdr", "macro_recall", "macro_fdr")
    }


def _evaluate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = load_proposal_manifest(args.inference_manifest)
    if sha256_file(args.inference_manifest) != config["inputs"]["inference_manifest_sha256"]:
        raise ValueError("R1 inference manifest SHA 不匹配")
    for fold in (0, 1, 2):
        base_path = args.base_logits_dir / f"fold_{fold}_logits.npz"
        actual = sha256_file(base_path)
        expected = config["inputs"]["base_logits_sha256"][str(fold)]
        if actual != expected:
            raise ValueError(f"fold{fold} R1-0 base logits SHA 不匹配: {actual}")
    conditions = {
        "p03_identity": (None, "identity"),
        "p03_d4": (args.p03_bundle_dir, "d4"),
        "ce_identity": (args.ce_bundle_dir, "identity"),
        "ce_d4": (args.ce_bundle_dir, "d4"),
        "selective_anchor_kd_identity": (args.kd_bundle_dir, "identity"),
        "selective_anchor_kd_d4": (args.kd_bundle_dir, "d4"),
    }
    if args.center_bundle_dir is not None:
        conditions.update(
            {
                "class_center_identity": (args.center_bundle_dir, "identity"),
                "class_center_d4": (args.center_bundle_dir, "d4"),
            }
        )
    requested_conditions = config.get("conditions", {}).get("evaluate")
    if requested_conditions is not None:
        unknown = set(requested_conditions) - set(conditions)
        if unknown:
            raise ValueError(f"未知 evaluation conditions: {sorted(unknown)}")
        conditions = {name: conditions[name] for name in requested_conditions}
    project = load_config(config["project_config"])
    protocol = parse_evaluation_protocol(project)
    y1_path = args.y1_calibration_result
    formal_path = args.formal_crop_manifest
    if sha256_file(y1_path) != config["inputs"]["y1_calibration_result_sha256"]:
        raise ValueError("Y1 result SHA 不匹配")
    if sha256_file(formal_path) != config["inputs"]["formal_crop_manifest_sha256"]:
        raise ValueError("formal crop manifest SHA 不匹配")
    y1 = json.loads(y1_path.read_text(encoding="utf-8"))
    search = config["search"]
    results: dict[str, Any] = {}
    raw_outputs: dict[str, Any] = {}
    for name, (bundle, view) in conditions.items():
        probabilities = condition_probabilities(
            records,
            args.base_logits_dir,
            bundle,
            view=view,
        )
        result, raw = run_reranking_crossfit(
            records=records,
            probabilities=probabilities,
            aggregate_dir=args.aggregate_dir,
            formal_crop_manifest_path=formal_path,
            y1_calibration_result=y1,
            protocol=protocol,
            variants=build_aircraft_variants(search),
            threshold_start=float(search["threshold_start"]),
            threshold_stop=float(search["threshold_stop"]),
            threshold_step=float(search["threshold_step"]),
            maximum_selection_recall_drop=float(search["maximum_selection_recall_drop"]),
        )
        result["condition"] = name
        result["scientific_scope"] = "aircraft_only_crossfit_development"
        results[name] = result
        raw_outputs[name] = raw
        _json(destination / f"{name}_result.json", result)
    _, _, image_folds = load_cv3_aggregate(args.aggregate_dir, candidate_floor=0.001)
    gt = load_gt_from_formal_crop_manifest(
        formal_path,
        expected_images=int(config["inputs"]["expected_images"]),
        expected_annotations=int(config["inputs"]["expected_annotations"]),
    )
    baseline_name = str(config["conditions"]["reference"])
    primary_name = str(config["conditions"]["primary"])
    if baseline_name not in results or primary_name not in results:
        raise ValueError(
            f"reference/primary condition 未执行: {baseline_name}, {primary_name}"
        )
    baseline_metrics = results[baseline_name]["merged"]["C2_frozen_after_r1"]
    primary_metrics = results[primary_name]["merged"]["C2_frozen_after_r1"]
    c2_outputs = {
        name: apply_frozen_c2(y1, raw, image_folds) for name, raw in raw_outputs.items()
    }
    transition = paired_transition(
        gt,
        c2_outputs[baseline_name],
        c2_outputs[primary_name],
        protocol,
    )
    coarse_parity = {
        coarse: {
            metric: float(primary_metrics["per_coarse"][coarse][metric])
            - float(baseline_metrics["per_coarse"][coarse][metric])
            for metric in ("macro_recall", "macro_fdr", "pooled_recall", "pooled_fdr")
        }
        for coarse in ("ship", "vehicle")
    }
    maximum_bypass_delta = max(
        abs(value) for payload in coarse_parity.values() for value in payload.values()
    )
    aircraft_delta = {
        metric: float(primary_metrics["per_coarse"]["aircraft"][metric])
        - float(baseline_metrics["per_coarse"]["aircraft"][metric])
        for metric in ("macro_recall", "macro_fdr", "pooled_recall", "pooled_fdr")
    }
    gates = config["decision_gates"]
    primary_pass = bool(
        primary_metrics["recall"] >= float(gates["official_recall_min"])
        and primary_metrics["fdr"] <= float(gates["official_fdr_max"])
        and aircraft_delta["macro_recall"] >= -float(gates["aircraft_macro_recall_tolerance"])
        and aircraft_delta["macro_fdr"] <= float(gates["aircraft_macro_fdr_tolerance"])
        and transition["net_tp"] >= 0
        and maximum_bypass_delta <= 1e-12
    )
    summary_conditions = {
        name: {
            "selected_variants": [item["selected_variant"] for item in result["per_fold"]],
            "C2": result["merged"]["C2_frozen_after_r1"],
            "transition_vs_C2": result["merged"]["paired_transition_c2_r1_vs_c2"],
        }
        for name, result in results.items()
    }
    decision = {
        "contract_version": R11_CONTRACT_VERSION,
        "status": "complete",
        "primary_condition": primary_name,
        "reference_condition": baseline_name,
        "primary_gate_passed": primary_pass,
        "formal_admission": False,
        "reason_formal_false": "iterative development on the same formal OOF corpus; requires final system confirmation",
        "delta_primary_vs_reference": _delta(primary_metrics, baseline_metrics),
        "aircraft_delta": aircraft_delta,
        "ship_vehicle_structural_parity": coarse_parity,
        "maximum_bypass_delta": maximum_bypass_delta,
        "paired_transition_primary_vs_reference": transition,
        "next_action": (
            f"retain_{primary_name}_for_final_system_confirmation"
            if primary_pass
            else "retain_reference_and_stop_current_refinement_method"
        ),
    }
    _json(destination / "condition_summary.json", summary_conditions)
    _json(destination / "decision.json", decision)
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, required=True)
    sub = value.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--training-manifest", type=Path, required=True)
    audit.add_argument("--inference-manifest", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--training-manifest", type=Path, required=True)
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--weights", type=Path, required=True)
    train.add_argument("--p03-checkpoint", type=Path, required=True)
    train.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    train.add_argument(
        "--method", choices=("ce", "selective_anchor_kd", "class_center"), required=True
    )
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--batch-size", type=int)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--smoke", action="store_true")
    infer = sub.add_parser("infer")
    infer.add_argument("--inference-manifest", type=Path, required=True)
    infer.add_argument("--data-root", type=Path, required=True)
    infer.add_argument("--weights", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    infer.add_argument(
        "--method",
        choices=("p03", "ce", "selective_anchor_kd", "class_center"),
        required=True,
    )
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--batch-size", type=int)
    infer.add_argument("--num-workers", type=int)
    infer.add_argument("--smoke", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--inference-manifest", type=Path, required=True)
    evaluate.add_argument("--base-logits-dir", type=Path, required=True)
    evaluate.add_argument("--p03-bundle-dir", type=Path, required=True)
    evaluate.add_argument("--ce-bundle-dir", type=Path, required=True)
    evaluate.add_argument("--kd-bundle-dir", type=Path, required=True)
    evaluate.add_argument("--center-bundle-dir", type=Path)
    evaluate.add_argument("--aggregate-dir", type=Path, required=True)
    evaluate.add_argument("--formal-crop-manifest", type=Path, required=True)
    evaluate.add_argument("--y1-calibration-result", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = _config(args.config)
    if args.command == "audit":
        return _audit(args, config)
    if args.command == "train":
        return _train(args, config)
    if args.command == "infer":
        return _infer(args, config)
    return _evaluate(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
