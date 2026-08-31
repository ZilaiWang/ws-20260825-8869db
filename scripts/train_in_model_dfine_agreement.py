#!/usr/bin/env python3
"""Distill OOF D-FINE support into a zero-impact YOLO ROI quality branch."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_rows(
    image_ids: np.ndarray,
    detector_scores: np.ndarray,
    support: np.ndarray,
    eligible: np.ndarray,
    *,
    max_per_image: int,
    priority: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Keep risk-labelled rows, then balance supported/unsupported evidence."""
    if max_per_image < 2:
        raise ValueError("max_per_image must be >= 2")
    if priority is None:
        priority = np.zeros_like(eligible, dtype=bool)
    if priority.shape != eligible.shape:
        raise ValueError("priority and eligible masks must align")
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in np.flatnonzero(eligible):
        grouped[int(image_ids[index])].append(int(index))
    output: dict[int, np.ndarray] = {}
    positive_cap = max_per_image // 2
    for image_id, raw in sorted(grouped.items()):
        protected = sorted(
            (index for index in raw if priority[index]),
            key=lambda index: (-float(detector_scores[index]), index),
        )[:max_per_image]
        remaining_raw = [index for index in raw if index not in protected]
        positive = sorted(
            (index for index in remaining_raw if support[index] > 0.0),
            key=lambda index: (-float(support[index]), -float(detector_scores[index]), index),
        )
        negative = sorted(
            (index for index in remaining_raw if support[index] <= 0.0),
            key=lambda index: (-float(detector_scores[index]), index),
        )
        available = max_per_image - len(protected)
        positive_slots = min(positive_cap, (available + 1) // 2)
        chosen = protected + positive[:positive_slots]
        chosen += negative[: max_per_image - len(chosen)]
        if len(chosen) < max_per_image:
            remaining = [
                index for index in positive[positive_slots:] + negative if index not in chosen
            ]
            remaining.sort(
                key=lambda index: (
                    -float(detector_scores[index] * max(support[index], 0.05)),
                    index,
                )
            )
            chosen.extend(remaining[: max_per_image - len(chosen)])
        output[image_id] = np.asarray(sorted(set(chosen)), dtype=np.int64)
    return output


def _letterbox_tensor(image: Image.Image, target_size: int, device: str):
    import torch

    from rsdet.innovation.in_model_agreement import letterbox_geometry

    image = image.convert("RGB")
    geometry = letterbox_geometry(image.width, image.height, target_size)
    resized = image.resize(
        (geometry.resized_width, geometry.resized_height), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    left = int(round(geometry.pad_left - 0.1))
    top = int(round(geometry.pad_top - 0.1))
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32).copy().transpose(2, 0, 1) / 255.0
    return torch.from_numpy(array).unsqueeze(0).to(device), geometry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-out-fold", type=int, choices=(-1, 0, 1, 2), required=True)
    parser.add_argument("--mode", choices=("branch_only", "terminal_fpn"), required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max-proposals-per-image", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--match-weight", type=float, default=1.0)
    parser.add_argument("--match-rank-weight", type=float, default=0.5)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--residual-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.epochs <= 0 or args.imgsz <= 0:
        raise ValueError("epochs and imgsz must be positive")
    actual_sha = _sha256(args.base_checkpoint)
    if actual_sha != args.expected_base_sha256.lower():
        raise ValueError("base checkpoint SHA mismatch")
    checkpoint_path = args.output_dir / "adapter_last.pt"
    detector_checkpoint_path = args.output_dir / "adapted_detector.pt"
    if checkpoint_path.exists() or detector_checkpoint_path.exists():
        raise FileExistsError(checkpoint_path if checkpoint_path.exists() else detector_checkpoint_path)

    import torch
    from torch.nn import functional as F

    from rsdet.innovation.in_model_agreement import (
        AgreementResidualHead,
        apply_logit_residual,
        boxes_to_letterbox,
        normalized_feature_anchor,
        pairwise_support_ranking_loss,
        proposal_metadata,
    )
    from rsdet.innovation.yolo_feature_quality import (
        MultiScaleROIFeatureEncoder,
        YoloPyramidTap,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    with np.load(args.teacher_cache, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    required = {
        "image_id",
        "fold",
        "category_id",
        "detector_score",
        "bbox_xyxy",
        "teacher_support_score",
        "protected_tp",
        "active_fp",
        "active_mask",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"teacher cache is missing {sorted(missing)}")
    count = len(arrays["image_id"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("teacher cache arrays are not row aligned")
    eligible = arrays["category_id"].astype(np.int64) == 24
    if args.held_out_fold >= 0:
        eligible &= arrays["fold"].astype(np.int64) != args.held_out_fold
    grouped = _select_rows(
        arrays["image_id"].astype(np.int64),
        arrays["detector_score"].astype(np.float32),
        arrays["teacher_support_score"].astype(np.float32),
        eligible,
        max_per_image=args.max_proposals_per_image,
        priority=arrays["active_mask"].astype(bool),
    )
    if args.max_images > 0:
        # A smoke run must exercise the official-match loss instead of merely
        # taking the smallest image IDs, which may contain support-only rows.
        ordered_images = sorted(
            grouped,
            key=lambda image_id: (
                not bool(arrays["active_mask"][grouped[image_id]].astype(bool).any()),
                image_id,
            ),
        )
        grouped = {image_id: grouped[image_id] for image_id in ordered_images[: args.max_images]}
    if not grouped:
        raise ValueError("no eligible vehicle proposals")

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    sample_paths = {
        int(row["image_id"]): (args.data_root / row["relative_path"]).resolve()
        for row in manifest["samples"]
    }
    unknown = sorted(set(grouped) - set(sample_paths))
    if unknown:
        raise ValueError(f"teacher image IDs missing from split manifest: {unknown[:10]}")
    for image_id in grouped:
        if not sample_paths[image_id].is_file():
            raise FileNotFoundError(sample_paths[image_id])

    from ultralytics import YOLO

    student = YOLO(str(args.base_checkpoint)).model.to(device).eval()
    layers = student.model
    detect = layers[-1]
    layer_indices = tuple(int(value) for value in detect.f)
    strides = tuple(int(value) for value in detect.stride.tolist())
    if len(layer_indices) != 3 or len(strides) != 3:
        raise ValueError("expected a three-level YOLO Detect pyramid")
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    trainable_layer_indices: tuple[int, ...] = ()
    reference = None
    reference_tap = None
    if args.mode == "terminal_fpn":
        trainable_layer_indices = layer_indices
        for index in trainable_layer_indices:
            for parameter in layers[index].parameters():
                parameter.requires_grad_(True)
        reference = YOLO(str(args.base_checkpoint)).model.to(device).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

    student_tap = YoloPyramidTap(student, layer_indices)
    if reference is not None:
        reference_tap = YoloPyramidTap(reference, layer_indices)
    first_image_id = next(iter(grouped))
    first_tensor, _ = _letterbox_tensor(
        Image.open(sample_paths[first_image_id]), args.imgsz, str(device)
    )
    student_tap.clear()
    with torch.no_grad():
        student(first_tensor)
    first_features = student_tap.features(detach=True)
    channels = tuple(int(value.shape[1]) for value in first_features)
    roi_encoder = MultiScaleROIFeatureEncoder(
        channels,
        strides,
        projection_dim=args.projection_dim,
        output_size=3,
        context_ratio=2.0,
    ).to(device)
    quality_head = AgreementResidualHead(
        roi_encoder.output_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    trainable = list(roi_encoder.parameters()) + list(quality_head.parameters())
    trainable += [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Exact zero-impact gate before the first optimizer update.
    probe_scores = torch.tensor([0.01, 0.2, 0.8], device=device)
    probe_residual = quality_head(
        torch.zeros((3, roi_encoder.output_dim), device=device),
        torch.zeros((3, quality_head.metadata_dim), device=device),
    )
    initial_delta = float(
        (apply_logit_residual(probe_scores, probe_residual, alpha=args.residual_alpha) - probe_scores)
        .abs()
        .max()
    )
    if initial_delta > 1e-7:
        raise RuntimeError(f"fresh agreement branch is not exact identity: {initial_delta}")

    image_ids = list(grouped)
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch)
        rng.shuffle(image_ids)
        totals = defaultdict(float)
        steps = 0
        for image_id in image_ids:
            indices = grouped[image_id]
            image = Image.open(sample_paths[image_id]).convert("RGB")
            tensor, geometry = _letterbox_tensor(image, args.imgsz, str(device))
            boxes = torch.from_numpy(arrays["bbox_xyxy"][indices].astype(np.float32)).to(device)
            model_boxes = boxes_to_letterbox(boxes, geometry)
            boxes_with_batch = torch.cat(
                (torch.zeros((len(boxes), 1), device=device), model_boxes), dim=1
            )
            scores = torch.from_numpy(
                arrays["detector_score"][indices].astype(np.float32)
            ).to(device)
            targets = torch.from_numpy(
                arrays["teacher_support_score"][indices].astype(np.float32)
            ).to(device)
            active = torch.from_numpy(
                arrays["active_mask"][indices].astype(bool)
            ).to(device)
            match_targets = torch.from_numpy(
                arrays["protected_tp"][indices].astype(np.float32)
            ).to(device)

            student_tap.clear()
            student(tensor)
            feature_maps = student_tap.features(detach=False)
            reference_features = None
            if reference is not None and reference_tap is not None:
                reference_tap.clear()
                with torch.no_grad():
                    reference(tensor)
                reference_features = reference_tap.features(detach=True)
            roi = roi_encoder(
                feature_maps,
                boxes_with_batch,
                image_height=args.imgsz,
                image_width=args.imgsz,
            )
            metadata = proposal_metadata(scores, model_boxes, image_size=args.imgsz)
            residual = quality_head(roi, metadata)
            support_loss = F.binary_cross_entropy_with_logits(residual, targets)
            support_rank_loss = pairwise_support_ranking_loss(residual, targets)
            match_loss = residual.sum() * 0.0
            match_rank_loss = residual.sum() * 0.0
            if bool(active.any()):
                active_logits = residual[active]
                active_targets = match_targets[active]
                raw_match = F.binary_cross_entropy_with_logits(
                    active_logits,
                    active_targets,
                    reduction="none",
                )
                positive_count = active_targets.sum().clamp_min(1.0)
                negative_count = (1.0 - active_targets).sum().clamp_min(1.0)
                balanced_weight = torch.where(
                    active_targets > 0.5,
                    0.5 / positive_count,
                    0.5 / negative_count,
                )
                match_loss = (raw_match * balanced_weight).sum() / balanced_weight.sum()
                match_rank_loss = pairwise_support_ranking_loss(
                    active_logits,
                    active_targets,
                    positive_floor=0.5,
                )
            anchor_loss = residual.sum() * 0.0
            if reference_features is not None:
                anchor_loss = normalized_feature_anchor(feature_maps, reference_features)
            loss = (
                support_loss
                + args.rank_weight * support_rank_loss
                + args.match_weight * match_loss
                + args.match_rank_weight * match_rank_loss
                + args.anchor_weight * anchor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 2.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["support_loss"] += float(support_loss.detach())
            totals["support_rank_loss"] += float(support_rank_loss.detach())
            totals["match_loss"] += float(match_loss.detach())
            totals["match_rank_loss"] += float(match_rank_loss.detach())
            totals["anchor_loss"] += float(anchor_loss.detach())
            steps += 1
        history.append(
            {
                "epoch": epoch + 1,
                **{name: value / steps for name, value in sorted(totals.items())},
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    state = {
        "protocol": "in_model_dfine_agreement_and_official_match_distillation_v2",
        "base_checkpoint_sha256": actual_sha,
        "mode": args.mode,
        "imgsz": args.imgsz,
        "layer_indices": layer_indices,
        "strides": strides,
        "channels": channels,
        "projection_dim": args.projection_dim,
        "hidden_dim": args.hidden_dim,
        "context_ratio": 2.0,
        "residual_alpha": args.residual_alpha,
        "trainable_layer_indices": trainable_layer_indices,
        "fpn_state_dict": {
            str(index): layers[index].state_dict() for index in trainable_layer_indices
        },
        "roi_encoder_state_dict": roi_encoder.state_dict(),
        "quality_head_state_dict": quality_head.state_dict(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
    student_tap.close()
    if reference_tap is not None:
        reference_tap.close()
    # Materialize the proposal generator as a normal Ultralytics checkpoint so
    # terminal-FPN arms regenerate proposals before the quality branch is read.
    try:
        base_payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        base_payload = torch.load(args.base_checkpoint, map_location="cpu")
    if not isinstance(base_payload, dict) or "model" not in base_payload:
        raise ValueError("base checkpoint is not an Ultralytics checkpoint dictionary")
    base_payload["model"] = copy.deepcopy(student).cpu().half()
    base_payload["ema"] = None
    base_payload["optimizer"] = None
    torch.save(base_payload, detector_checkpoint_path)
    selected_indices = np.concatenate(list(grouped.values()))
    result = {
        "status": "complete_smoke" if args.max_images > 0 else "complete_diagnostic",
        "protocol": state["protocol"],
        "scientific_scope": (
            "outer-fold candidate only; formal admission requires three held-out adapters "
            "and frozen Hard10K plus source-disjoint sentinel evaluation"
        ),
        "teacher_cache_sha256": _sha256(args.teacher_cache),
        "split_manifest_sha256": _sha256(args.split_manifest),
        "base_checkpoint_sha256": actual_sha,
        "held_out_fold": args.held_out_fold,
        "training_folds": sorted(
            set(arrays["fold"][selected_indices].astype(np.int64).tolist())
        ),
        "mode": args.mode,
        "training_image_count": len(grouped),
        "training_proposal_count": int(len(selected_indices)),
        "supported_proposal_count": int(
            (arrays["teacher_support_score"][selected_indices] > 0).sum()
        ),
        "official_match_active_count": int(
            arrays["active_mask"][selected_indices].astype(bool).sum()
        ),
        "official_match_tp_count": int(
            arrays["protected_tp"][selected_indices].astype(bool).sum()
        ),
        "official_match_fp_count": int(
            arrays["active_fp"][selected_indices].astype(bool).sum()
        ),
        "initial_max_abs_score_delta": initial_delta,
        "feature_layer_indices": layer_indices,
        "feature_strides": strides,
        "feature_channels": channels,
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()),
        "adapted_detector_checkpoint": str(detector_checkpoint_path.resolve()),
    }
    result_path = args.output_dir / "training_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    result["checkpoint_sha256"] = _sha256(checkpoint_path)
    result["adapted_detector_checkpoint_sha256"] = _sha256(detector_checkpoint_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("IN_MODEL_DFINE_AGREEMENT_TRAINING_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
