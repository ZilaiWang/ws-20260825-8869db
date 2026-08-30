#!/usr/bin/env python3
"""Outer-CV3 train/infer the HERA-Guard V3 seven-channel verifier."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.hera_guard.dual_view import build_dual_view_verifier, render_seven_channel_view
from rsdet.hera_guard.metric_aligned import CANONICAL, build_metric_aligned_roles
from rsdet.hera_guard.metric_risk import align_anchor_scores, safe_logit
from rsdet.models.crop_classifier import sha256_file
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box)
    return x, y, x + width, y + height


def _tensor(view: np.ndarray, *, augment_code: int = 0) -> Any:
    import torch

    array = view
    rotation = augment_code % 4
    if rotation:
        array = np.rot90(array, k=rotation, axes=(1, 2))
    if augment_code >= 4:
        array = array[:, :, ::-1]
    array = np.ascontiguousarray(array).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406] * 2, dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225] * 2, dtype=np.float32)[:, None, None]
    array[:6] = (array[:6] - mean) / std
    array[6] = array[6] * 2.0 - 1.0
    return torch.from_numpy(array)


def _load_images(
    *,
    pseudo_root: Path,
    image_meta: dict[int, dict[str, Any]],
    folds: set[int],
    image_layout: str,
) -> dict[int, Image.Image]:
    result = {}
    Image.MAX_IMAGE_PIXELS = None
    for image_id, meta in sorted(image_meta.items()):
        fold = int(meta["fold"])
        if fold not in folds:
            continue
        if image_layout == "pseudo_fold":
            path = pseudo_root / f"fold_{fold}" / "images" / str(meta["file_name"])
        elif image_layout == "relative":
            path = pseudo_root / str(meta["file_name"])
        else:  # protected by argparse, retained as a defensive contract
            raise ValueError(f"unsupported image layout: {image_layout}")
        if not path.is_file():
            raise FileNotFoundError(f"dual-view source image is missing: {path}")
        with Image.open(path) as source:
            result[image_id] = source.convert("RGB")
            result[image_id].load()
    return result


def _batch_tensor(
    indices: np.ndarray,
    *,
    records: list[dict[str, Any]],
    images: dict[int, Image.Image],
    resolution: int,
    context_ratio: float,
    augment_codes: np.ndarray | None,
    executor: concurrent.futures.Executor | None = None,
    view_cache: np.ndarray | None = None,
) -> Any:
    def render_one(arguments: tuple[int, int]) -> Any:
        offset, index = arguments
        if view_cache is None:
            row = records[index]
            view = render_seven_channel_view(
                images[int(row["image_id"])],
                _xywh_to_xyxy(row["bbox"]),
                resolution=resolution,
                context_ratio=context_ratio,
            )
        else:
            view = view_cache[index]
        code = 0 if augment_codes is None else int(augment_codes[offset])
        return _tensor(view, augment_code=code)

    arguments = list(enumerate(indices.tolist()))
    views = (
        list(map(render_one, arguments))
        if executor is None
        else list(executor.map(render_one, arguments))
    )
    return __import__("torch").stack(views)


def _precache_views(
    *,
    records: list[dict[str, Any]],
    images: dict[int, Image.Image],
    resolution: int,
    context_ratio: float,
    workers: int,
) -> np.ndarray:
    """Render every deterministic base view once into host RAM.

    The cache is deliberately uint8.  Augmentation and normalization remain in
    ``_tensor`` and therefore have exactly the same numerical contract as the
    streaming renderer.
    """

    cache = np.empty((len(records), 7, resolution, resolution), dtype=np.uint8)

    def render(index: int) -> tuple[int, np.ndarray]:
        row = records[index]
        return (
            index,
            render_seven_channel_view(
                images[int(row["image_id"])],
                _xywh_to_xyxy(row["bbox"]),
                resolution=resolution,
                context_ratio=context_ratio,
            ),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for index, view in executor.map(render, range(len(records))):
            cache[index] = view
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--evidence-pred", type=Path, required=True)
    parser.add_argument("--anchor-pred", type=Path, required=True)
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument(
        "--image-layout",
        choices=("pseudo_fold", "relative"),
        default="pseudo_fold",
        help="pseudo_fold uses fold_N/images/file_name; relative uses root/file_name",
    )
    parser.add_argument("--imagenet-weight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=96)
    parser.add_argument("--render-workers", type=int, default=8)
    parser.add_argument(
        "--precache-views",
        action="store_true",
        help="render deterministic uint8 7-channel views once into host RAM",
    )
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--context-ratio", type=float, default=1.75)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--residual-limit", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batch_size < 4 or args.batch_size % 2:
        raise ValueError("batch size must be an even integer >=4")

    import torch
    from torch.nn import functional

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt_raw = json.loads(args.gt.read_text(encoding="utf-8"))
    image_meta = {int(item["id"]): item for item in gt_raw["images"]}
    evidence: list[dict[str, Any]] = json.loads(args.evidence_pred.read_text(encoding="utf-8"))
    anchor: list[dict[str, Any]] = json.loads(args.anchor_pred.read_text(encoding="utf-8"))
    anchor_scores = np.asarray(align_anchor_scores(evidence, anchor), dtype=np.float32)
    role_predictions = [
        {**row, "score": float(score)}
        for row, score in zip(evidence, anchor_scores, strict=True)
    ]
    roles = build_metric_aligned_roles(
        gt_boxes=load_coco_ground_truth(args.gt),
        predictions=role_predictions,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    ).roles
    folds = np.asarray([int(row["source_fold"]) for row in evidence], dtype=np.int64)
    foreground = np.asarray([bool(row.object_group_id) for row in roles])
    canonical = np.asarray([row.role == CANONICAL for row in roles], dtype=np.float32)
    fine = np.asarray(
        [-1 if row.support_category_id is None else row.support_category_id for row in roles],
        dtype=np.int64,
    )
    quality = np.asarray([min(max(row.support_iou, 0.0), 1.0) for row in roles], dtype=np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    oof_scores = np.full(len(evidence), np.nan, dtype=np.float64)
    fold_summaries = []
    view_cache: np.ndarray | None = None
    if args.precache_views:
        all_images = _load_images(
            pseudo_root=args.pseudo_root,
            image_meta=image_meta,
            folds={0, 1, 2},
            image_layout=args.image_layout,
        )
        view_cache = _precache_views(
            records=evidence,
            images=all_images,
            resolution=args.resolution,
            context_ratio=args.context_ratio,
            workers=args.render_workers,
        )
        del all_images

    for held_out in (0, 1, 2):
        rng = np.random.default_rng(args.seed + held_out)
        train = np.flatnonzero(folds != held_out)
        validation = np.flatnonzero(folds == held_out)
        positives = train[foreground[train]]
        negatives = train[~foreground[train]]
        if not len(positives) or not len(negatives):
            raise ValueError(f"fold {held_out} lacks positive/negative training candidates")
        train_images = (
            {}
            if view_cache is not None
            else _load_images(
                pseudo_root=args.pseudo_root,
                image_meta=image_meta,
                folds={0, 1, 2} - {held_out},
                image_layout=args.image_layout,
            )
        )
        model = build_dual_view_verifier(weight_path=args.imagenet_weight).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
        history = []
        model.train()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.render_workers
        ) as render_executor:
            for epoch in range(args.epochs):
                totals = []
                for _ in range(args.batches_per_epoch):
                    half = args.batch_size // 2
                    indices = np.concatenate(
                        [
                            rng.choice(positives, half, replace=True),
                            rng.choice(negatives, half, replace=True),
                        ]
                    )
                    rng.shuffle(indices)
                    images = _batch_tensor(
                        indices,
                        records=evidence,
                        images=train_images,
                        resolution=args.resolution,
                        context_ratio=args.context_ratio,
                        augment_codes=rng.integers(0, 8, size=len(indices)),
                        executor=render_executor,
                        view_cache=view_cache,
                    ).to(device, non_blocking=True)
                    target_fg = torch.from_numpy(foreground[indices].astype(np.float32)).to(device)
                    target_canonical = torch.from_numpy(canonical[indices]).to(device)
                    target_fine = torch.from_numpy(fine[indices]).to(device)
                    target_quality = torch.from_numpy(quality[indices]).to(device)
                    base_logits = torch.as_tensor(
                        [safe_logit(anchor_scores[index]) for index in indices],
                        dtype=torch.float32,
                        device=device,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    output = model(images)
                    risk_logits = base_logits + args.residual_limit * torch.tanh(
                        output.risk_residual_logit
                    )
                    risk_loss = functional.binary_cross_entropy_with_logits(
                        risk_logits,
                        target_canonical,
                        pos_weight=torch.as_tensor(12.0, device=device),
                    )
                    foreground_loss = functional.binary_cross_entropy_with_logits(
                        output.foreground_logit, target_fg
                    )
                    positive = target_fg > 0.5
                    fine_loss = functional.cross_entropy(
                        output.fine_logits[positive], target_fine[positive]
                    )
                    quality_loss = functional.binary_cross_entropy_with_logits(
                        output.quality_logit, target_quality
                    )
                    loss = (
                        risk_loss
                        + 0.5 * foreground_loss
                        + 0.5 * fine_loss
                        + 0.25 * quality_loss
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    totals.append(float(loss.detach().cpu()))
                history.append(
                    {"epoch": epoch + 1, "mean_loss": float(np.mean(totals))}
                )
        del train_images

        validation_images = (
            {}
            if view_cache is not None
            else _load_images(
                pseudo_root=args.pseudo_root,
                image_meta=image_meta,
                folds={held_out},
                image_layout=args.image_layout,
            )
        )
        model.eval()
        with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(
            max_workers=args.render_workers
        ) as render_executor:
            for start in range(0, len(validation), args.inference_batch_size):
                indices = validation[start : start + args.inference_batch_size]
                images = _batch_tensor(
                    indices,
                    records=evidence,
                    images=validation_images,
                    resolution=args.resolution,
                    context_ratio=args.context_ratio,
                    augment_codes=None,
                    executor=render_executor,
                    view_cache=view_cache,
                ).to(device, non_blocking=True)
                output = model(images)
                base_logits = torch.as_tensor(
                    [safe_logit(anchor_scores[index]) for index in indices],
                    dtype=torch.float32,
                    device=device,
                )
                logits = base_logits + args.residual_limit * torch.tanh(
                    output.risk_residual_logit
                )
                oof_scores[indices] = torch.sigmoid(logits).cpu().numpy()
        del validation_images
        checkpoint = args.output_dir / f"dual_view_fold{held_out}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "held_out_fold": held_out,
                "resolution": args.resolution,
                "context_ratio": args.context_ratio,
                "residual_limit": args.residual_limit,
            },
            checkpoint,
        )
        fold_summaries.append(
            {
                "held_out_fold": held_out,
                "n_train": len(train),
                "n_validation": len(validation),
                "history": history,
                "checkpoint_sha256": _sha256(checkpoint),
            }
        )
        del model
        torch.cuda.empty_cache()

    if not np.isfinite(oof_scores).all():
        raise RuntimeError("dual-view OOF inference is incomplete")
    predictions = []
    for row, score in zip(evidence, oof_scores, strict=True):
        predictions.append(
            {
                "image_id": int(row["image_id"]),
                "category_id": int(row["category_id"]),
                "bbox": [float(value) for value in row["bbox"]],
                "score": float(score),
                "source_fold": int(row["source_fold"]),
                "source_model": str(row.get("source_model", "")),
            }
        )
    prediction_path = args.output_dir / "dual_view_oof_predictions.json"
    prediction_path.write_text(
        json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "dual_view_metric_verifier_oof_complete",
        "protocol": "outer_cv3_7ch_tight_masked_context_mask_v1",
        "input_candidates": len(evidence),
        "resolution": args.resolution,
        "context_ratio": args.context_ratio,
        "image_layout": args.image_layout,
        "precache_views": args.precache_views,
        "view_cache_bytes": 0 if view_cache is None else int(view_cache.nbytes),
        "residual_limit": args.residual_limit,
        "folds": fold_summaries,
        "input_sha256": {
            "gt": _sha256(args.gt),
            "evidence_pred": _sha256(args.evidence_pred),
            "anchor_pred": _sha256(args.anchor_pred),
            "imagenet_weight": sha256_file(args.imagenet_weight),
        },
        "predictions_sha256": _sha256(prediction_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
