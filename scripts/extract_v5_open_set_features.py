#!/usr/bin/env python3
"""Extract fold-heldout tight/context P03 embeddings for every Y5 proposal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.hera_guard.manifest import square_crop_box


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(path: Path, device):
    import torch
    from torchvision.models import convnext_tiny

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model = convnext_tiny(weights=None, num_classes=25)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def _embed(model, images):
    feature_map = model.features(images)
    pooled = model.avgpool(feature_map)
    normalized = model.classifier[0](pooled)
    return model.classifier[1](normalized)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-pattern", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--context-scale", type=float, default=1.25)
    parser.add_argument(
        "--coarse-filter",
        choices=("ship", "aircraft", "vehicle"),
        help="Optional selective route; omitted means all proposals.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.resolution <= 0 or args.context_scale <= 1.0 or args.batch_size <= 0:
        raise ValueError("invalid extraction geometry/batch contract")

    import torch

    from rsdet.data.crop_classification import render_crop

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig", newline="")))
    if not rows or [int(row["candidate_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("manifest candidate_index must be contiguous and ordered")
    if args.coarse_filter is not None:
        rows = [row for row in rows if row["coarse"] == args.coarse_filter]
        if not rows:
            raise ValueError(f"coarse filter has no rows: {args.coarse_filter}")
    n = len(rows)
    candidate_index = np.asarray([int(row["candidate_index"]) for row in rows], dtype=np.int64)
    tight = np.empty((n, 768), dtype=np.float16)
    context = np.empty((n, 768), dtype=np.float16)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    labels = np.asarray([int(row["open_set_label"]) for row in rows], dtype=np.int64)
    coarse = np.asarray(
        [{"ship": 0, "aircraft": 1, "vehicle": 2}[row["coarse"]] for row in rows],
        dtype=np.int64,
    )
    detector_score = np.asarray([float(row["detector_score"]) for row in rows], dtype=np.float32)
    image_id = np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64)
    category_id = np.asarray([int(row["category_id"]) for row in rows], dtype=np.int64)
    bbox_xyxy = np.asarray(
        [
            [float(row[name]) for name in ("x0", "y0", "x1", "y1")]
            for row in rows
        ],
        dtype=np.float32,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)
    cache: OrderedDict[Path, Image.Image] = OrderedDict()
    checkpoint_sha: dict[str, str] = {}
    started = torch.cuda.Event(enable_timing=True) if str(args.device).startswith("cuda") else None
    finished = torch.cuda.Event(enable_timing=True) if started is not None else None
    if started is not None:
        started.record()

    for fold in range(3):
        checkpoint = Path(args.checkpoint_pattern.format(fold=fold))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_sha[str(fold)] = _sha256(checkpoint)
        model = _load_model(checkpoint, torch.device(args.device))
        indices = np.flatnonzero(folds == fold)
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            rendered: list[torch.Tensor] = []
            for index in batch_indices:
                row = rows[int(index)]
                path = (args.data_root / row["relative_path"]).resolve()
                image = cache.get(path)
                if image is None:
                    image = Image.open(path).convert("RGB")
                    cache[path] = image
                    if len(cache) > 32:
                        _, old = cache.popitem(last=False)
                        old.close()
                else:
                    cache.move_to_end(path)
                box = tuple(float(row[name]) for name in ("x0", "y0", "x1", "y1"))
                for scale in (1.0, args.context_scale):
                    crop = render_crop(
                        image,
                        square_crop_box(box, scale=scale),
                        args.resolution,
                    )
                    array = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)
                    rendered.append(torch.from_numpy(array))
            images = torch.stack(rendered).to(args.device, non_blocking=True)
            images = (images / 255.0 - mean) / std
            with torch.inference_mode():
                embedding = _embed(model, images).float().cpu().numpy()
            tight[batch_indices] = embedding[0::2].astype(np.float16)
            context[batch_indices] = embedding[1::2].astype(np.float16)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for image in cache.values():
        image.close()
    elapsed_ms = None
    if finished is not None:
        finished.record()
        torch.cuda.synchronize()
        elapsed_ms = float(started.elapsed_time(finished))
    if not np.isfinite(tight).all() or not np.isfinite(context).all():
        raise RuntimeError("extracted embeddings contain NaN/Inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        tight_embedding=tight,
        context_embedding=context,
        open_set_label=labels,
        coarse_id=coarse,
        detector_score=detector_score,
        fold=folds,
        candidate_index=candidate_index,
        image_id=image_id,
        category_id=category_id,
        bbox_xyxy=bbox_xyxy,
    )
    payload = {
        "status": "complete",
        "protocol": "v5_fold_heldout_p03_tight_context_embedding_v1",
        "rows": n,
        "embedding_dim": 768,
        "resolution": args.resolution,
        "context_scale": args.context_scale,
        "coarse_filter": args.coarse_filter,
        "checkpoint_sha256": checkpoint_sha,
        "nonfinite": 0,
        "gpu_elapsed_ms": elapsed_ms,
        "output_sha256": _sha256(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
