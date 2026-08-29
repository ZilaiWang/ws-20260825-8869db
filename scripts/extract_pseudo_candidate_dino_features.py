#!/usr/bin/env python3
"""Extract frozen DINOv2-B CLS+patch-mean features for pseudo-10K candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.background_gate import expand_context_bbox
from rsdet.data.crop_classification import render_crop
from rsdet.features.p04_teachers import DinoV2Adapter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box)
    return x, y, x + width, y + height


def _batches(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--weight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--context-ratio", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    predictions = json.loads(args.pred.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in gt["images"]}
    adapter = DinoV2Adapter(
        architecture="dinov2_vitb14",
        repo=args.dino_repo,
        weights=args.dino_weight,
        expected_weight_sha256=args.weight_sha256,
        device=args.device,
        compute_dtype="float16",
        include_patch_mean=True,
    )
    features = np.empty((len(predictions), 1536), dtype=np.float16)
    by_image: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(predictions):
        by_image[int(item["image_id"])].append(index)
    started = time.monotonic()
    Image.MAX_IMAGE_PIXELS = None
    for image_id, indices in sorted(by_image.items()):
        meta = images[image_id]
        fold = int(meta["fold"])
        path = args.pseudo_root / f"fold_{fold}" / "images" / meta["file_name"]
        with Image.open(path) as source:
            source.load()
            for batch_indices in _batches(indices, args.batch_size):
                crops = [
                    render_crop(
                        source,
                        expand_context_bbox(
                            _xywh_to_xyxy(predictions[index]["bbox"]),
                            ratio=args.context_ratio,
                        ),
                        args.resolution,
                    )
                    for index in batch_indices
                ]
                result = adapter.extract(
                    crops,
                    sample_keys=[f"candidate-{index}" for index in batch_indices],
                )["dino_cls_patchmean"]
                if result.shape != (len(batch_indices), 1536):
                    raise RuntimeError(f"unexpected DINO feature shape: {result.shape}")
                features[np.asarray(batch_indices, dtype=np.int64)] = result.astype(
                    np.float16
                )
    if not np.isfinite(features).all():
        raise RuntimeError("DINO features contain NaN/Inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, dino_cls_patchmean=features)
    elapsed = time.monotonic() - started
    summary = {
        "status": "complete",
        "protocol": "pseudo10k_candidate_dinov2b_cls_patchmean_v1",
        "candidate_count": len(predictions),
        "feature_shape": list(features.shape),
        "storage_dtype": str(features.dtype),
        "context_ratio": args.context_ratio,
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "input_sha256": {"gt": _sha256(args.gt), "pred": _sha256(args.pred)},
        "teacher": adapter.metadata(),
        "elapsed_seconds": elapsed,
        "rows_per_second": len(predictions) / elapsed,
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
