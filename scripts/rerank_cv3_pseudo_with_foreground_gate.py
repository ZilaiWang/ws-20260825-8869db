#!/usr/bin/env python3
"""Apply fold-specific pseudo-10K foreground gates to COCO proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.background_gate import coarse_of_category_id, expand_context_bbox
from rsdet.data.crop_classification import render_crop
from rsdet.models.crop_classifier import sha256_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--imagenet-weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--context-ratio", type=float, default=1.25)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    import torch
    from torchvision import transforms

    from rsdet.models.background_gate_classifier import build_coarse_foreground_gate

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    predictions = json.loads(args.pred.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in gt["images"]}
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    device = torch.device(args.device)
    probabilities = [float("nan")] * len(predictions)
    fold_counts: dict[str, int] = {}
    checkpoint_hashes: dict[str, str] = {}
    started = time.monotonic()
    Image.MAX_IMAGE_PIXELS = None
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

    for fold in (0, 1, 2):
        checkpoint = args.checkpoint_dir / f"bg_gate_fold{fold}_final.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hashes[str(fold)] = sha256_file(checkpoint)
        model = build_coarse_foreground_gate(
            weight_path={
                "convnext": str(args.imagenet_weight),
                "checkpoint": str(checkpoint),
            },
            freeze="freeze_backbone",
            verify_weight_sha256=True,
            device=device,
        )
        model.eval()
        indices = [
            index
            for index, item in enumerate(predictions)
            if int(images[int(item["image_id"])] ["fold"]) == fold
        ]
        fold_counts[str(fold)] = len(indices)
        by_image: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            by_image[int(predictions[index]["image_id"])].append(index)
        with torch.inference_mode():
            for image_id, image_indices in sorted(by_image.items()):
                meta = images[image_id]
                path = args.pseudo_root / f"fold_{fold}" / "images" / meta["file_name"]
                with Image.open(path) as source:
                    source.load()
                    for batch_indices in _batches(image_indices, args.batch_size):
                        tensors = [
                            transform(
                                render_crop(
                                    source,
                                    expand_context_bbox(
                                        _xywh_to_xyxy(predictions[index]["bbox"]),
                                        ratio=args.context_ratio,
                                    ),
                                    args.resolution,
                                )
                            )
                            for index in batch_indices
                        ]
                        output = model(torch.stack(tensors).to(device, non_blocking=True))
                        shared = output.shared_logit.squeeze(1)
                        residual = output.coarse_logits
                        for row, index in enumerate(batch_indices):
                            coarse = coarse_of_category_id(
                                int(predictions[index]["category_id"])
                            )
                            coarse_index = ("ship", "aircraft", "vehicle").index(coarse)
                            logit = shared[row] + residual[row, coarse_index]
                            probabilities[index] = float(torch.sigmoid(logit).cpu())
        del model
        torch.cuda.empty_cache()

    if any(not math.isfinite(value) for value in probabilities):
        raise RuntimeError("not all predictions received a finite foreground probability")
    epsilon = 1e-8
    outputs: list[dict[str, Any]] = []
    for item, probability in zip(predictions, probabilities, strict=True):
        detector_score = float(item["score"])
        score = math.exp(
            (1.0 - args.alpha) * math.log(max(detector_score, epsilon))
            + args.alpha * math.log(max(probability, epsilon))
        )
        record = dict(item)
        record["score"] = score
        record["detector_score"] = detector_score
        record["foreground_probability"] = probability
        outputs.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outputs, ensure_ascii=False) + "\n", encoding="utf-8")
    payload = {
        "status": "complete",
        "protocol": "fold_specific_pseudo10k_coarse_foreground_gate_r4_v1",
        "input_predictions": len(predictions),
        "output_predictions": len(outputs),
        "alpha": args.alpha,
        "context_ratio": args.context_ratio,
        "fold_counts": fold_counts,
        "checkpoint_sha256": checkpoint_hashes,
        "imagenet_weight_sha256": sha256_file(args.imagenet_weight),
        "elapsed_seconds": time.monotonic() - started,
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
