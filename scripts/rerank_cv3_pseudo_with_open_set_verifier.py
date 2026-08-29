#!/usr/bin/env python3
"""Apply fold-specific 25-class-plus-background proposal verifiers."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.background_gate import expand_context_bbox
from rsdet.data.crop_classification import render_crop
from rsdet.models.crop_classifier import build_convnext_tiny_classifier, sha256_file

BACKGROUND_CLASS_ID = 25


def foreground_statistics(probabilities: np.ndarray, predicted_class: int) -> dict[str, float | int]:
    if probabilities.shape != (26,) or not np.isfinite(probabilities).all():
        raise ValueError("open-set probabilities must be a finite 26-vector")
    if not 0 <= predicted_class < 25:
        raise ValueError("predicted_class must be in [0, 24]")
    foreground_probability = float(1.0 - probabilities[BACKGROUND_CLASS_ID])
    conditional = probabilities[:25] / max(foreground_probability, 1e-12)
    conditional = conditional / max(float(conditional.sum()), 1e-12)
    top = np.argsort(-conditional, kind="stable")
    top1_class = int(top[0])
    top1 = float(conditional[top1_class])
    top2 = float(conditional[int(top[1])])
    entropy = float(-(conditional * np.log(np.clip(conditional, 1e-12, 1.0))).sum())
    return {
        "foreground_probability": foreground_probability,
        "predicted_class_probability": float(probabilities[predicted_class]),
        "conditional_predicted_class_probability": float(conditional[predicted_class]),
        "top1_class": top1_class,
        "top1_probability": top1,
        "margin": top1 - top2,
        "entropy": entropy,
        "agree": int(top1_class == predicted_class),
    }


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
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    import torch
    from torchvision import transforms

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
    statistics: list[dict[str, float | int] | None] = [None] * len(predictions)
    fold_counts: dict[str, int] = {}
    checkpoint_hashes: dict[str, str] = {}
    started = time.monotonic()
    Image.MAX_IMAGE_PIXELS = None
    for fold in (0, 1, 2):
        checkpoint = args.checkpoint_dir / f"open_set_fold{fold}.pt"
        checkpoint_hashes[str(fold)] = sha256_file(checkpoint)
        model = build_convnext_tiny_classifier(
            26, weight_path=args.imagenet_weight, regime="fine_tune"
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.to(device).eval()
        indices = [
            index
            for index, item in enumerate(predictions)
            if int(images[int(item["image_id"])]["fold"]) == fold
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
                        probabilities = torch.softmax(
                            model(torch.stack(tensors).to(device, non_blocking=True)), dim=1
                        ).cpu().numpy()
                        for row, index in enumerate(batch_indices):
                            statistics[index] = foreground_statistics(
                                probabilities[row], int(predictions[index]["category_id"])
                            )
        del model
        torch.cuda.empty_cache()
    if any(item is None for item in statistics):
        raise RuntimeError("not all proposals received open-set statistics")

    outputs: list[dict[str, Any]] = []
    epsilon = 1e-10
    for prediction, values in zip(predictions, statistics, strict=True):
        assert values is not None
        detector_score = float(prediction["score"])
        verifier_score = float(values["predicted_class_probability"])
        score = math.exp(
            (1.0 - args.alpha) * math.log(max(detector_score, epsilon))
            + args.alpha * math.log(max(verifier_score, epsilon))
        )
        record = dict(prediction)
        record.update(
            {
                "score": score,
                "detector_score": detector_score,
                "foreground_probability": float(values["foreground_probability"]),
                "crop_class_probability": verifier_score,
                "crop_conditional_class_probability": float(
                    values["conditional_predicted_class_probability"]
                ),
                "crop_top1_class": int(values["top1_class"]),
                "crop_top1": float(values["top1_probability"]),
                "crop_top1_absolute": float(values["top1_probability"])
                * float(values["foreground_probability"]),
                "crop_margin": float(values["margin"]),
                "crop_entropy": float(values["entropy"]),
                "detector_crop_agree": int(values["agree"]),
            }
        )
        outputs.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outputs, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "protocol": "fold_heldout_pseudo10k_open_set_verifier_inference_v1",
        "input_predictions": len(predictions),
        "output_predictions": len(outputs),
        "alpha": args.alpha,
        "context_ratio": args.context_ratio,
        "fold_counts": fold_counts,
        "checkpoint_sha256": checkpoint_hashes,
        "imagenet_weight_sha256": sha256_file(args.imagenet_weight),
        "elapsed_seconds": time.monotonic() - started,
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
