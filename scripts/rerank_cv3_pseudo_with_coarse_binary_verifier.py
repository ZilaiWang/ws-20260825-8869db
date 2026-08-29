#!/usr/bin/env python3
"""Apply fold-heldout coarse-specific foreground/background crop verifiers."""

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
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.models.crop_classifier import build_convnext_tiny_classifier, sha256_file
from rsdet.utils.config import load_config

COARSE_CLASSES = ("ship", "aircraft", "vehicle")


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
    parser.add_argument("--context-ratio", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    import torch
    from torchvision import transforms

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    predictions = json.loads(args.pred.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in gt["images"]}
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    device = torch.device(args.device)
    probabilities = np.full(len(predictions), np.nan, dtype=np.float64)
    checkpoint_hashes: dict[str, str] = {}
    group_counts: dict[str, int] = {}
    started = time.monotonic()
    Image.MAX_IMAGE_PIXELS = None
    for fold in (0, 1, 2):
        for coarse in COARSE_CLASSES:
            checkpoint = args.checkpoint_dir / f"coarse_{coarse}_fold{fold}.pt"
            checkpoint_hashes[f"{coarse}_fold{fold}"] = sha256_file(checkpoint)
            model = build_convnext_tiny_classifier(
                2, weight_path=args.imagenet_weight, regime="fine_tune"
            )
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if state["coarse"] != coarse or int(state["held_out_fold"]) != fold:
                raise ValueError(f"checkpoint contract mismatch: {checkpoint}")
            model.load_state_dict(state["model_state_dict"], strict=True)
            model.to(device).eval()
            indices = [
                index
                for index, item in enumerate(predictions)
                if int(images[int(item["image_id"])]["fold"]) == fold
                and protocol.category_mapping[int(item["category_id"])] == coarse
            ]
            group_counts[f"{coarse}_fold{fold}"] = len(indices)
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
                            values = torch.softmax(
                                model(torch.stack(tensors).to(device, non_blocking=True)), dim=1
                            )[:, 1].cpu().numpy()
                            probabilities[np.asarray(batch_indices, dtype=np.int64)] = values
            del model
            torch.cuda.empty_cache()
    if not np.isfinite(probabilities).all():
        raise RuntimeError("not all proposals received coarse foreground probabilities")

    outputs: list[dict[str, Any]] = []
    epsilon = 1e-10
    for item, probability in zip(predictions, probabilities, strict=True):
        row = dict(item)
        base_score = float(item.get("detector_score", item["score"]))
        row["coarse_foreground_probability"] = float(probability)
        row["score"] = math.exp(
            (1.0 - args.alpha) * math.log(max(base_score, epsilon))
            + args.alpha * math.log(max(float(probability), epsilon))
        )
        outputs.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outputs, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "protocol": "fold_heldout_coarse_binary_verifier_inference_v1",
        "input_predictions": len(predictions),
        "output_predictions": len(outputs),
        "alpha": args.alpha,
        "context_ratio": args.context_ratio,
        "group_counts": group_counts,
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
