#!/usr/bin/env python3
"""Restore fine labels on fold-heldout coarse-detector proposals with P03."""

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

from rsdet.data.crop_classification import render_crop
from rsdet.models.crop_classifier import build_convnext_tiny_classifier, sha256_file

PLACEHOLDER_TO_FINE = {0: tuple(range(0, 4)), 4: tuple(range(4, 24)), 24: (24,)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allowed_fine_ids(placeholder: int) -> tuple[int, ...]:
    try:
        return PLACEHOLDER_TO_FINE[int(placeholder)]
    except KeyError as error:
        raise ValueError(f"invalid coarse placeholder: {placeholder}") from error


def _square_window(bbox: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in bbox)
    if width <= 0.0 or height <= 0.0 or not all(
        math.isfinite(value) for value in (x, y, width, height)
    ):
        raise ValueError("invalid proposal bbox")
    side = max(width, height)
    cx, cy = x + width / 2.0, y + height / 2.0
    return cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0


def _checkpoint_paths(pattern: str) -> dict[int, Path]:
    if "{fold}" not in pattern:
        raise ValueError("checkpoint pattern must contain {fold}")
    paths = {fold: Path(pattern.format(fold=fold)).resolve() for fold in (0, 1, 2)}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return paths


def _batches(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-pattern", required=True)
    parser.add_argument("--imagenet-weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--detector-weight", type=float, default=0.60)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0.0 <= args.detector_weight <= 1.0:
        raise ValueError("detector weight must be in [0,1]")

    import torch
    from torchvision import transforms

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    predictions = json.loads(args.pred.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in gt["images"]}
    checkpoints = _checkpoint_paths(args.checkpoint_pattern)
    weight = args.imagenet_weight.resolve()
    if not weight.is_file():
        raise FileNotFoundError(weight)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))]
    )
    device = torch.device(args.device)
    rows: list[dict[str, Any] | None] = [None] * len(predictions)
    fold_counts: dict[str, int] = {}
    Image.MAX_IMAGE_PIXELS = None
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    started = time.monotonic()
    for fold in (0, 1, 2):
        checkpoint = torch.load(checkpoints[fold], map_location="cpu", weights_only=False)
        config = checkpoint.get("resolved_config", {})
        if int(config.get("fold", -1)) != fold or config.get("policy") != "tight":
            raise ValueError(f"fold {fold} P03 contract mismatch")
        model = build_convnext_tiny_classifier(25, weight_path=weight, regime="fine_tune")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        indices = [index for index, item in enumerate(predictions)
                   if int(images[int(item["image_id"])]["fold"]) == fold]
        fold_counts[str(fold)] = len(indices)
        by_image: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            by_image[int(predictions[index]["image_id"])].append(index)
        with torch.inference_mode():
            for image_id, image_indices in sorted(by_image.items()):
                meta = images[image_id]
                path = args.pseudo_root / f"fold_{fold}" / "images" / str(meta["file_name"])
                with Image.open(path) as source:
                    source.load()
                    for batch_indices in _batches(image_indices, args.batch_size):
                        tensors = [transform(render_crop(source,
                                                         _square_window(predictions[i]["bbox"]),
                                                         args.resolution))
                                   for i in batch_indices]
                        probabilities = torch.softmax(
                            model(torch.stack(tensors).to(device, non_blocking=True)), dim=1
                        ).cpu()
                        for row_index, prediction_index in enumerate(batch_indices):
                            original = predictions[prediction_index]
                            allowed = allowed_fine_ids(int(original["category_id"]))
                            selected = probabilities[row_index, list(allowed)]
                            selected = selected / selected.sum().clamp_min(1e-12)
                            order = torch.argsort(selected, descending=True)
                            fine_id = int(allowed[int(order[0])])
                            fine_probability = float(selected[order[0]])
                            margin = float(selected[order[0]] - selected[order[1]]) if len(order) > 1 else 1.0
                            detector_score = float(original["score"])
                            score = math.exp(
                                args.detector_weight * math.log(max(detector_score, 1e-8))
                                + (1.0 - args.detector_weight)
                                * math.log(max(fine_probability, 1e-8))
                            )
                            output = dict(original)
                            output.update(
                                category_id=fine_id,
                                score=score,
                                coarse_detector_score=detector_score,
                                crop_class_probability=fine_probability,
                                crop_margin=margin,
                                coarse_placeholder=int(original["category_id"]),
                                proposal_index=prediction_index,
                            )
                            rows[prediction_index] = output
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if any(item is None for item in rows):
        raise RuntimeError("not all proposals were classified")
    output_rows = [item for item in rows if item is not None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_rows, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "protocol": "fold_heldout_coarse_detector_to_p03_fine_v1",
        "predictions": len(output_rows),
        "fold_counts": fold_counts,
        "detector_weight": args.detector_weight,
        "runtime_seconds": time.monotonic() - started,
        "input_sha256": {"gt": _sha256(args.gt), "pred": _sha256(args.pred)},
        "output_sha256": _sha256(args.output),
        "checkpoints": {str(fold): sha256_file(path) for fold, path in checkpoints.items()},
        "imagenet_weight_sha256": sha256_file(weight),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print("COARSE_PROPOSAL_P03_CLASSIFICATION_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
