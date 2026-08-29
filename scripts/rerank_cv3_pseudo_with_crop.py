#!/usr/bin/env python3
"""Apply the frozen P03 crop classifiers to formal-CV3 pseudo-10K proposals.

Each proposal is scored only by the crop classifier from the same held-out
fold.  The detector category is never changed.  The frozen E1 rule is used:

    fused = detector_score ** (1 - alpha) * p(category | crop) ** alpha

The script also writes the frozen E2 aircraft-only, same-fine-class NMS@0.5
variant.  It intentionally does not choose a score threshold; threshold
selection remains the responsibility of the cross-fit frontier evaluator.
"""

from __future__ import annotations

import argparse
import csv
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

AIRCRAFT_IDS = frozenset(range(4, 24))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _square_window(bbox_xywh: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in bbox_xywh)
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("bbox contains a non-finite value")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"bbox must have positive extent: {(x, y, width, height)}")
    side = max(width, height)
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    return (
        center_x - side / 2.0,
        center_y - side / 2.0,
        center_x + side / 2.0,
        center_y + side / 2.0,
    )


def _iou_xywh(first: list[float], second: list[float]) -> float:
    ax0, ay0, aw, ah = (float(value) for value in first)
    bx0, by0, bw, bh = (float(value) for value in second)
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    height = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = width * height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def aircraft_same_class_nms(
    predictions: list[dict[str, Any]], iou_threshold: float = 0.5
) -> list[dict[str, Any]]:
    """Frozen E2: NMS only aircraft, and only within the same fine class."""

    grouped: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    passthrough: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(predictions):
        category_id = int(item["category_id"])
        if category_id in AIRCRAFT_IDS:
            grouped[(int(item["image_id"]), category_id)].append((index, item))
        else:
            passthrough.append((index, item))

    kept: list[tuple[int, dict[str, Any]]] = list(passthrough)
    for values in grouped.values():
        ordered = sorted(values, key=lambda pair: (-float(pair[1]["score"]), pair[0]))
        selected: list[tuple[int, dict[str, Any]]] = []
        for candidate in ordered:
            if all(
                _iou_xywh(candidate[1]["bbox"], previous[1]["bbox"]) <= iou_threshold
                for previous in selected
            ):
                selected.append(candidate)
        kept.extend(selected)
    return [item for _, item in sorted(kept, key=lambda pair: pair[0])]


def _batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _checkpoint_paths(pattern: str) -> dict[int, Path]:
    paths = {fold: Path(pattern.format(fold=fold)).expanduser().resolve() for fold in (0, 1, 2)}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing P03 checkpoints: {missing}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-pattern", required=True, help="must contain {fold}")
    parser.add_argument("--imagenet-weight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.40)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nms-iou", type=float, default=0.50)
    args = parser.parse_args()

    if "{fold}" not in args.checkpoint_pattern:
        raise ValueError("--checkpoint-pattern must contain {fold}")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if args.resolution != 224:
        raise ValueError("formal P03 checkpoints require resolution=224")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    import torch
    from torchvision import transforms

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    predictions = json.loads(args.pred.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("--pred must be a COCO detection list")
    images = {int(item["id"]): item for item in gt["images"]}
    if set(int(item["fold"]) for item in images.values()) != {0, 1, 2}:
        raise ValueError("ground truth must contain folds 0, 1 and 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = _checkpoint_paths(args.checkpoint_pattern)
    weight = args.imagenet_weight.expanduser().resolve()
    if not weight.is_file():
        raise FileNotFoundError(weight)
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
    crop_top1 = [float("nan")] * len(predictions)
    crop_margin = [float("nan")] * len(predictions)
    crop_entropy = [float("nan")] * len(predictions)
    crop_top1_class = [-1] * len(predictions)
    fold_counts: dict[str, int] = {}
    started = time.monotonic()
    Image.MAX_IMAGE_PIXELS = None
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

    for fold in (0, 1, 2):
        checkpoint = torch.load(checkpoints[fold], map_location="cpu", weights_only=False)
        config = checkpoint.get("resolved_config", {})
        if (
            int(config.get("fold", -1)) != fold
            or config.get("policy") != "tight"
            or int(config.get("resolution", -1)) != args.resolution
            or config.get("regime") != "fine_tune"
        ):
            raise ValueError(f"fold {fold} checkpoint contract mismatch: {config}")
        model = build_convnext_tiny_classifier(
            25, weight_path=weight, regime="fine_tune"
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
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
                path = (
                    args.pseudo_root
                    / f"fold_{fold}"
                    / "images"
                    / str(meta["file_name"])
                ).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                with Image.open(path) as source:
                    source.load()
                    for batch_indices in _batches(image_indices, args.batch_size):
                        tensors = [
                            transform(
                                render_crop(
                                    source,
                                    _square_window(predictions[index]["bbox"]),
                                    args.resolution,
                                )
                            )
                            for index in batch_indices
                        ]
                        logits = model(torch.stack(tensors).to(device, non_blocking=True))
                        batch_probabilities = torch.softmax(logits, dim=1).cpu()
                        for row, index in enumerate(batch_indices):
                            category_id = int(predictions[index]["category_id"])
                            values = batch_probabilities[row]
                            ordered = torch.argsort(values, descending=True)
                            probabilities[index] = float(values[category_id])
                            crop_top1[index] = float(values[ordered[0]])
                            crop_margin[index] = float(values[ordered[0]] - values[ordered[1]])
                            crop_entropy[index] = float(
                                -(values * torch.log(values.clamp_min(1e-9))).sum()
                            )
                            crop_top1_class[index] = int(ordered[0])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if any(
        not math.isfinite(value)
        for values in (probabilities, crop_top1, crop_margin, crop_entropy)
        for value in values
    ) or any(value < 0 for value in crop_top1_class):
        raise RuntimeError("not all predictions received finite crop evidence")

    fused: list[dict[str, Any]] = []
    epsilon = 1e-8
    for index, (item, crop_probability) in enumerate(zip(predictions, probabilities, strict=True)):
        detector_score = float(item["score"])
        score = math.exp(
            (1.0 - args.alpha) * math.log(max(detector_score, epsilon))
            + args.alpha * math.log(max(crop_probability, epsilon))
        )
        output = dict(item)
        output["score"] = score
        output["detector_score"] = detector_score
        output["crop_class_probability"] = crop_probability
        output["crop_top1"] = crop_top1[index]
        output["crop_margin"] = crop_margin[index]
        output["crop_entropy"] = crop_entropy[index]
        output["crop_top1_class"] = crop_top1_class[index]
        output["detector_crop_agree"] = int(
            crop_top1_class[index] == int(item["category_id"])
        )
        output["proposal_index"] = index
        fused.append(output)
    nms = aircraft_same_class_nms(fused, args.nms_iou)

    fused_path = args.output_dir / "predictions_r3_fused.json"
    nms_path = args.output_dir / "predictions_r3_fused_aircraft_nms.json"
    fused_path.write_text(json.dumps(fused, ensure_ascii=False) + "\n", encoding="utf-8")
    nms_path.write_text(json.dumps(nms, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output_dir / "proposal_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "proposal_index",
                "image_id",
                "fold",
                "category_id",
                "detector_score",
                "crop_class_probability",
                "crop_top1",
                "crop_margin",
                "crop_entropy",
                "crop_top1_class",
                "detector_crop_agree",
                "fused_score",
            ]
        )
        for item in fused:
            image_id = int(item["image_id"])
            writer.writerow(
                [
                    item["proposal_index"],
                    image_id,
                    int(images[image_id]["fold"]),
                    item["category_id"],
                    item["detector_score"],
                    item["crop_class_probability"],
                    item["crop_top1"],
                    item["crop_margin"],
                    item["crop_entropy"],
                    item["crop_top1_class"],
                    item["detector_crop_agree"],
                    item["score"],
                ]
            )
    summary = {
        "status": "complete",
        "protocol": "formal_cv3_heldout_p03_r3_e2_pseudo10k_v1",
        "alpha": args.alpha,
        "nms": {
            "scope": "aircraft_same_fine_class_only",
            "iou_threshold": args.nms_iou,
        },
        "counts": {
            "input": len(predictions),
            "fused": len(fused),
            "after_nms": len(nms),
            "removed_by_nms": len(fused) - len(nms),
            "by_fold": fold_counts,
        },
        "elapsed_seconds": time.monotonic() - started,
        "assets": {
            "ground_truth_sha256": _sha256(args.gt),
            "input_predictions_sha256": _sha256(args.pred),
            "imagenet_weight_sha256": sha256_file(weight),
            "checkpoints": {str(fold): sha256_file(path) for fold, path in checkpoints.items()},
        },
        "outputs": {
            "fused": {"path": str(fused_path), "sha256": _sha256(fused_path)},
            "fused_aircraft_nms": {"path": str(nms_path), "sha256": _sha256(nms_path)},
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
