#!/usr/bin/env python3
"""Run one DEIM-family checkpoint through the frozen large-image tiling pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from rsdet.contracts import InferenceSample, Prediction  # noqa: E402
from rsdet.models.base import BaseDetector  # noqa: E402
from rsdet.pipeline.large_image import PipelineConfig, run_pipeline  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_image_path(image_root: Path, filename: str) -> Path:
    direct = image_root / filename
    if direct.is_file():
        return direct
    candidates = sorted(image_root.glob(f"fold_*/images/{filename}"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"cannot resolve exactly one image for {filename!r} below {image_root}: {candidates}"
        )
    return candidates[0]


class _DeployModel(nn.Module):
    def __init__(self, model: nn.Module, postprocessor: nn.Module) -> None:
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()

    def forward(self, images: torch.Tensor, sizes: torch.Tensor):
        return self.postprocessor(self.model(images), sizes)


class _DeimDetector(BaseDetector):
    def __init__(
        self,
        *,
        deim_root: Path,
        config_path: Path,
        checkpoint_path: Path,
        expected_epoch: int,
        image_size: int,
        score_floor: float,
        num_classes: int,
        device: str,
    ) -> None:
        sys.path.insert(0, str(deim_root))
        from engine.core import YAMLConfig

        cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        decoder_name = cfg.yaml_cfg.get("DEIM", {}).get("decoder")
        if decoder_name == "BHCLDFINETransformer":
            decoder = cfg.model.decoder
            initialize = getattr(decoder, "initialize_after_tuning", None)
            if initialize is None:
                raise RuntimeError(
                    "BHCLDFINETransformer requires initialize_after_tuning before "
                    "loading its decoupled checkpoint"
                )
            initialize()
            if not getattr(decoder.decoder, "decoupled_ready", False):
                raise RuntimeError("BHCL decoder materialization did not complete")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        observed_epoch = int(checkpoint.get("last_epoch", -1))
        if observed_epoch != expected_epoch:
            raise RuntimeError(
                f"checkpoint epoch mismatch: expected={expected_epoch}, observed={observed_epoch}"
            )
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state)
        self.model = _DeployModel(cfg.model, cfg.postprocessor)
        self.image_size = image_size
        self.score_floor = score_floor
        self.num_classes = num_classes
        self.device = device

    def load(self, checkpoint_path: str) -> None:
        del checkpoint_path

    def to(self, device: str) -> None:
        self.device = device
        self.model.to(device)

    def eval(self) -> None:
        self.model.eval()

    def predict(self, batch: Sequence[InferenceSample]) -> list[Prediction]:
        if not batch:
            return []
        tensors: list[torch.Tensor] = []
        sizes: list[list[int]] = []
        for sample in batch:
            image = np.asarray(sample.image, dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"expected RGB HWC image, got {image.shape}")
            resized = tvf.resize(Image.fromarray(image), [self.image_size, self.image_size])
            tensors.append(tvf.to_tensor(resized))
            sizes.append([sample.width, sample.height])
        tensor = torch.stack(tensors).to(self.device, non_blocking=True)
        original_sizes = torch.tensor(sizes, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            labels, boxes, scores = self.model(tensor, original_sizes)
        outputs: list[Prediction] = []
        for sample, item_labels, item_boxes, item_scores in zip(
            batch, labels, boxes, scores, strict=True
        ):
            keep = item_scores >= self.score_floor
            kept_boxes: list[list[float]] = []
            kept_scores: list[float] = []
            kept_labels: list[int] = []
            for box, score, label in zip(
                item_boxes[keep].detach().cpu().float().tolist(),
                item_scores[keep].detach().cpu().float().tolist(),
                item_labels[keep].detach().cpu().long().tolist(),
                strict=True,
            ):
                x0, y0, x1, y1 = (float(value) for value in box)
                score = float(score)
                label = int(label)
                if not all(math.isfinite(value) for value in (x0, y0, x1, y1, score)):
                    raise RuntimeError("non-finite output produced by DEIM")
                if not 0 <= label < self.num_classes:
                    raise RuntimeError(f"prediction category outside taxonomy: {label}")
                x0 = min(max(x0, 0.0), float(sample.width))
                y0 = min(max(y0, 0.0), float(sample.height))
                x1 = min(max(x1, 0.0), float(sample.width))
                y1 = min(max(y1, 0.0), float(sample.height))
                if x1 <= x0 or y1 <= y0:
                    continue
                kept_boxes.append([x0, y0, x1, y1])
                kept_scores.append(score)
                kept_labels.append(label)
            outputs.append(
                Prediction(
                    image_id=sample.image_id,
                    boxes_xyxy=kept_boxes,
                    scores=kept_scores,
                    labels=kept_labels,
                )
            )
        return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deim-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-epoch", type=int, default=39)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--num-classes", type=int, default=25)
    parser.add_argument("--fine-nms-iou", type=float, default=0.70)
    parser.add_argument("--coarse-nms-iou", type=float, default=0.85)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    ledger = json.loads(args.coco.read_text(encoding="utf-8"))
    images = sorted(ledger["images"], key=lambda item: int(item["id"]))
    if not images:
        raise ValueError("COCO ledger has no images")
    detector = _DeimDetector(
        deim_root=args.deim_root,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        expected_epoch=args.expected_checkpoint_epoch,
        image_size=args.imgsz,
        score_floor=args.score_floor,
        num_classes=args.num_classes,
        device=args.device,
    )
    detector.to(args.device)
    detector.eval()
    pipeline = PipelineConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        fusion="safe",
        score_threshold=args.score_floor,
        fine_nms_iou=args.fine_nms_iou,
        coarse_nms_iou=args.coarse_nms_iou,
        merge_iou=0.50,
        merge_ios=0.75,
        border_margin=8.0,
        max_detections=2000,
    )

    predictions: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    started = time.perf_counter()
    for item in images:
        image_path = resolve_image_path(args.image_root, str(item["file_name"]))
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        fused, timing = run_pipeline(
            rgb,
            detector,
            config=pipeline,
            parent_image_id=int(item["id"]),
        )
        for box, score, label in zip(
            fused.boxes_xyxy, fused.scores, fused.labels, strict=True
        ):
            x0, y0, x1, y1 = (float(value) for value in box)
            predictions.append(
                {
                    "image_id": int(item["id"]),
                    "category_id": int(label),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": float(score),
                }
            )
        timings.append({"image_id": int(item["id"]), **timing.to_dict()})
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "protocol": "deim_safe_tiled_large_image_candidate_export_v1",
        "images": len(images),
        "predictions": len(predictions),
        "elapsed_seconds": elapsed,
        "average_seconds_per_image": elapsed / len(images),
        "pipeline": pipeline.__dict__,
        "timings": timings,
        "input_sha256": {
            "config": _sha256(args.config),
            "checkpoint": _sha256(args.checkpoint),
            "coco": _sha256(args.coco),
        },
        "output_sha256": _sha256(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
