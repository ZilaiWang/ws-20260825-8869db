#!/usr/bin/env python3
"""Run frozen D-FINE inference on a COCO image ledger and export candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _DeployModel(nn.Module):
    def __init__(self, model: nn.Module, postprocessor: nn.Module) -> None:
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()

    def forward(
        self, images: torch.Tensor, original_sizes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.postprocessor(self.model(images), original_sizes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfine-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--num-classes", type=int, default=25)
    parser.add_argument("--expected-checkpoint-epoch", type=int, default=39)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.imgsz <= 0:
        raise ValueError("batch-size and imgsz must be positive")
    if not 0.0 <= args.score_floor <= 1.0:
        raise ValueError("score-floor must be within [0, 1]")

    sys.path.insert(0, str(args.dfine_root))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config), resume=str(args.checkpoint))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if int(checkpoint.get("last_epoch", -1)) != args.expected_checkpoint_epoch:
        raise RuntimeError(
            "checkpoint is not the frozen final epoch: "
            f"observed={checkpoint.get('last_epoch')!r}, "
            f"expected={args.expected_checkpoint_epoch}"
        )
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)
    device = torch.device(args.device)
    model = _DeployModel(cfg.model, cfg.postprocessor).to(device).eval()

    ledger = json.loads(args.coco.read_text(encoding="utf-8"))
    images = sorted(ledger["images"], key=lambda item: int(item["id"]))
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    peak_memory = 0
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            batch = images[start : start + args.batch_size]
            tensors = []
            sizes = []
            for item in batch:
                with Image.open(args.image_root / item["file_name"]) as image:
                    rgb = image.convert("RGB")
                    width, height = rgb.size
                    resized = tvf.resize(rgb, [args.imgsz, args.imgsz])
                    tensors.append(tvf.to_tensor(resized))
                    sizes.append([width, height])
            tensor = torch.stack(tensors).to(device, non_blocking=True)
            original_sizes = torch.tensor(sizes, dtype=torch.float32, device=device)
            labels, boxes, scores = model(tensor, original_sizes)
            for item, item_labels, item_boxes, item_scores in zip(
                batch, labels, boxes, scores, strict=True
            ):
                keep = item_scores >= args.score_floor
                for label, box, score in zip(
                    item_labels[keep].cpu().tolist(),
                    item_boxes[keep].cpu().tolist(),
                    item_scores[keep].cpu().tolist(),
                    strict=True,
                ):
                    label = int(label)
                    score = float(score)
                    x0, y0, x1, y1 = (float(value) for value in box)
                    if not 0 <= label < args.num_classes:
                        raise RuntimeError(f"prediction category outside taxonomy: {label}")
                    if not all(math.isfinite(value) for value in (score, x0, y0, x1, y1)):
                        raise RuntimeError("non-finite detector output")
                    width = float(item["width"])
                    height = float(item["height"])
                    x0 = min(max(x0, 0.0), width)
                    y0 = min(max(y0, 0.0), height)
                    x1 = min(max(x1, 0.0), width)
                    y1 = min(max(y1, 0.0), height)
                    if x1 <= x0 or y1 <= y0:
                        raise RuntimeError("non-positive detector box after clipping")
                    predictions.append(
                        {
                            "image_id": int(item["id"]),
                            "category_id": label,
                            "bbox": [x0, y0, x1 - x0, y1 - y0],
                            "score": score,
                        }
                    )
            if device.type == "cuda":
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
    elapsed = time.perf_counter() - started
    if not predictions:
        raise RuntimeError("detector exported no predictions at the frozen score floor")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "protocol": "dfine_fixed_checkpoint_coco_candidate_export_v1",
        "images": len(images),
        "predictions": len(predictions),
        "score_floor": args.score_floor,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "checkpoint_epoch": int(checkpoint["last_epoch"]),
        "num_classes": args.num_classes,
        "elapsed_seconds": elapsed,
        "images_per_second": len(images) / elapsed,
        "peak_cuda_bytes": peak_memory,
        "input_sha256": {
            "config": _sha256(args.config),
            "checkpoint": _sha256(args.checkpoint),
            "coco": _sha256(args.coco),
        },
        "output_sha256": _sha256(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
