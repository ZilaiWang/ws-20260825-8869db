#!/usr/bin/env python3
"""Apply an in-model agreement branch to existing adapted-YOLO proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _letterbox_tensor(image: Image.Image, target_size: int, device: str):
    import torch

    from rsdet.innovation.in_model_agreement import letterbox_geometry

    image = image.convert("RGB")
    geometry = letterbox_geometry(image.width, image.height, target_size)
    resized = image.resize(
        (geometry.resized_width, geometry.resized_height), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    canvas.paste(
        resized,
        (
            int(round(geometry.pad_left - 0.1)),
            int(round(geometry.pad_top - 0.1)),
        ),
    )
    array = np.asarray(canvas, dtype=np.float32).copy().transpose(2, 0, 1) / 255.0
    return torch.from_numpy(array).unsqueeze(0).to(device), geometry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--adapted-detector", type=Path, required=True)
    parser.add_argument("--expected-detector-sha256", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--category-id", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if _sha256(args.adapted_detector) != args.expected_detector_sha256.lower():
        raise ValueError("adapted detector SHA mismatch")
    if _sha256(args.adapter) != args.expected_adapter_sha256.lower():
        raise ValueError("agreement adapter SHA mismatch")

    import torch

    from rsdet.innovation.in_model_agreement import (
        AgreementResidualHead,
        apply_logit_residual,
        boxes_to_letterbox,
        proposal_metadata,
    )
    from rsdet.innovation.yolo_feature_quality import (
        MultiScaleROIFeatureEncoder,
        YoloPyramidTap,
    )

    try:
        state = torch.load(args.adapter, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(args.adapter, map_location="cpu")
    from ultralytics import YOLO

    device = torch.device(args.device)
    model = YOLO(str(args.adapted_detector)).model.to(device).eval()
    indices = tuple(int(value) for value in state["layer_indices"])
    detect_indices = tuple(int(value) for value in model.model[-1].f)
    if indices != detect_indices:
        raise ValueError("adapter feature layers do not match detector architecture")
    roi_encoder = MultiScaleROIFeatureEncoder(
        state["channels"],
        state["strides"],
        projection_dim=int(state["projection_dim"]),
        output_size=3,
        context_ratio=float(state["context_ratio"]),
    ).to(device)
    quality_head = AgreementResidualHead(
        roi_encoder.output_dim,
        hidden_dim=int(state["hidden_dim"]),
    ).to(device)
    roi_encoder.load_state_dict(state["roi_encoder_state_dict"])
    quality_head.load_state_dict(state["quality_head_state_dict"])
    roi_encoder.eval()
    quality_head.eval()

    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("predictions must be a COCO result list")
    by_image: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if int(row["category_id"]) == args.category_id:
            by_image[int(row["image_id"])].append(index)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    image_paths = {
        int(row["image_id"]): (args.data_root / row["relative_path"]).resolve()
        for row in manifest["samples"]
    }
    unknown = sorted(set(by_image) - set(image_paths))
    if unknown:
        raise ValueError(f"prediction image IDs missing from split manifest: {unknown[:10]}")
    deltas = []
    tap = YoloPyramidTap(model, indices)
    with torch.inference_mode():
        for image_id, row_indices in sorted(by_image.items()):
            image = Image.open(image_paths[image_id]).convert("RGB")
            tensor, geometry = _letterbox_tensor(image, int(state["imgsz"]), str(device))
            tap.clear()
            model(tensor)
            features = tap.features(detach=True)
            boxes = []
            scores = []
            for index in row_indices:
                x, y, width, height = (float(value) for value in rows[index]["bbox"])
                boxes.append([x, y, x + width, y + height])
                scores.append(float(rows[index]["score"]))
            box_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
            model_boxes = boxes_to_letterbox(box_tensor, geometry)
            boxes_with_batch = torch.cat(
                (torch.zeros((len(boxes), 1), device=device), model_boxes), dim=1
            )
            score_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
            roi = roi_encoder(
                features,
                boxes_with_batch,
                image_height=int(state["imgsz"]),
                image_width=int(state["imgsz"]),
            )
            metadata = proposal_metadata(
                score_tensor, model_boxes, image_size=int(state["imgsz"])
            )
            residual = quality_head(roi, metadata)
            rescored = apply_logit_residual(
                score_tensor,
                residual,
                alpha=float(state["residual_alpha"]),
            ).cpu()
            for index, score in zip(row_indices, rescored.tolist(), strict=True):
                old = float(rows[index]["score"])
                rows[index]["score"] = float(score)
                deltas.append(float(score) - old)
    tap.close()
    if not all(np.isfinite(delta) for delta in deltas):
        raise RuntimeError("agreement branch produced NaN/Inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    summary = {
        "status": "complete",
        "protocol": "apply_in_model_dfine_agreement_to_adapted_yolo_proposals_v1",
        "prediction_rows": len(rows),
        "rescored_rows": len(deltas),
        "rescored_images": len(by_image),
        "mean_score_delta": float(np.mean(deltas)) if deltas else 0.0,
        "minimum_score_delta": float(np.min(deltas)) if deltas else 0.0,
        "maximum_score_delta": float(np.max(deltas)) if deltas else 0.0,
        "input_prediction_sha256": _sha256(args.predictions),
        "adapted_detector_sha256": _sha256(args.adapted_detector),
        "adapter_sha256": _sha256(args.adapter),
        "output_sha256": _sha256(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
