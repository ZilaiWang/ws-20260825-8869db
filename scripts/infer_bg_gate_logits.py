#!/usr/bin/env python3
"""用 N2-CFG 模型对 R1-6 候选渲染 context crop，输出前景 logit。

输出 JSON：``{proposal_uid: {"shared": float, "ship": float, "aircraft": float, "vehicle": float}}``，
其中粗类条件 logit = shared + coarse residual。S1 用 ``shared``，S2 用对应
粗类条件 logit。

按 fold 使用对应 checkpoint（fold-specific，无跨折泄漏）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsdet.analysis.background_gate import (
    COARSE_CLASSES,
    INPUT_RESOLUTION,
    expand_context_bbox,
)
from rsdet.analysis.background_gate_manifest import (
    _load_image_ledger,
    _load_selected_predictions,
)
from rsdet.analysis.proposal_crops import load_image_path_mapping
from rsdet.data.crop_classification import render_crop


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--predictions", type=Path, required=True)
    value.add_argument("--formal-crop-manifest", type=Path, required=True)
    value.add_argument("--image-ledger", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--convnext-weights", type=Path, required=True)
    value.add_argument("--checkpoint-dir", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--batch-size", type=int, default=256)
    value.add_argument("--resolution", type=int, default=INPUT_RESOLUTION)
    value.add_argument("--device", default="cuda")
    value.add_argument("--freeze", default="freeze_backbone")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    import numpy as np
    import torch
    from PIL import Image

    from rsdet.models.background_gate_classifier import build_coarse_foreground_gate

    predictions, flat = _load_selected_predictions(args.predictions)
    ledger = _load_image_ledger(args.image_ledger)
    path_map = load_image_path_mapping(args.formal_crop_manifest)
    data_root = args.data_root.expanduser().resolve()
    device = torch.device(args.device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    # 按 fold 加载对应 checkpoint。
    models: dict[int, Any] = {}
    for fold in (0, 1, 2):
        checkpoint = args.checkpoint_dir / f"bg_gate_fold{fold}_final.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"缺少 fold{fold} checkpoint: {checkpoint}")
        model = build_coarse_foreground_gate(
            weight_path=str(args.convnext_weights),
            freeze=args.freeze,
            verify_weight_sha256=False,
            device=device,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        models[fold] = model

    result: dict[str, dict[str, float]] = {}
    image_cache: dict[Path, Image.Image] = {}
    batches: list[tuple[list[Mapping[str, Any]], int]] = []
    current: list[Mapping[str, Any]] = []
    current_fold: int | None = None

    def flush() -> None:
        nonlocal current, current_fold
        if not current:
            return
        batches.append((current, current_fold))
        current = []
        current_fold = None

    # 按 image_id 遍历候选，保持 fold 分组（同 fold 候选可批量，跨 fold 用不同模型）。
    for image_id, records in predictions.items():
        fold = ledger[image_id]["fold"]
        for prediction_index, record in enumerate(records):
            if current_fold is not None and fold != current_fold:
                flush()
            proposal_uid = str(record.get("proposal_uid") or f"r1-i{image_id}-p{prediction_index}")
            row = {
                "proposal_uid": proposal_uid,
                "image_id": image_id,
                "fold": fold,
                "category_id": int(record["category_id"]),
                "bbox_xyxy": [float(v) for v in record["bbox_xyxy"]],
                "source_relative_path": path_map[image_id]["source_relative_path"],
            }
            current.append(row)
            current_fold = fold
    flush()

    for rows, fold in batches:
        model = models[fold]
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            tensors: list[Any] = []
            for row in chunk:
                image_path = (data_root / row["source_relative_path"]).resolve()
                if image_path not in image_cache:
                    image_cache[image_path] = Image.open(image_path)
                context = expand_context_bbox(row["bbox_xyxy"], ratio=1.25)
                crop = render_crop(image_cache[image_path], context, args.resolution)
                array = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)
                tensors.append(torch.from_numpy(array))
            x = torch.stack(tensors).to(device)
            x = (x / 255.0 - mean) / std
            with torch.no_grad():
                out = model(x)
            shared = out.shared_logit.squeeze(1).cpu().tolist()
            coarse = out.coarse_logits.cpu().tolist()
            for row, s, c in zip(chunk, shared, coarse, strict=True):
                result[row["proposal_uid"]] = {
                    "shared": float(s),
                    **{
                        name: float(s + c[index])
                        for index, name in enumerate(COARSE_CLASSES)
                    },
                }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"BG_GATE_LOGITS_PASS proposals={len(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
