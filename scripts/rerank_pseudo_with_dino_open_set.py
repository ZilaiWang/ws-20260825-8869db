#!/usr/bin/env python3
"""Apply fold-heldout DINOv2 open-set heads with one frozen score formula."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BACKGROUND_CLASS_ID = 25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    import torch

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    predictions = json.loads(args.pred.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in gt["images"]}
    with np.load(args.features) as archive:
        features = np.asarray(archive["dino_cls_patchmean"], dtype=np.float32)
    if features.shape != (len(predictions), 1536) or not np.isfinite(features).all():
        raise ValueError("feature/prediction contract mismatch")
    device = torch.device(args.device)
    all_probabilities = np.empty((len(predictions), 26), dtype=np.float32)
    checkpoint_hashes: dict[str, str] = {}
    for fold in (0, 1, 2):
        checkpoint_path = args.checkpoint_dir / f"dino_open_set_fold{fold}.pt"
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(state["held_out_fold"]) != fold:
            raise ValueError("held-out checkpoint mismatch")
        model = torch.nn.Linear(1536, 26)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.to(device).eval()
        train_rms = float(state["train_rms"])
        indices = np.asarray(
            [
                index
                for index, item in enumerate(predictions)
                if int(images[int(item["image_id"])]["fold"]) == fold
            ],
            dtype=np.int64,
        )
        with torch.inference_mode():
            for start in range(0, len(indices), args.batch_size):
                batch = indices[start : start + args.batch_size]
                x = torch.from_numpy(features[batch] / np.float32(train_rms)).to(device)
                all_probabilities[batch] = torch.softmax(model(x), dim=1).cpu().numpy()
        checkpoint_hashes[str(fold)] = _sha256(checkpoint_path)
        del model
    if not np.isfinite(all_probabilities).all():
        raise RuntimeError("nonfinite DINO probabilities")

    epsilon = 1e-10
    outputs: list[dict[str, object]] = []
    for prediction, probabilities in zip(predictions, all_probabilities, strict=True):
        category_id = int(prediction["category_id"])
        # Existing verifier outputs retain the raw detector confidence here.
        # Prefer it so DINO is compared/fused directly, rather than stacked on
        # top of a prior crop-verifier blend.
        detector_score = float(prediction.get("detector_score", prediction["score"]))
        class_probability = float(probabilities[category_id])
        foreground_probability = float(1.0 - probabilities[BACKGROUND_CLASS_ID])
        conditional = probabilities[:25] / max(foreground_probability, 1e-12)
        conditional /= max(float(conditional.sum()), 1e-12)
        top = np.argsort(-conditional, kind="stable")
        score = math.exp(
            (1.0 - args.alpha) * math.log(max(detector_score, epsilon))
            + args.alpha * math.log(max(class_probability, epsilon))
        )
        record = dict(prediction)
        record.update(
            {
                "score": score,
                "detector_score": detector_score,
                "dino_foreground_probability": foreground_probability,
                "dino_class_probability": class_probability,
                "dino_conditional_class_probability": float(conditional[category_id]),
                "dino_top1_class": int(top[0]),
                "dino_top1_probability": float(conditional[int(top[0])]),
                "dino_margin": float(conditional[int(top[0])] - conditional[int(top[1])]),
                "detector_dino_agree": int(int(top[0]) == category_id),
            }
        )
        outputs.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outputs, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "protocol": "fold_heldout_pseudo10k_dino_open_set_fixed_fusion_v1",
        "alpha": args.alpha,
        "candidate_count": len(outputs),
        "input_sha256": {"gt": _sha256(args.gt), "pred": _sha256(args.pred)},
        "features_sha256": _sha256(args.features),
        "checkpoint_sha256": checkpoint_hashes,
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
