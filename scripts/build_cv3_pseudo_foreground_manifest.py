#!/usr/bin/env python3
"""Mine foreground and clear-background proposals from held-out pseudo-10K.

The detector that produced each proposal never trained on the proposal's
formal CV3 fold.  Proposals with strong geometric support from any GT object
are foreground; proposals far from every GT object are clear background.
The uncertain IoU band is excluded instead of being assigned a noisy label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.background_gate import coarse_of_category_id, expand_context_bbox


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("non-finite bbox")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("bbox must have positive extent")
    return x, y, x + width, y + height


def _iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def classify_proposal(
    prediction_box: tuple[float, float, float, float],
    ground_truth: list[tuple[tuple[float, float, float, float], int]],
    *,
    negative_iou: float,
) -> tuple[str, float, int | None]:
    """Return label, max IoU and supporting GT category.

    The foreground threshold follows the supported GT object's official
    geometry threshold: 0.35 for vehicle and 0.50 otherwise.  Category is not
    used for matching because this is an object-vs-background verifier.
    """

    if not ground_truth:
        return "background", 0.0, None
    overlaps = [(_iou(prediction_box, box), category_id) for box, category_id in ground_truth]
    max_iou, category_id = max(overlaps, key=lambda value: value[0])
    positive_iou = 0.35 if category_id == 24 else 0.50
    if max_iou >= positive_iou:
        return "foreground", max_iou, category_id
    if max_iou <= negative_iou:
        return "background", max_iou, category_id
    return "ambiguous", max_iou, category_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--negative-iou", type=float, default=0.05)
    parser.add_argument("--context-ratio", type=float, default=1.25)
    args = parser.parse_args()
    if not 0.0 <= args.negative_iou < 0.35:
        raise ValueError("negative-iou must be in [0, 0.35)")

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    predictions = json.loads(args.pred.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in gt["images"]}
    ground_truth: dict[int, list[tuple[tuple[float, float, float, float], int]]] = (
        defaultdict(list)
    )
    for item in gt["annotations"]:
        ground_truth[int(item["image_id"])].append(
            (_xywh_to_xyxy(item["bbox"]), int(item["category_id"]))
        )

    fields = [
        "proposal_uid",
        "fold",
        "coarse",
        "class_id",
        "is_foreground",
        "score",
        "context_x0",
        "context_y0",
        "context_x1",
        "context_y1",
        "source_relative_path",
        "image_id",
        "proposal_index",
        "max_iou_any_gt",
        "support_gt_category_id",
    ]
    rows: list[dict[str, Any]] = []
    labels: Counter[str] = Counter()
    by_fold_coarse_label: Counter[str] = Counter()
    for index, item in enumerate(predictions):
        image_id = int(item["image_id"])
        meta = images[image_id]
        fold = int(meta["fold"])
        category_id = int(item["category_id"])
        coarse = coarse_of_category_id(category_id)
        box = _xywh_to_xyxy(item["bbox"])
        label, max_iou, support = classify_proposal(
            box, ground_truth.get(image_id, []), negative_iou=args.negative_iou
        )
        labels[label] += 1
        by_fold_coarse_label[f"fold{fold}/{coarse}/{label}"] += 1
        if label == "ambiguous":
            continue
        context = expand_context_bbox(box, ratio=args.context_ratio)
        rows.append(
            {
                "proposal_uid": f"pseudo-i{image_id}-p{index:06d}",
                "fold": fold,
                "coarse": coarse,
                "class_id": category_id,
                "is_foreground": int(label == "foreground"),
                "score": float(item["score"]),
                "context_x0": context[0],
                "context_y0": context[1],
                "context_x1": context[2],
                "context_y1": context[3],
                "source_relative_path": str(
                    Path(f"fold_{fold}") / "images" / str(meta["file_name"])
                ),
                "image_id": image_id,
                "proposal_index": index,
                "max_iou_any_gt": max_iou,
                "support_gt_category_id": "" if support is None else support,
            }
        )

    # The existing balanced trainer requires foreground and background support
    # for every coarse category in every two-fold training complement.
    for held_out in (0, 1, 2):
        training = [row for row in rows if int(row["fold"]) != held_out]
        for coarse in ("ship", "aircraft", "vehicle"):
            for is_foreground in (0, 1):
                if not any(
                    row["coarse"] == coarse
                    and int(row["is_foreground"]) == is_foreground
                    for row in training
                ):
                    raise ValueError(
                        f"held_out={held_out} lacks coarse={coarse} label={is_foreground}"
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "complete",
        "protocol": "formal_cv3_pseudo10k_geometric_foreground_mining_v1",
        "warning": "Pseudo-10K is a deployment proxy, not an independent benchmark.",
        "gt_sha256": _sha256(args.gt),
        "pred_sha256": _sha256(args.pred),
        "pseudo_root": str(args.pseudo_root.resolve()),
        "negative_iou": args.negative_iou,
        "context_ratio": args.context_ratio,
        "input_predictions": len(predictions),
        "retained_rows": len(rows),
        "label_counts_before_exclusion": dict(sorted(labels.items())),
        "by_fold_coarse_label": dict(sorted(by_fold_coarse_label.items())),
        "manifest_sha256": _sha256(args.output),
    }
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
