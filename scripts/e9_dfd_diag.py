#!/usr/bin/env python3
"""E9 DFD 诊断: 对比 Y5/COPH/DFD fold0 的 candidate-floor 与粗类 NO_CAND。

判断 DFD 密集前景监督是否治疗了候选缺失(vehicle/ship 漏检)。
- candidate-floor: 贪心匹配口径(与官方评估一致), 无限低阈值下能匹配的 GT 比例;
- NO_CAND: 多对多位置关联口径, 完全没有候选位置匹配的 GT 数(漏检);
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth  # noqa: E402
from rsdet.evaluation.official_metric import compute_iou  # noqa: E402
from rsdet.evaluation.protocol import parse_evaluation_protocol  # noqa: E402
from rsdet.utils.config import load_config  # noqa: E402


def load_preds(path: str, fold0_img: set[int]) -> list[dict]:
    preds = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for p in preds:
        img = int(p["image_id"])
        if img not in fold0_img:
            continue
        out.append({
            "image_id": img,
            "category_id": int(p["category_id"]),
            "score": float(p["score"]),
            "bbox_xyxy": [float(v) for v in p["bbox_xyxy"]],
        })
    return out


def candidate_floor(preds: list[dict], formal, proto, fold_img: set[int]) -> float:
    """贪心匹配口径的候选 floor(按 score 降序贪心)。"""
    pb: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    used: dict[int, set] = defaultdict(set)
    tp = 0
    for img, gts in formal.boxes.items():
        if img not in fold_img:
            continue
        pl = pb.get(img, [])
        ordered = sorted(enumerate(pl), key=lambda t: -t[1]["score"])
        for g in gts:
            cid = int(g["category_id"])
            thr = proto.iou_thresholds[proto.category_mapping[cid]]
            bi, bix = 0.0, None
            for idx, p in ordered:
                if idx in used[img] or p["category_id"] != cid:
                    continue
                iou = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if iou > bi:
                    bi, bix = iou, idx
            if bix is not None and bi >= thr:
                used[img].add(bix)
                tp += 1
    return tp


def no_cand_by_coarse(preds: list[dict], formal, proto, fold_img: set[int]):
    """多对多位置关联口径的 NO_CAND 计数(按粗类)。"""
    pb: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    no_cand = defaultdict(int)
    total = defaultdict(int)
    for img, gts in formal.boxes.items():
        if img not in fold_img:
            continue
        pl = pb.get(img, [])
        for g in gts:
            cid = int(g["category_id"])
            coarse = proto.category_mapping[cid]
            thr = proto.iou_thresholds[coarse]
            total[coarse] += 1
            matched = any(
                compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr for p in pl
            )
            if not matched:
                no_cand[coarse] += 1
    return {c: no_cand[c] for c in total}, dict(total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--preds", action="append", required=True,
                    help="name:path 可重复")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    fold_img = {i for (i, _), o in formal.objects.items() if o.fold == args.fold}
    n_gt = sum(
        1 for img, gts in formal.boxes.items() if img in fold_img for _ in gts
    )

    print(f"fold{args.fold} GT 数: {n_gt}\n")
    print(f"{'模型':12s} {'候选':>8s} {'cand-floor':>11s} "
          f"{'aircraft_NO_CAND':>16s} {'ship_NO_CAND':>13s} {'vehicle_NO_CAND':>15s}")
    for spec in args.preds:
        name, path = spec.split(":", 1)
        preds = load_preds(path, fold_img)
        tp = candidate_floor(preds, formal, proto, fold_img)
        floor = tp / n_gt
        no_cand, total = no_cand_by_coarse(preds, formal, proto, fold_img)
        a = f"{no_cand.get('aircraft',0)}/{total.get('aircraft',0)}"
        s = f"{no_cand.get('ship',0)}/{total.get('ship',0)}"
        v = f"{no_cand.get('vehicle',0)}/{total.get('vehicle',0)}"
        print(f"{name:12s} {len(preds):8d} {floor:11.4f} {a:>16s} {s:>13s} {v:>15s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
