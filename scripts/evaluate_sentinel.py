#!/usr/bin/env python3
"""E12 L3: prospective sentinel 评估——只统计冻结图(不参与任何开发的 555 图)。

用法:
  python scripts/evaluate_sentinel.py \
    --predictions /tmp/Balanced-full.json \
    --sentinel outputs/PROSPECTIVE_SENTINEL_20260820.json \
    --formal-crop-manifest outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv \
    --project-config configs/project.yaml \
    --output /tmp/sentinel-eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

THRESHOLDS = (0.05, 0.1, 0.2)


def evaluate(preds: list[dict], formal: Any, proto: Any, threshold: float,
             sentinel_ids: set[int]) -> dict:
    pb: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        if int(p["image_id"]) in sentinel_ids:
            pb[int(p["image_id"])].append(p)
    used: dict[int, set[int]] = defaultdict(set)
    tp = fn = 0
    macro_tp: dict[str, int] = defaultdict(int)
    macro_fn: dict[str, int] = defaultdict(int)
    for img_id, gts in formal.boxes.items():
        if img_id not in sentinel_ids:
            continue
        plist = [p for p in pb.get(img_id, []) if p["score"] >= threshold]
        ordered = sorted(enumerate(plist), key=lambda t: -t[1]["score"])
        for g in gts:
            cid = int(g["category_id"])
            thr = proto.iou_thresholds[proto.category_mapping[cid]]
            bi, bix = 0.0, None
            for idx, p in ordered:
                if idx in used[img_id] or p["category_id"] != cid:
                    continue
                iou = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if iou > bi:
                    bi, bix = iou, idx
            if bix is not None and bi >= thr:
                used[img_id].add(bix)
                tp += 1
                macro_tp[proto.category_mapping[cid]] += 1
            else:
                fn += 1
                macro_fn[proto.category_mapping[cid]] += 1
    fp = 0
    for img_id, plist in pb.items():
        for i, p in enumerate(plist):
            if p["score"] >= threshold and i not in used[img_id]:
                fp += 1
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fdr = fp / (tp + fp) if (tp + fp) else 0.0
    macros = []
    for coarse in ("aircraft", "ship", "vehicle"):
        m_tp = sum(v for k, v in macro_tp.items() if k == coarse)
        m_fn = sum(v for k, v in macro_fn.items() if k == coarse)
        macros.append(m_tp / (m_tp + m_fn) if (m_tp + m_fn) else 0.0)
    return {"recall": recall, "fdr": fdr, "tp": tp, "fp": fp, "fn": fn,
            "macro_recall": sum(macros) / len(macros)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--sentinel", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    sentinel = json.loads(Path(args.sentinel).read_text(encoding="utf-8"))
    sentinel_ids = set(sentinel["frozen_image_ids"])
    preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))

    n_gt = sum(1 for img_id, gts in formal.boxes.items() if img_id in sentinel_ids for _ in gts)
    print(f"sentinel: {len(sentinel_ids)} 图, {n_gt} GT")

    tables = {}
    for t in THRESHOLDS:
        m = evaluate(preds, formal, proto, t, sentinel_ids)
        tables[str(t)] = m
        print(f"t={t}: R={m['recall']:.4f} F={m['fdr']:.4f} macroR={m['macro_recall']:.4f} "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "sentinel": {"n_images": len(sentinel_ids), "n_gt": n_gt},
        "tables": tables,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
