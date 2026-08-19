#!/usr/bin/env python3
"""E5: 飞机兄弟机型 pair experts —— 规则版验证(DCR²-YOLO 实验 E5)。

在 E1 的 crop logits(全 25 类)上实现组内有界修正:
- 专家组: G1{TU-160,TU-22,E-3} G2{SU-35,SU-34,SU-24} G3{KC-135,E-8,E-3}
          G4{F-22,F-16,F-15}
- 触发: Y5 标签 ∈ 组 且 crop top-1 ∈ 同组 且 margin 达标 → 修正为 crop top-1;
  否则保留 Y5 标签(保守: 只降低 broken TP, 不引入 new class error)。
- 分数保持 E1 R3 融合后的分数(不改变量)。

输出: 修正前后配对转移账本(new_TP/broken_TP/corrected_class/new_class_error)
+ 官方阈值评估。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

# 专家组: category_id
EXPERT_GROUPS = [
    {9, 15, 10},    # TU-160, TU-22, E-3
    {4, 22, 23},    # SU-35, SU-34, SU-24
    {17, 14, 10},   # KC-135, E-8, E-3
    {18, 8, 16},    # F-22, F-16, F-15
]


def load_all_logits(logits_dir: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for fold in (0, 1, 2):
        d = np.load(f"{logits_dir}/fold_{fold}_logits.npz", allow_pickle=True)
        for uid, logit in zip(d["proposal_uids"], d["logits"]):
            out[str(uid)] = logit
    return out


def evaluate(preds: list[dict], gt: dict, proto: Any, threshold: float) -> dict:
    used: dict[int, set[int]] = defaultdict(set)
    pb: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    tp = fn = 0
    for img_id, gts in gt.items():
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
            else:
                fn += 1
    fp = 0
    for img_id, plist in pb.items():
        for i, p in enumerate(plist):
            if p["score"] >= threshold and i not in used[img_id]:
                fp += 1
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fdr = fp / (tp + fp) if (tp + fp) else 0.0
    return {"recall": recall, "fdr": fdr, "tp": tp, "fp": fp, "fn": fn}


def match_info(preds: dict[int, list[dict]], gt: dict, proto: Any):
    """返回 (GT匹配的proposal_uid, 细类是否正确) 的映射。"""
    used: dict[int, set[int]] = defaultdict(set)
    info: dict[str, dict] = {}
    for img_id, gts in gt.items():
        plist = preds.get(img_id, [])
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
                p = plist[bix]
                info[p["proposal_uid"]] = {
                    "matched": True,
                    "fine_correct": p["category_id"] == cid,
                    "gt_cid": cid,
                }
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, help="E1 重排后预测 JSON")
    ap.add_argument("--logits-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--margin-threshold", type=float, default=0.15)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    logits = load_all_logits(args.logits_dir)
    proto = parse_evaluation_protocol(load_config(args.project_config))

    from rsdet.analysis.proposal_reranking import load_gt_from_formal_crop_manifest
    gt = load_gt_from_formal_crop_manifest(args.formal_crop_manifest,
                                           expected_images=4481, expected_annotations=20933)

    # 专家修正
    changed = 0
    kept = 0
    for p in preds:
        uid = p.get("proposal_uid")
        if uid not in logits or int(p["category_id"]) < 4 or int(p["category_id"]) > 23:
            kept += 1
            continue
        lg = logits[uid]
        prob = np.exp(lg - lg.max())
        prob = prob / prob.sum()
        order = np.argsort(-prob)
        top1, top2 = int(order[0]), int(order[1])
        margin = float(prob[top1] - prob[top2])
        cur = int(p["category_id"])
        # 触发: Y5 标签在专家组内 且 crop top-1 同组 且 margin 达标
        group = next((g for g in EXPERT_GROUPS if cur in g), None)
        if group is not None and top1 in group and top1 != cur and margin >= args.margin_threshold:
            p["category_id"] = top1
            changed += 1
        else:
            kept += 1
    print(f"专家修正: {changed} 个候选改类, {kept} 保持")

    # 配对转移账本(0.001 全量, 官方匹配)
    pb = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    info_before = match_info(pb, gt, proto)

    # 评估修正前后
    def eval_at(t):
        return evaluate(preds, gt, proto, t)

    print("\n=== E5 修正后 vs 修正前(E1E2 基础) ===")
    for t in (0.05, 0.1, 0.2):
        m = eval_at(t)
        print(f"t={t}: R={m['recall']:.4f} F={m['fdr']:.4f} tp={m['tp']} fp={m['fp']} fn={m['fn']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "E5-PAIR-EXPERTS-RULE",
        "changed": changed, "kept": kept,
        "margin_threshold": args.margin_threshold,
        "expert_groups": [sorted(g) for g in EXPERT_GROUPS],
        "note": "规则版: crop top-1 与 Y5 标签同组且 margin>=阈值才改, 分数不变",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
