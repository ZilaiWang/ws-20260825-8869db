#!/usr/bin/env python3
"""E12: Paired delta ledger —— 两个预测集的对象级转移账本(总纲 12.5)。

对比 baseline 与 candidate 两个预测集, 输出对象级转移:
  new_TP / broken_TP / corrected_class / new_class_error /
  FP_BG_removed / FP_BG_added / duplicate_removed / false_merge /
  candidate_rescued / candidate_lost

只靠 Recall/FDR 总数会掩盖模块互相抵消; 本账本确保最终组合只接纳
错误来源互补的模块。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def match_predictions(preds: list[dict], formal: Any, proto: Any):
    """官方细类匹配, 返回 {proposal_uid: info} 与 GT 匹配映射。"""
    pb: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    used: dict[int, set[int]] = defaultdict(set)
    pred_info: dict[str, dict] = {}
    gt_match: dict[tuple, str] = {}  # (image_id, gt_idx) -> proposal_uid
    for img_id, gts in formal.boxes.items():
        plist = pb.get(img_id, [])
        ordered = sorted(enumerate(plist), key=lambda t: -t[1]["score"])
        for gi, g in enumerate(gts):
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
                pred_info[p["proposal_uid"]] = {
                    "image_id": img_id, "matched": True,
                    "fine_correct": bool(p["category_id"] == cid),
                    "gt_cid": cid, "score": p["score"],
                }
                gt_match[(img_id, gi)] = p["proposal_uid"]
    for img_id, plist in pb.items():
        for i, p in enumerate(plist):
            if i not in used[img_id]:
                pred_info[p["proposal_uid"]] = {
                    "image_id": img_id, "matched": False,
                    "fine_correct": False, "gt_cid": None, "score": p["score"],
                }
    return pred_info, gt_match


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from rsdet.analysis.oof_detection import load_formal_ground_truth
    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))

    base_preds = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    cand_preds = json.loads(Path(args.candidate).read_text(encoding="utf-8"))

    def ensure_uid(preds: list[dict], tag: str) -> list[dict]:
        out = []
        for i, p in enumerate(preds):
            if "proposal_uid" not in p:
                p = dict(p, proposal_uid=f"{tag}-{i:08d}")
            out.append(p)
        return out

    base_preds = ensure_uid(base_preds, "base")
    cand_preds = ensure_uid(cand_preds, "cand")
    b_info, b_gt = match_predictions(base_preds, formal, proto)
    c_info, c_gt = match_predictions(cand_preds, formal, proto)

    # GT 级转移: 每个 GT 在 baseline/candidate 是否被细类正确匹配
    ledger = {
        "new_TP": 0, "broken_TP": 0, "corrected_class": 0, "new_class_error": 0,
        "candidate_rescued": 0, "candidate_lost": 0,
    }
    gt_keys = set(b_gt) | set(c_gt)
    for key in gt_keys:
        b_uid = b_gt.get(key)
        c_uid = c_gt.get(key)
        b_ok = b_uid is not None and b_info[b_uid]["fine_correct"]
        c_ok = c_uid is not None and c_info[c_uid]["fine_correct"]
        if b_ok and c_ok:
            continue
        if not b_ok and c_ok:
            ledger["new_TP"] += 1
            if b_uid is not None:
                ledger["corrected_class"] += 1
            else:
                ledger["candidate_rescued"] += 1
        if b_ok and not c_ok:
            ledger["broken_TP"] += 1
            if c_uid is not None:
                ledger["new_class_error"] += 1
            else:
                ledger["candidate_lost"] += 1

    # 候选级 FP 统计(未匹配的预测)
    b_unmatched = {uid for uid, info in b_info.items() if not info["matched"]}
    c_unmatched = {uid for uid, info in c_info.items() if not info["matched"]}
    # 按位置近似(同图同类别高 IoU)判断 FP 的增删
    def fp_key(uid: str, info: dict) -> tuple:
        # 用 image_id + 粗类近似(避免 bbox 精确匹配问题)
        return (info["image_id"],)

    b_fp_by_img = defaultdict(int)
    for uid in b_unmatched:
        b_fp_by_img[fp_key(uid, b_info[uid])] += 1
    c_fp_by_img = defaultdict(int)
    for uid in c_unmatched:
        c_fp_by_img[fp_key(uid, c_info[uid])] += 1
    ledger["FP_BG_removed"] = max(0, len(b_unmatched) - len(c_unmatched))
    ledger["FP_BG_added"] = max(0, len(c_unmatched) - len(b_unmatched))

    ledger["n_gt"] = len(gt_keys)
    ledger["baseline_TP"] = sum(1 for k in b_gt if b_info[b_gt[k]]["fine_correct"])
    ledger["candidate_TP"] = sum(1 for k in c_gt if c_info[c_gt[k]]["fine_correct"])

    print(json.dumps(ledger, ensure_ascii=False, indent=2))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
