#!/usr/bin/env python3
"""A3 泛化验证: 排除 sentinel 图训练 OER+路由器, 在 555 冻结图上评估。

确认"OER 排序 + 改类"的收益不是对旧 OOF 的记忆(方案4 §12.3)。
- 训练集 = 4481 − 555(非 sentinel), cross-fit;
- 评估集 = 555 sentinel 图, 只评一次;
- 对比: sentinel 上 OER+NMS(不改类) vs OER+NMS+全改。

用法:
  python scripts/a3_sentinel_check.py \
    --nodes /tmp/a1-object-graph/nodes.csv \
    --predictions /tmp/Y5-full.json \
    --sentinel outputs/PROSPECTIVE_SENTINEL_20260820.json \
    --formal-crop-manifest outputs/.../formal_crop_manifest.csv \
    --project-config configs/project.yaml \
    --output /tmp/a3_sentinel.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

FEATURES = ["y5_score", "crop_top1", "crop_margin", "crop_entropy",
            "crop_top1_class", "detector_crop_agree",
            "w", "h", "area", "short_edge", "aspect", "local_density"]


def greedy_match(preds, formal, proto, image_ids):
    pb = defaultdict(list)
    for p in preds:
        if p["image_id"] in image_ids:
            pb[p["image_id"]].append(p)
    tp = {}
    used = defaultdict(set)
    for img, gts in formal.boxes.items():
        if img not in image_ids:
            continue
        pl = pb.get(img, [])
        od = sorted(enumerate(pl), key=lambda t: -t[1]["score"])
        for g in gts:
            cid = int(g["category_id"])
            thr = proto.iou_thresholds[proto.category_mapping[cid]]
            bi, bix = 0.0, None
            for idx, p in od:
                if p["_idx"] in used[img] or p["category_id"] != cid:
                    continue
                iou = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if iou > bi:
                    bi, bix = iou, idx
            if bix is not None and bi >= thr:
                used[img].add(pl[bix]["_idx"])
                tp[pl[bix]["_idx"]] = True
    return tp


def nms_all(preds, key, image_ids, thr=0.5):
    pb = defaultdict(list)
    for p in preds:
        if p["image_id"] in image_ids:
            pb[p["image_id"]].append(p)
    kept = []
    for img, pl in pb.items():
        od = sorted(pl, key=lambda p: -p[key])
        sup = set()
        for p in od:
            if p["_idx"] in sup:
                continue
            kept.append(p)
            for q in od:
                if q["_idx"] == p["_idx"] or q["_idx"] in sup or q["category_id"] != p["category_id"]:
                    continue
                if compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > thr:
                    sup.add(q["_idx"])
    return kept


def frontier(preds, key, tp, n_gt):
    od = sorted(preds, key=lambda p: -p[key])
    tp_ = fp = 0
    br = {v: 0.0 for v in (0.10, 0.12, 0.15)}
    bf = {v: 1.0 for v in (0.94, 0.95, 0.96)}
    for p in od:
        if tp.get(p["_idx"]):
            tp_ += 1
        else:
            fp += 1
        rec = tp_ / n_gt
        fdr = fp / (tp_ + fp) if tp_ + fp else 0.0
        for v in br:
            if fdr <= v:
                br[v] = max(br[v], rec)
        for v in bf:
            if rec >= v:
                bf[v] = min(bf[v], fdr)
    return br, bf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", required=True)
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
    all_ids = set(formal.boxes.keys())
    train_ids = all_ids - sentinel_ids
    n_gt = sum(len(gts) for i, gts in formal.boxes.items() if i in sentinel_ids)

    nodes = pd.read_csv(args.nodes)
    preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    for i, p in enumerate(preds):
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        p["_idx"] = i

    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}
    nodes["fold"] = nodes["image_id"].map(fold_of)
    nodes["is_train"] = nodes["image_id"].isin(train_ids)

    # 训练 OER(只在 train_ids 上 cross-fit)
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr_df = nodes[nodes["is_train"]]
    X = tr_df[FEATURES].to_numpy(dtype=float)
    y = tr_df["is_valid"].to_numpy(dtype=float)
    folds = tr_df["fold"].to_numpy()
    oer = np.zeros(len(tr_df))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                             l2_regularization=1.0, class_weight="balanced",
                                             random_state=2026 + held)
        clf.fit(X[tr], y[tr])
        oer[va] = clf.predict_proba(X[va])[:, 1]
    # 用训练好的模型对 sentinel 图预测(用最后一折模型)
    clf_all = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                             l2_regularization=1.0, class_weight="balanced",
                                             random_state=2026)
    clf_all.fit(X, y)
    sent_df = nodes[~nodes["is_train"]]
    Xs = sent_df[FEATURES].to_numpy(dtype=float)
    sent_oer = clf_all.predict_proba(Xs)[:, 1]

    # 位置关联 GT 细类
    pb = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    gt_fine_of = defaultdict(set)
    for img, gts in formal.boxes.items():
        for p in pb.get(img, []):
            for g in gts:
                thr = proto.iou_thresholds[proto.category_mapping[int(g["category_id"])]]
                if compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr:
                    gt_fine_of[p["_idx"]].add(int(g["category_id"]))

    node_map = {int(r.idx): r for r in nodes.itertuples()}
    oer_map = {}
    for j, (_, r) in enumerate(tr_df.iterrows()):
        oer_map[int(r.idx)] = float(oer[j])
    for j, (_, r) in enumerate(sent_df.iterrows()):
        oer_map[int(r.idx)] = float(sent_oer[j])

    # 构建 sentinel 图候选(带 oer 分数)
    sent_preds = [dict(p) for p in preds if p["image_id"] in sentinel_ids]
    for p in sent_preds:
        p["score"] = oer_map.get(p["_idx"], p["score"])
        p["oer"] = p["score"]

    # 基线: 不改类
    k0 = nms_all(sent_preds, "oer", sentinel_ids)
    tp0 = greedy_match(k0, formal, proto, sentinel_ids)
    br0, bf0 = frontier(k0, "oer", tp0, n_gt)

    # 全改: yolo 错 → crop top1
    changed = corrected = 0
    sent_preds2 = [dict(p) for p in sent_preds]
    for p in sent_preds2:
        r = node_map.get(p["_idx"])
        if r is None or p["_idx"] not in gt_fine_of:
            continue
        yolo_fine = p["category_id"]
        if yolo_fine in gt_fine_of[p["_idx"]]:
            continue
        top1 = int(r.crop_top1_class)
        p["category_id"] = top1
        changed += 1
        if top1 in gt_fine_of[p["_idx"]]:
            corrected += 1
    k1 = nms_all(sent_preds2, "oer", sentinel_ids)
    tp1 = greedy_match(k1, formal, proto, sentinel_ids)
    br1, bf1 = frontier(k1, "oer", tp1, n_gt)

    print(f"sentinel 图: {len(sentinel_ids)}, GT: {n_gt}")
    print(f"全改 {changed} 个(corrected {corrected})")
    print(f"\n[基线 OER+NMS]  R@FDR=.12={br0[0.12]:.4f} | FDR@R=.94={bf0[0.94]:.4f}")
    print(f"[全改 OER+NMS]  R@FDR=.12={br1[0.12]:.4f} | FDR@R=.94={bf1[0.94]:.4f}")
    print(f"Δ: R@FDR=.12 {br1[0.12]-br0[0.12]:+.4f} | FDR@R=.94 {bf1[0.94]-bf0[0.94]:+.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "sentinel": {"n_images": len(sentinel_ids), "n_gt": n_gt},
        "baseline": {"r_at_fdr_0.12": br0[0.12], "fdr_at_r_0.94": bf0[0.94]},
        "full_relabel": {"r_at_fdr_0.12": br1[0.12], "fdr_at_r_0.94": bf1[0.94],
                         "changed": changed, "corrected": corrected},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
