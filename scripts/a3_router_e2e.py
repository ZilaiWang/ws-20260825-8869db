#!/usr/bin/env python3
"""A3 端到端: 可观测性路由器改类 + OER 排序 + NMS → 固定风险前沿。

路由器(5折 cross-fit, 预测"crop 是否对")只对 yolo 错细类且 crop 高可信的候选改类,
评估 corrected/broken 与 Recall@FDR 前沿, 对比不改类的 OER+NMS(0.9415)。

用法:
  python scripts/a3_router_e2e.py \
    --nodes /tmp/a1-object-graph/nodes.csv \
    --predictions /tmp/Y5-full.json \
    --scores /tmp/oer_scores.csv \
    --formal-crop-manifest outputs/.../formal_crop_manifest.csv \
    --project-config configs/project.yaml \
    --output /tmp/a3_router_e2e.json
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

OBS_FEATURES = ["short_edge", "area", "aspect", "crop_margin", "crop_entropy",
                "crop_top1", "detector_crop_agree", "local_density"]


def greedy_match(preds, formal, proto):
    pb = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    tp = {}
    used = defaultdict(set)
    for img, gts in formal.boxes.items():
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


def nms_all(preds, key, thr=0.5):
    pb = defaultdict(list)
    for p in preds:
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
    ap.add_argument("--scores", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    n_gt = sum(len(gts) for gts in formal.boxes.values())

    nodes = pd.read_csv(args.nodes)
    scores = pd.read_csv(args.scores)
    preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    for i, p in enumerate(preds):
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        p["_idx"] = i

    # 位置关联 GT 细类 + 还原候选 + OER 分数
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
    score_map = {int(r.idx): float(r.oer_score) for r in scores.itertuples()}
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}

    # 训练路由器(5折 cross-fit, 在 yolo 错的候选上预测 crop 对)
    from sklearn.ensemble import HistGradientBoostingClassifier
    train_rows = []
    for p in preds:
        if p["_idx"] not in gt_fine_of:
            continue
        r = node_map.get(p["_idx"])
        if r is None:
            continue
        yolo_fine = p["category_id"]
        if yolo_fine in gt_fine_of[p["_idx"]]:
            continue  # yolo 对, 不改类
        crop_top1 = int(r.crop_top1_class)
        crop_correct = 1 if crop_top1 in gt_fine_of[p["_idx"]] else 0
        feat = [float(r.short_edge), float(r.area), float(r.aspect), float(r.crop_margin),
                float(r.crop_entropy), float(r.crop_top1), float(r.detector_crop_agree),
                float(r.local_density)]
        train_rows.append((p["_idx"], feat, crop_correct, crop_top1, fold_of.get(p["image_id"], 0)))

    X = np.array([r[1] for r in train_rows])
    y = np.array([r[2] for r in train_rows])
    folds = np.array([r[4] for r in train_rows])
    probs = np.zeros(len(train_rows))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4,
                                             random_state=2026 + held)
        clf.fit(X[tr], y[tr])
        probs[va] = clf.predict_proba(X[va])[:, 1]

    # 路由器筛选改类
    for thr_name, thr in [("全改(规则E5)", 0.0), ("路由阈值0.7", 0.7), ("路由阈值0.9", 0.9)]:
        changed = 0
        corrected = 0
        broken = 0
        new_preds = [dict(p) for p in preds]
        for k, (idx, feat, cc, top1, _) in enumerate(train_rows):
            if probs[k] < thr:
                continue
            p = new_preds[idx]
            gt_fines = gt_fine_of[idx]
            old_cat = p["category_id"]
            # 改类前: 这个候选如果原来是 TP(细类对), 改类后可能 broken
            was_correct = old_cat in gt_fines
            new_cat = top1
            p["category_id"] = new_cat
            changed += 1
            if new_cat in gt_fines:
                corrected += 1
            elif was_correct:
                broken += 1
        # 赋予 OER 分数
        for p in new_preds:
            p["score"] = score_map.get(p["_idx"], p["score"])
            p["oer"] = p["score"]
        # NMS + frontier
        k = nms_all(new_preds, "oer")
        tp = greedy_match(k, formal, proto)
        br, bf = frontier(k, "oer", tp, n_gt)
        print(f"\n[{thr_name}] 改类 {changed} 个(corrected {corrected}, broken {broken})")
        print(f"  n={len(k)} | R@FDR=.12={br[0.12]:.4f} | R@FDR=.15={br[0.15]:.4f} | FDR@R=.94={bf[0.94]:.4f}")

    # 基线(不改类)
    base = [dict(p) for p in preds]
    for p in base:
        p["score"] = score_map.get(p["_idx"], p["score"])
        p["oer"] = p["score"]
    k = nms_all(base, "oer")
    tp = greedy_match(k, formal, proto)
    br, bf = frontier(k, "oer", tp, n_gt)
    print(f"\n[基线 不改类 OER+NMS] n={len(k)} | R@FDR=.12={br[0.12]:.4f} | FDR@R=.94={bf[0.94]:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
