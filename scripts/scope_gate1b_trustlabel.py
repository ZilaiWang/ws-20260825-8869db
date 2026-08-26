#!/usr/bin/env python3
"""Gate 1b 深化：对比两种改类监督目标，找到能复现 oracle 改类收益的正确目标。

监督目标 A：trust_label = 位置匹配 GT 且 yolo 类别错（oracle 改类的定义，10398 正样本）
监督目标 B：delta_utility > 0（单候选 frontier 收益，稀疏 1.2%）

用 LightGBM 分别学，然后「一次性改所有预测该改的候选」+ 阈值扫描，
输出 frontier 曲线 + corrected/broken，量化两种目标的改类决策质量。
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
from rsdet.scope.official_scorer import FrontierScorer, CandidateView

FEAT_BASE = ["y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
             "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}
    nodes = pd.read_csv(args.nodes)
    nodes["fold"] = nodes["image_id"].map(fold_of)

    preds = json.loads(Path(args.preds_d4).read_text(encoding="utf-8"))
    for i, p in enumerate(preds):
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        p["_idx"] = i

    oto_by_img = defaultdict(list)
    for f in (0, 1, 2):
        for p in json.loads(Path(args.oto_dir, f"a5_oto_fold{f}.json").read_text(encoding="utf-8")):
            oto_by_img[int(p["image_id"])].append(p)
    has_oto = np.zeros(len(preds), dtype=np.int8)
    for i, p in enumerate(preds):
        for q in oto_by_img.get(p["image_id"], []):
            if int(q["category_id"]) == p["category_id"] and q["score"] > 0.5 and compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > 0.5:
                has_oto[i] = 1
                break
    nodes["has_oto"] = has_oto
    node_map = {int(r.idx): r for r in nodes.itertuples()}

    from sklearn.ensemble import HistGradientBoostingClassifier
    X = nodes[FEAT_BASE + ["has_oto"]].to_numpy(dtype=float)
    y = nodes["is_valid"].to_numpy(dtype=float)
    folds = nodes["fold"].to_numpy()
    oer = np.zeros(len(nodes))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]; va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                             l2_regularization=1.0, class_weight="balanced", random_state=2026 + held)
        clf.fit(X[tr], y[tr]); oer[va] = clf.predict_proba(X[va])[:, 1]

    # GT 匹配
    gt_fine = defaultdict(set)
    pb = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    for img, gts in formal.boxes.items():
        for p in pb.get(img, []):
            for g in gts:
                thr = proto.iou_thresholds[proto.category_mapping[int(g["category_id"])]]
                if compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr:
                    gt_fine[p["_idx"]].add(int(g["category_id"]))

    # trust_label（监督 A）：位置匹配且类别错
    trust = np.array([1 if (i in gt_fine and preds[i]["category_id"] not in gt_fine[i]) else 0
                      for i in range(len(preds))], dtype=int)
    print(f"trust_label 正样本: {trust.sum()} / {len(preds)}")

    # 路由器特征（含 has_oto，不含 GT）
    router_feat = FEAT_BASE + ["has_oto"]
    Xr = nodes[router_feat].to_numpy(dtype=float)

    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    probs = np.zeros(len(preds))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]; va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=5,
                                             class_weight="balanced", random_state=42 + held)
        clf.fit(Xr[tr], trust[tr]); probs[va] = clf.predict_proba(Xr[va])[:, 1]
    auc = roc_auc_score(trust, probs)
    print(f"trust_label 路由器 AUC(三折): {auc:.4f}")

    scorer = FrontierScorer(proto, formal.boxes)
    base_cands = [CandidateView(_idx=i, image_id=p["image_id"], category_id=p["category_id"],
                                score=float(oer[i]), bbox_xyxy=p["bbox_xyxy"]) for i, p in enumerate(preds)]
    base_u = scorer.frontier(base_cands).recall_at_fdr[0.12]

    oracle_cands = []
    for i, p in enumerate(preds):
        cid = p["category_id"]
        gf = gt_fine.get(i, set())
        r = node_map.get(i)
        if gf and cid not in gf and r is not None:
            cid = int(r.crop_top1_class)
        oracle_cands.append(CandidateView(_idx=i, image_id=p["image_id"], category_id=cid,
                                          score=float(oer[i]), bbox_xyxy=p["bbox_xyxy"]))
    oracle_u = scorer.frontier(oracle_cands).recall_at_fdr[0.12]
    print(f"base={base_u:.4f}  oracle改类={oracle_u:.4f} (gap {oracle_u-base_u:+.4f})")

    # 阈值扫描：改所有 probs > thr 的候选（信任 crop_top1）
    print("\n[trust_label router 阈值扫描]")
    best = (0.0, base_u, None)
    curve = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        chosen = set(i for i in range(len(preds)) if probs[i] > thr)
        cands = []
        corrected = broken = 0
        for i, p in enumerate(preds):
            cid = p["category_id"]
            if i in chosen:
                r = node_map.get(i)
                if r is not None:
                    cid = int(r.crop_top1_class)
                gf = gt_fine.get(i, set())
                orig_wrong = p["category_id"] not in gf
                new_right = cid in gf
                if orig_wrong and new_right: corrected += 1
                elif (not orig_wrong) and (not new_right): broken += 1
            cands.append(CandidateView(_idx=i, image_id=p["image_id"], category_id=cid,
                                       score=float(oer[i]), bbox_xyxy=p["bbox_xyxy"]))
        u = scorer.frontier(cands).recall_at_fdr[0.12]
        curve.append({"thr": thr, "frontier": u, "corrected": corrected, "broken": broken})
        print(f"  thr={thr}: frontier={u:.4f} (Δ{u-base_u:+.4f}) corrected={corrected} broken={broken} 改{len(chosen)}")
        if u > best[1]:
            best = (thr, u, corrected, broken)
    print(f"\n最优 thr={best[0]}: frontier={best[1]:.4f} corrected={best[2]} broken={best[3]}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "base": base_u, "oracle": oracle_u, "router_auc": auc,
        "best": {"thr": best[0], "frontier": best[1], "corrected": best[2], "broken": best[3]},
        "curve": curve,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
