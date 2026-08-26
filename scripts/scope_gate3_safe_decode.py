#!/usr/bin/env python3
"""Gate 3 桥接验证：用学习到的动作价值做安全门控，评估 deploy frontier 提升。

流程：
1. 读反事实标签 + join 候选特征；
2. LightGBM 三折 cross-fit 预测每个 RELABEL 动作的 delta_utility（收益）；
3. 安全门控：只对「预测收益 > 阈值」的候选执行 RELABEL（crop_top1 改类）；
4. 用官方 FrontierScorer 评估门控后的 deploy frontier，对比 base。

严格 OOF：按 fold 隔离训练 LightGBM。
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
    ap.add_argument("--labels", required=True)
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

    # has_oto + OER
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

    # 读标签，join 特征
    labels = [json.loads(l) for l in open(args.labels, encoding="utf-8")]
    df = pd.DataFrame(labels)
    feat_cols = [c for c in FEAT_BASE + ["has_oto"] if c in nodes.columns]
    df["_idx"] = df["candidate_id"].astype(int)
    df["fold"] = df["_idx"].map(lambda i: fold_of.get(int(node_map[i].image_id), -1))
    Xl = np.zeros((len(df), len(feat_cols)))
    for j, c in enumerate(feat_cols):
        Xl[:, j] = df["_idx"].map(lambda i: float(node_map[i].__getattribute__(c))).to_numpy()

    # RELABEL 动作的 delta_utility 标签（用于训练 + 评估）
    relabel_mask = df["action"].apply(lambda a: a.get("kind") == "relabel")
    df_r = df[relabel_mask].reset_index(drop=True)
    Xr = Xl[relabel_mask.to_numpy()]
    yr = df_r["delta_utility"].to_numpy()

    # LightGBM 三折 cross-fit 预测 delta_utility
    from sklearn.ensemble import HistGradientBoostingRegressor
    folds_r = df_r["fold"].to_numpy()
    pred_du = np.zeros(len(df_r))
    for held in (0, 1, 2):
        tr = np.where(folds_r != held)[0]; va = np.where(folds_r == held)[0]
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=6, random_state=42)
        reg.fit(Xr[tr], yr[tr]); pred_du[va] = reg.predict(Xr[va])

    # base frontier（OER+NMS 不含改类）
    scorer = FrontierScorer(proto, formal.boxes)
    base_cands = [CandidateView(_idx=i, image_id=p["image_id"], category_id=p["category_id"],
                                score=float(oer[i]), bbox_xyxy=p["bbox_xyxy"]) for i, p in enumerate(preds)]
    base_fr = scorer.frontier(base_cands)
    base_u = base_fr.recall_at_fdr[0.12]
    print(f"base frontier(三折 OER+NMS) = {base_u:.4f}")

    # oracle 改类 frontier（上界参考）
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
    print(f"oracle 改类 frontier = {oracle_u:.4f}")

    # 安全门控：按预测收益排序，逐步执行 top-K RELABEL，看 frontier 曲线
    # 每个 RELABEL 动作：candidate_id -> 改类到 crop_top1_class
    order = np.argsort(-pred_du)  # 预测收益降序
    cand_rank = {int(df_r.iloc[i]["candidate_id"]): i for i in range(len(df_r))}
    cand_alt = {int(df_r.iloc[i]["candidate_id"]): int(node_map[int(df_r.iloc[i]["candidate_id"])].crop_top1_class)
                for i in range(len(df_r))}

    results = []
    # 扫描 top-K
    for topk in (0, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 8000):
        if topk == 0:
            u = base_u
        else:
            chosen = set(int(df_r.iloc[order[j]]["candidate_id"]) for j in range(min(topk, len(order))))
            cands = []
            for i, p in enumerate(preds):
                cid = cand_alt.get(i, p["category_id"]) if i in chosen else p["category_id"]
                cands.append(CandidateView(_idx=i, image_id=p["image_id"], category_id=cid,
                                           score=float(oer[i]), bbox_xyxy=p["bbox_xyxy"]))
            u = scorer.frontier(cands).recall_at_fdr[0.12]
        results.append({"topk": topk, "frontier": u, "delta": u - base_u})
        print(f"  top-K={topk:5d}: frontier={u:.4f} (Δ{u - base_u:+.4f})")

    best = max(results, key=lambda r: r["frontier"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "base": base_u, "oracle": oracle_u,
        "best": best, "curve": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n最佳 top-K={best['topk']}: frontier={best['frontier']:.4f} (base {base_u:.4f}, oracle {oracle_u:.4f}, "
          f"恢复 {(best['frontier']-base_u)/(oracle_u-base_u)*100:.1f}%)")
    print(f"已写: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
