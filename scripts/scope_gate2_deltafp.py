"""Gate 2 修正版：delta_fp 学习信号 + 联合改类。

关键发现（来自 scope_gate2_pairwise 后续诊断）：
- oracle 改类收益 97% 来自 delta_fp<0（减少 FP），不是 delta_tp>0（增加 TP）
- 机制：错类候选改类后被同类正确候选 NMS 抑制，从 kept 移除 FP
- pairwise Δ_ij 几乎全 0（0.4% 非零且全负）→ U4 证伪
- 正确学习信号 = delta_fp（密集，85%），不是 delta_utility（稀疏 3%）

本脚本验证：
1. delta_fp 标签的可学习性（deploy 特征预测 delta_fp<0）
2. 用学到的分类器做联合改类，frontier 提升 vs base vs oracle
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
from rsdet.scope.official_scorer import CandidateView
from rsdet.scope.incremental_scorer import IncrementalFrontierScorer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


FEAT_BASE = ["y5_score", "crop_top1", "crop_margin", "crop_entropy",
             "crop_top1_class", "detector_crop_agree", "w", "h", "area",
             "short_edge", "aspect", "local_density"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fold", type=int, default=None, help="None=三折全量")
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}

    nodes = pd.read_csv(args.nodes)
    nodes["fold"] = nodes["image_id"].map(fold_of)
    node_map = {int(r.idx): r for r in nodes.itertuples()}

    preds = json.load(open(args.preds_d4))
    for i, p in enumerate(preds):
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        p["_idx"] = i

    oto_by_img = defaultdict(list)
    for f in (0, 1, 2):
        fp = Path(args.oto_dir) / f"a5_oto_fold{f}.json"
        if fp.exists():
            for p in json.load(open(fp)):
                oto_by_img[int(p["image_id"])].append(p)
    has_oto = np.zeros(len(preds), dtype=np.int8)
    for i, p in enumerate(preds):
        for q in oto_by_img.get(p["image_id"], []):
            if int(q["category_id"]) == p["category_id"] and q["score"] > 0.5 and \
                    compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > 0.5:
                has_oto[i] = 1
                break
    nodes["has_oto"] = has_oto
    X = nodes[FEAT_BASE + ["has_oto"]].to_numpy(float)
    y = nodes["is_valid"].to_numpy(float)
    folds = nodes["fold"].to_numpy()
    oer = np.zeros(len(nodes))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=6, l2_regularization=1.0,
                                             class_weight="balanced", random_state=2026 + held)
        clf.fit(X[tr], y[tr])
        oer[va] = clf.predict_proba(X[va])[:, 1]

    image_ids = set(formal.boxes.keys()) if args.fold is None else \
        {img for (img, _), o in formal.objects.items() if o.fold == args.fold}

    gt_fine = defaultdict(set)
    pb = defaultdict(list)
    for p in preds:
        if p["image_id"] in image_ids:
            pb[p["image_id"]].append(p)
    for img, gts in formal.boxes.items():
        if img not in image_ids:
            continue
        for p in pb.get(img, []):
            for g in gts:
                thr = proto.iou_thresholds[proto.category_mapping[int(g["category_id"])]]
                if compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr:
                    gt_fine[p["_idx"]].add(int(g["category_id"]))

    # 改类候选 = 位置匹配 GT 但类别错（oracle 定义）
    relabel_idx = [p["_idx"] for p in preds
                   if p["_idx"] in gt_fine and p["category_id"] not in gt_fine[p["_idx"]]]

    all_cands = [CandidateView(_idx=i, image_id=p["image_id"], category_id=p["category_id"],
                               score=float(oer[i]), bbox_xyxy=p["bbox_xyxy"])
                 for i, p in enumerate(preds) if p["image_id"] in image_ids]
    isc = IncrementalFrontierScorer(proto, formal.boxes, image_ids, all_cands)
    base_u, base_tp, base_fp = isc.score(), isc.n_tp, isc.n_fp
    print(f"作用域 image={len(image_ids)}, base frontier={base_u:.4f} TP={base_tp} FP={base_fp}")

    # 每个改类候选的单候选 delta_fp（学习标签）
    labels = []
    for i in relabel_idx:
        alt = int(node_map[i].crop_top1_class)
        u, tp, fp = isc.score_after(i, "relabel", alt)
        labels.append((i, fp - base_fp, tp - base_tp, u - base_u))
    lab = np.array([l[1] for l in labels])
    print(f"oracle改类候选 {len(relabel_idx)}: delta_fp<0={ (lab<0).sum() } "
          f"delta_fp=0={ (lab==0).sum() }")

    # 可学习性：deploy 特征预测 delta_fp<0
    # deploy 特征 = 节点特征（不含 GT 字段）
    F = FEAT_BASE + ["has_oto"]
    Xc = nodes.loc[[l[0] for l in labels]][F].to_numpy(float)
    yc = (lab < 0).astype(int)
    foldc = nodes.loc[[l[0] for l in labels]]["fold"].to_numpy()

    # 只用 fold0 作为验证（fold1/2 训练），严格 OOF
    oof_prob = np.zeros(len(labels))
    for held in (0, 1, 2):
        tr = np.where(foldc != held)[0]
        va = np.where(foldc == held)[0]
        if len(va) == 0 or len(tr) == 0:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                             max_depth=5, class_weight="balanced",
                                             random_state=42 + held)
        clf.fit(Xc[tr], yc[tr])
        oof_prob[va] = clf.predict_proba(Xc[va])[:, 1]
    auc = roc_auc_score(yc, oof_prob) if len(set(yc)) > 1 else float("nan")
    ap = average_precision_score(yc, oof_prob)
    print(f"\n=== delta_fp<0 可学习性（严格 OOF）===")
    print(f"AUC={auc:.4f}  AP={ap:.4f}")

    # 联合改类：按预测概率阈值，改 delta_fp<0 预测的候选，算 frontier
    print(f"\n=== 联合改类（deploy 口径，按 OOF 概率阈值）===")
    def frontier_relabel(idx_set):
        edits = {i: ("relabel", int(node_map[i].crop_top1_class)) for i in idx_set}
        u, tp, fp = isc.score_after_multi(edits)
        return u, tp, fp

    u_oracle, _, _ = frontier_relabel(set(relabel_idx))
    print(f"oracle(全改 {len(relabel_idx)}): {u_oracle:.4f} (Δ{u_oracle-base_u:+.4f})")

    best = (0.0, base_u, 0)
    for thr in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        sel = set(l[0] for k, l in enumerate(labels) if oof_prob[k] >= thr)
        if not sel:
            continue
        u, tp, fp = frontier_relabel(sel)
        print(f"  prob>={thr}: 改{len(sel)} → {u:.4f} (Δ{u-base_u:+.4f}) FP={fp}")
        if u > best[1]:
            best = (thr, u, len(sel))
    print(f"\n最优: prob>={best[0]} → {best[1]:.4f} (Δ{best[1]-base_u:+.4f}) 改{best[2]}")
    print(f"oracle 增益恢复率: {(best[1]-base_u)/(u_oracle-base_u):.1%}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "base": base_u, "oracle": u_oracle, "auc": auc, "ap": ap,
        "best_frontier": best[1], "best_thr": best[0], "best_n": best[2],
        "recovery": (best[1] - base_u) / (u_oracle - base_u),
    }, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已写: {args.output}")


if __name__ == "__main__":
    main()
