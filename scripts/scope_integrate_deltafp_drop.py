"""整合实验：delta_fp 标签（SCOPE 减 FP 机制）+ 集合上下文特征（F5）+ 联合 DROP。

核心突破（本轮发现）：
- trust_label（学"位置对但类别错"）的 broken 率 = tp 里 crop 判错 = 5.5%
- delta_fp<0（学"drop 后减 FP"）的 broken 率 = tp 里 drop 后减 FP = 仅 1.2%
- delta_fp 标签直接对应"减 FP"这个真正目标，天然避开 broken

本脚本三折全量严格 OOF：
1. 对 OER>=0.1 的候选，算 drop 后 delta_fp（标签）
2. OOF 训练 delta_fp<0 分类器（基础特征 vs +集合上下文特征）
3. 联合 DROP（按预测概率阈值），评估 frontier vs base vs oracle
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

FEAT_BASE = [
    "y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
    "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density",
]


def build_context(nodes):
    """复现 f5/c7-lite 15 集合上下文特征。"""
    nodes = nodes.copy()
    nodes["rank_in_img"] = nodes.groupby("image_id")["y5_score"].rank(pct=True)
    nodes["img_n"] = nodes.groupby("image_id")["idx"].transform("count")
    nodes["img_score_mean"] = nodes.groupby("image_id")["y5_score"].transform("mean")
    nodes["img_score_max"] = nodes.groupby("image_id")["y5_score"].transform("max")
    nodes["img_oto_ratio"] = nodes.groupby("image_id")["has_oto"].transform("mean")

    def class_entropy(group):
        cnt = group.value_counts(normalize=True)
        return float(-(cnt * np.log(cnt + 1e-9)).sum())
    nodes["img_cls_entropy"] = nodes.groupby("image_id")["crop_top1_class"].transform(class_entropy)
    nodes["img_highconf_ratio"] = nodes.groupby("image_id")["y5_score"].transform(
        lambda s: float((s > 0.5).mean()))

    def spatial_neighbors(g):
        cxs = g["cx"].to_numpy(); cys = g["cy"].to_numpy()
        scores = g["y5_score"].to_numpy(); cls = g["crop_top1_class"].to_numpy()
        n = len(g)
        nn_dist = np.full(n, np.nan); nn_score = np.full(n, np.nan); same_cls = np.zeros(n)
        for i in range(n):
            d = np.sqrt((cxs - cxs[i]) ** 2 + (cys - cys[i]) ** 2)
            d[i] = np.inf
            j = np.argmin(d)
            nn_dist[i] = d[j]; nn_score[i] = scores[j]
            same_cls[i] = (cls == cls[i]).sum() - 1
        return pd.DataFrame({"nn_dist": nn_dist, "nn_score": nn_score, "same_cls_n": same_cls},
                            index=g.index)
    spatial = nodes.groupby("image_id", group_keys=False).apply(spatial_neighbors)
    nodes = pd.concat([nodes, spatial], axis=1)

    nodes["img_area_median"] = nodes.groupby("image_id")["area"].transform("median")
    nodes["rel_area"] = nodes["area"] / (nodes["img_area_median"] + 1e-6)
    nodes["img_top3_score"] = nodes.groupby("image_id")["y5_score"].transform(
        lambda s: float(s.nlargest(3).mean()))
    nodes["nn_same_cls"] = (nodes["same_cls_n"] > 0).astype(int)
    nodes["nn_dist_log"] = np.log1p(nodes["nn_dist"])

    def local_density_200(g):
        cxs = g["cx"].to_numpy(); cys = g["cy"].to_numpy()
        n = len(g); out = np.zeros(n)
        for i in range(n):
            d = np.sqrt((cxs - cxs[i]) ** 2 + (cys - cys[i]) ** 2)
            out[i] = (d < 200).sum() - 1
        return pd.DataFrame({"local_density_200": out}, index=g.index)
    ld = nodes.groupby("image_id", group_keys=False).apply(local_density_200)
    nodes = pd.concat([nodes, ld], axis=1)
    CTX = ["rank_in_img", "img_n", "img_score_mean", "img_score_max",
           "img_oto_ratio", "img_cls_entropy", "img_highconf_ratio",
           "nn_dist_log", "nn_score", "same_cls_n", "nn_same_cls",
           "rel_area", "img_top3_score", "local_density_200"]
    return nodes, CTX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--oer-min", type=float, default=0.1, help="只对 OER>=阈值的候选算 delta_fp")
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
    nodes["area"] = nodes["w"] * nodes["h"]

    # 集合上下文
    nodes, CTX = build_context(nodes)

    # OER 训练（is_valid 标签）
    F = FEAT_BASE + ["has_oto"]
    X = nodes[F].to_numpy(float)
    y = nodes["is_valid"].to_numpy(float)
    folds = nodes["fold"].to_numpy()
    oer = np.zeros(len(nodes))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]; va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                             l2_regularization=1.0, class_weight="balanced",
                                             random_state=2026 + held)
        clf.fit(X[tr], y[tr]); oer[va] = clf.predict_proba(X[va])[:, 1]

    image_ids = set(formal.boxes.keys())
    # 按 fold 分别算 delta_fp（避免三折全量增量 scorer OOM）
    hi_idx = [i for i in range(len(preds)) if oer[i] >= args.oer_min]
    print(f"OER>={args.oer_min} 的候选: {len(hi_idx)}")
    labels = np.zeros(len(hi_idx), dtype=int)
    base_u_global = 0.0
    base_fp_global = 0
    for fold in (0, 1, 2):
        fold_img = {img for (img, _), o in formal.objects.items() if o.fold == fold}
        fold_cands = [CandidateView(_idx=i, image_id=p["image_id"], category_id=p["category_id"],
                                    score=float(oer[i]), bbox_xyxy=p["bbox_xyxy"])
                      for i, p in enumerate(preds) if p["image_id"] in fold_img]
        isc = IncrementalFrontierScorer(proto, formal.boxes, fold_img, fold_cands)
        base_u, base_tp, base_fp = isc.score(), isc.n_tp, isc.n_fp
        base_u_global += base_u
        base_fp_global += base_fp
        fold_hi = [k for k, i in enumerate(hi_idx) if preds[i]["image_id"] in fold_img]
        for k in fold_hi:
            i = hi_idx[k]
            u, tp_, fp = isc.score_after(i, "drop", None)
            labels[k] = 1 if fp < base_fp else 0
        print(f"  fold{fold}: base={base_u:.4f} FP={base_fp}, 算 {len(fold_hi)} 个 delta_fp")
    hi_idx = np.array(hi_idx)
    print(f"delta_fp<0 占比: {labels.mean():.4f}")

    # 三折全局 base（用于最终评估，逐 fold 用各自 isc）
    base_u = base_u_global / 3.0

    # OOF 训练 delta_fp<0 分类器
    F_c = FEAT_BASE + ["has_oto"]
    Xc = nodes.loc[hi_idx][F_c].to_numpy(float)
    Xc_ctx = nodes.loc[hi_idx][F_c + CTX].to_numpy(float)
    foldc = nodes.loc[hi_idx]["fold"].to_numpy()

    def oof(feat):
        oof_p = np.zeros(len(hi_idx))
        for held in (0, 1, 2):
            tr = np.where(foldc != held)[0]; va = np.where(foldc == held)[0]
            if len(va) == 0 or len(tr) == 0:
                continue
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=5,
                                                 class_weight="balanced", random_state=42 + held)
            clf.fit(feat[tr], labels[tr]); oof_p[va] = clf.predict_proba(feat[va])[:, 1]
        return oof_p

    p_base = oof(Xc)
    p_ctx = oof(Xc_ctx)
    auc_b = roc_auc_score(labels, p_base)
    auc_c = roc_auc_score(labels, p_ctx)
    ap_c = average_precision_score(labels, p_ctx)
    print(f"\n=== delta_fp<0 可学习性（严格 OOF）===")
    print(f"基础特征 AUC={auc_b:.4f}")
    print(f"+集合上下文 AUC={auc_c:.4f}  AP={ap_c:.4f}")

    # 联合 DROP 评估（用 FrontierScorer 三折全局，单次调用不 OOM）
    from rsdet.scope.official_scorer import FrontierScorer
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

    scorer = FrontierScorer(proto, formal.boxes, image_ids)
    all_cands = [CandidateView(_idx=i, image_id=p["image_id"], category_id=p["category_id"],
                               score=float(oer[i]), bbox_xyxy=p["bbox_xyxy"])
                 for i, p in enumerate(preds) if p["image_id"] in image_ids]

    def frontier_with(drop_set):
        cands = [c for c in all_cands if c._idx not in drop_set]
        return scorer.frontier(cands).recall_at_fdr[0.12]

    base_u = frontier_with(set())
    fn_wrong = [i for i in range(len(preds)) if i in gt_fine and preds[i]["category_id"] not in gt_fine[i]]
    u_oracle = frontier_with(set(fn_wrong))
    print(f"\n三折 base={base_u:.4f}  oracle DROP(全部fn_wrong)={u_oracle:.4f} (Δ{u_oracle-base_u:+.4f})")

    def drop_eval(prob, thr):
        sel = set(hi_idx[np.where(prob >= thr)[0]])
        u = frontier_with(sel)
        broken = sum(1 for i in sel if i in gt_fine and preds[i]["category_id"] in gt_fine[i])
        return u, len(sel), broken

    print(f"\n=== 联合 DROP（按 delta_fp 概率阈值）===")
    best = (0, base_u, 0)
    for thr in (0.3, 0.5, 0.7, 0.8, 0.9):
        u, n, broken = drop_eval(p_ctx, thr)
        print(f"  prob>={thr}: drop {n} (broken {broken}) → {u:.4f} (Δ{u-base_u:+.4f})")
        if u > best[1]:
            best = (thr, u, n)
    print(f"\n最优: prob>={best[0]} → {best[1]:.4f} (Δ{best[1]-base_u:+.4f}) drop {best[2]}")
    print(f"oracle DROP 恢复率: {(best[1]-base_u)/(u_oracle-base_u):.1%}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "base": base_u, "oracle": u_oracle,
        "auc_base": auc_b, "auc_ctx": auc_c, "ap_ctx": ap_c,
        "best_frontier": best[1], "best_thr": best[0], "best_n": best[2],
        "recovery": (best[1] - base_u) / (u_oracle - base_u),
    }, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已写: {args.output}")


if __name__ == "__main__":
    main()
