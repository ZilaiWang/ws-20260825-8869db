"""F5-deploy: 集合上下文纳入 GT-blind 路由器(方案5 §二 P0-2 + §十 F5)。

上一轮唯一正向方向是"集合上下文"(C7-lite 15 特征, 同图邻居统计, +0.26pp),
且它不依赖 GT(纯预测), 理应能同时提升 deploy 口径。本脚本:
1. 复现集合上下文 15 特征;
2. OER(14) vs OER(14+15上下文);
3. GT-blind 路由器(11) vs 路由器(11+15上下文);
4. 对比 oracle 改类 vs deploy 路由器改类(有/无上下文)的 fixed-risk frontier,
   回答"集合上下文能否在可部署口径下带来增益"。

纯本地(基于 nodes + d4 + oto, 无需 GPU)。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

FEAT_BASE = [
    "y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
    "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density",
    "d4_support", "has_oto",
]
ROUTER_FEAT = ["y5_score", "crop_top1", "crop_margin", "crop_entropy",
               "detector_crop_agree", "short_edge", "area", "aspect", "local_density",
               "d4_support", "has_oto"]


def build_context(nodes):
    """复现 c7-lite 15 集合上下文特征。"""
    nodes = nodes.copy()
    nodes["rank_in_img"] = nodes.groupby("image_id")["y5_score"].rank(pct=True)
    nodes["img_n"] = nodes.groupby("image_id")["idx"].transform("count")
    nodes["img_score_mean"] = nodes.groupby("image_id")["y5_score"].transform("mean")
    nodes["img_score_max"] = nodes.groupby("image_id")["y5_score"].transform("max")
    nodes["img_oto_ratio"] = nodes.groupby("image_id")["has_oto"].transform("mean")
    nodes["img_d4_mean"] = nodes.groupby("image_id")["d4_support"].transform("mean")

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

    nodes["area"] = nodes["w"] * nodes["h"]
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
           "img_oto_ratio", "img_d4_mean", "img_cls_entropy", "img_highconf_ratio",
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
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}
    n_gt = sum(len(gts) for gts in formal.boxes.values())

    nodes = pd.read_csv(args.nodes)
    nodes["fold"] = nodes["image_id"].map(fold_of)
    preds4 = json.load(open(args.preds_d4))
    nodes["d4_support"] = nodes["proposal_uid"].map(
        {p["proposal_uid"]: p["d4_support"] for p in preds4}).fillna(0).astype(int)
    oto_by_img = defaultdict(list)
    for f in (0, 1, 2):
        for p in json.load(open(Path(args.oto_dir) / f"a5_oto_fold{f}.json")):
            p["category_id"] = int(p["category_id"])
            p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
            p["image_id"] = int(p["image_id"])
            oto_by_img[p["image_id"]].append(p)

    def has_oto(r):
        bb = [r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2]
        for q in oto_by_img.get(int(r.image_id), []):
            if q["category_id"] != int(r.category_id) or q["score"] < 0.5:
                continue
            if compute_iou(bb, q["bbox_xyxy"]) > 0.5:
                return 1
        return 0
    nodes["has_oto"] = nodes.apply(has_oto, axis=1)

    # 集合上下文
    nodes, CTX = build_context(nodes)
    print(f"集合上下文特征: {len(CTX)} 个")

    # 候选 + GT 匹配
    cand = []
    for r in nodes.itertuples():
        cand.append({"_idx": int(r.idx), "image_id": int(r.image_id),
                     "category_id": int(r.category_id), "crop_top1_class": int(r.crop_top1_class),
                     "bbox_xyxy": [r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2]})
    pb = defaultdict(list)
    for p in cand:
        pb[p["image_id"]].append(p)
    gt_fine = defaultdict(set)
    for img, gts in formal.boxes.items():
        for p in pb.get(img, []):
            for g in gts:
                thr = proto.iou_thresholds[proto.category_mapping[int(g["category_id"])]]
                if compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr:
                    gt_fine[p["_idx"]].add(int(g["category_id"]))

    trust_label = {}
    for p in cand:
        gf = gt_fine.get(p["_idx"], set())
        if not gf:
            trust_label[p["_idx"]] = 0
        else:
            trust_label[p["_idx"]] = 1 if (p["category_id"] not in gf) else 0

    folds = nodes["fold"].to_numpy()

    def train_oer(feats):
        X = nodes[feats].to_numpy(dtype=float)
        y = nodes["is_valid"].to_numpy(dtype=float)
        probs = np.zeros(len(nodes))
        for held in (0, 1, 2):
            tr = np.where(folds != held)[0]; va = np.where(folds == held)[0]
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                                 l2_regularization=1.0, random_state=2026 + held)
            clf.fit(X[tr], y[tr]); probs[va] = clf.predict_proba(X[va])[:, 1]
        return probs

    def train_router(feats):
        X = nodes[feats].to_numpy(dtype=float)
        yr = np.array([trust_label[int(r.idx)] for r in nodes.itertuples()], dtype=float)
        probs = np.zeros(len(nodes))
        for held in (0, 1, 2):
            tr = np.where(folds != held)[0]; va = np.where(folds == held)[0]
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=5,
                                                 class_weight="balanced", random_state=42 + held)
            clf.fit(X[tr], yr[tr]); probs[va] = clf.predict_proba(X[va])[:, 1]
        return probs, yr

    def evaluate(oer_probs, relabel_mode, relabel_decision=None, thr=0.5):
        newp = [dict(p) for p in cand]
        changed = set()
        for i, p in enumerate(newp):
            if relabel_mode == "oracle":
                gf = gt_fine.get(p["_idx"], set())
                if gf and p["category_id"] not in gf:
                    p["category_id"] = p["crop_top1_class"]
                    changed.add(p["_idx"])
            else:
                if relabel_decision[i] > thr:
                    p["category_id"] = p["crop_top1_class"]
                    changed.add(p["_idx"])
            p["oer"] = float(oer_probs[i])
        corrected = broken = 0
        for idx in changed:
            gf = gt_fine.get(idx, set())
            orig_wrong = cand[idx]["category_id"] not in gf
            new_right = cand[idx]["crop_top1_class"] in gf
            if orig_wrong and new_right:
                corrected += 1
            elif (not orig_wrong) and (not new_right):
                broken += 1
        pb2 = defaultdict(list)
        for p in newp:
            pb2[p["image_id"]].append(p)
        kept = []
        for img, pl in pb2.items():
            od = sorted(pl, key=lambda p: -p["oer"]); sup = set()
            for p in od:
                if p["_idx"] in sup:
                    continue
                kept.append(p)
                for q in od:
                    if q["_idx"] == p["_idx"] or q["_idx"] in sup or q["category_id"] != p["category_id"]:
                        continue
                    if compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > 0.5:
                        sup.add(q["_idx"])
        tp = {}; used = defaultdict(set)
        pb3 = defaultdict(list)
        for p in kept:
            pb3[p["image_id"]].append(p)
        for img, gts in formal.boxes.items():
            pl = pb3.get(img, []); od = sorted(pl, key=lambda p: -p["oer"])
            for g in gts:
                cid = int(g["category_id"]); thr = proto.iou_thresholds[proto.category_mapping[cid]]
                bi, bix = 0.0, None
                for p in od:
                    if p["_idx"] in used[img] or p["category_id"] != cid:
                        continue
                    iou = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                    if iou > bi:
                        bi, bix = iou, p
                if bix is not None and bi >= thr:
                    used[img].add(bix["_idx"]); tp[bix["_idx"]] = True
        od2 = sorted(kept, key=lambda p: -p["oer"])
        tp_ = fp = 0; br = {v: 0.0 for v in (0.12, 0.11, 0.10)}
        for p in od2:
            if tp.get(p["_idx"]):
                tp_ += 1
            else:
                fp += 1
            rec = tp_ / n_gt; fdr = fp / (tp_ + fp) if tp_ + fp else 0
            for v in br:
                if fdr <= v:
                    br[v] = max(br[v], rec)
        return br, corrected, broken, len(changed)

    # OER: 14 vs 14+15ctx
    oer_base = train_oer(FEAT_BASE)
    oer_ctx = train_oer(FEAT_BASE + CTX)
    # 路由器: 11 vs 11+15ctx
    router_base, yr = train_router(ROUTER_FEAT)
    router_ctx, _ = train_router(ROUTER_FEAT + CTX)
    print(f"路由器 AUC(11特征)={roc_auc_score(yr, router_base):.4f}  "
          f"(11+15上下文)={roc_auc_score(yr, router_ctx):.4f}")

    # oracle 改类
    br_ora_base, _, _, _ = evaluate(oer_base, "oracle")
    br_ora_ctx, _, _, _ = evaluate(oer_ctx, "oracle")
    print(f"\n[oracle 改类]  OER14    R@FDR=.12={br_ora_base[0.12]:.4f}")
    print(f"[oracle 改类]  OER14+15 R@FDR=.12={br_ora_ctx[0.12]:.4f} (Δ{br_ora_ctx[0.12]-br_ora_base[0.12]:+.4f})")

    # deploy 路由器改类(阈值扫描)
    best_base = None; best_ctx = None
    print(f"\n[deploy 路由器, 阈值扫描]")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        br_b, _, _, _ = evaluate(oer_base, "router", router_base, thr)
        br_c, _, _, _ = evaluate(oer_ctx, "router", router_ctx, thr)
        print(f"  thr={thr}: 基线 R@FDR=.12={br_b[0.12]:.4f} | +上下文={br_c[0.12]:.4f} "
              f"(Δ{br_c[0.12]-br_b[0.12]:+.4f})")
        if best_base is None or br_b[0.12] > best_base[1]:
            best_base = (thr, br_b[0.12])
        if best_ctx is None or br_c[0.12] > best_ctx[1]:
            best_ctx = (thr, br_c[0.12])

    print(f"\n最优 deploy 基线:      thr={best_base[0]} R@FDR=.12={best_base[1]:.4f}")
    print(f"最优 deploy +上下文:   thr={best_ctx[0]} R@FDR=.12={best_ctx[1]:.4f} "
          f"(Δ{best_ctx[1]-best_base[1]:+.4f})")

    Path(args.output).write_text(json.dumps({
        "oracle_base": br_ora_base[0.12], "oracle_ctx": br_ora_ctx[0.12],
        "deploy_base": {"thr": best_base[0], "r": best_base[1]},
        "deploy_ctx": {"thr": best_ctx[0], "r": best_ctx[1]},
        "router_auc_base": float(roc_auc_score(yr, router_base)),
        "router_auc_ctx": float(roc_auc_score(yr, router_ctx)),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")


if __name__ == "__main__":
    main()
