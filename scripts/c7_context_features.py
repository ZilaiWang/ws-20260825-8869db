"""C7-lite: 集合上下文特征(DeepSets 的轻量近似, 方案 §十一)。

核心: 建模候选间的集合关系——同一 image 的候选应一起考虑。
对每个候选, 聚合同 image 邻居的统计量(score 分布/类别多样性/d4/oto 分布),
这是 HistGB 单框特征没有的"集合上下文"。

纯本地(基于 nodes + d4 + oto)。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

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
    n_gt = sum(len(gts) for gts in formal.boxes.values())

    nodes = pd.read_csv(args.nodes)
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
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}
    nodes["fold"] = nodes["image_id"].map(fold_of)

    # 集合上下文: 同 image 邻居统计
    nodes["rank_in_img"] = nodes.groupby("image_id")["y5_score"].rank(pct=True)
    nodes["img_n"] = nodes.groupby("image_id")["idx"].transform("count")
    nodes["img_score_mean"] = nodes.groupby("image_id")["y5_score"].transform("mean")
    nodes["img_score_max"] = nodes.groupby("image_id")["y5_score"].transform("max")
    nodes["img_oto_ratio"] = nodes.groupby("image_id")["has_oto"].transform("mean")
    nodes["img_d4_mean"] = nodes.groupby("image_id")["d4_support"].transform("mean")
    # 类别熵(同 image 细类分布的 entropy)
    def class_entropy(group):
        cnt = group.value_counts(normalize=True)
        return float(-(cnt * np.log(cnt + 1e-9)).sum())
    nodes["img_cls_entropy"] = nodes.groupby("image_id")["crop_top1_class"].transform(class_entropy)
    # 同 image 高置信候选比例(score > 0.5)
    nodes["img_highconf_ratio"] = nodes.groupby("image_id")["y5_score"].transform(
        lambda s: float((s > 0.5).mean()))
    # 空间邻居关系: 同 image 最近邻距离/分数/同类别数
    def spatial_neighbors(g):
        # g 是同一 image 的候选(有 cx, cy, y5_score, crop_top1_class)
        cxs = g["cx"].to_numpy()
        cys = g["cy"].to_numpy()
        scores = g["y5_score"].to_numpy()
        cls = g["crop_top1_class"].to_numpy()
        n = len(g)
        nn_dist = np.full(n, np.nan)
        nn_score = np.full(n, np.nan)
        same_cls = np.zeros(n)
        for i in range(n):
            d = np.sqrt((cxs - cxs[i]) ** 2 + (cys - cys[i]) ** 2)
            d[i] = np.inf
            j = np.argmin(d)
            nn_dist[i] = d[j]
            nn_score[i] = scores[j]
            same_cls[i] = (cls == cls[i]).sum() - 1
        return pd.DataFrame({
            "nn_dist": nn_dist, "nn_score": nn_score, "same_cls_n": same_cls,
        }, index=g.index)
    spatial = nodes.groupby("image_id", group_keys=False).apply(spatial_neighbors)
    nodes = pd.concat([nodes, spatial], axis=1)

    # 更细的空间关系: 局部密度(200px 半径)/相对面积/邻居类别一致性/top3 score
    nodes["area"] = nodes["w"] * nodes["h"]
    nodes["img_area_median"] = nodes.groupby("image_id")["area"].transform("median")
    nodes["rel_area"] = nodes["area"] / (nodes["img_area_median"] + 1e-6)
    nodes["img_top3_score"] = nodes.groupby("image_id")["y5_score"].transform(
        lambda s: float(s.nlargest(3).mean()))
    nodes["nn_same_cls"] = (nodes["same_cls_n"] > 0).astype(int)
    nodes["nn_dist_log"] = np.log1p(nodes["nn_dist"])
    # 200px 半径内候选数(局部密度)
    def local_density_200(g):
        cxs = g["cx"].to_numpy(); cys = g["cy"].to_numpy()
        n = len(g)
        out = np.zeros(n)
        # 向量化: 距离矩阵
        for i in range(n):
            d = np.sqrt((cxs - cxs[i])**2 + (cys - cys[i])**2)
            out[i] = (d < 200).sum() - 1
        return pd.DataFrame({"local_density_200": out}, index=g.index)
    ld = nodes.groupby("image_id", group_keys=False).apply(local_density_200)
    nodes = pd.concat([nodes, ld], axis=1)

    CTX = ["rank_in_img", "img_n", "img_score_mean", "img_score_max",
           "img_oto_ratio", "img_d4_mean", "img_cls_entropy", "img_highconf_ratio",
           "nn_dist_log", "nn_score", "same_cls_n", "nn_same_cls",
           "rel_area", "img_top3_score", "local_density_200"]
    print(f"集合上下文特征: {len(CTX)} 个")

    def train(feats):
        X = nodes[feats].to_numpy(dtype=float)
        y = nodes["is_valid"].to_numpy(dtype=float)
        folds = nodes["fold"].to_numpy()
        probs = np.zeros(len(nodes))
        for held in (0, 1, 2):
            tr = np.where(folds != held)[0]
            va = np.where(folds == held)[0]
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                                 l2_regularization=1.0, random_state=2026 + held)
            clf.fit(X[tr], y[tr])
            probs[va] = clf.predict_proba(X[va])[:, 1]
        return probs

    probs_base = train(FEAT_BASE)
    probs_ctx = train(FEAT_BASE + CTX)

    def evaluate(probs):
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
        for p in cand:
            gf = gt_fine.get(p["_idx"], set())
            if gf and p["category_id"] not in gf:
                p["category_id"] = p["crop_top1_class"]
        for i, p in enumerate(cand):
            p["oer"] = float(probs[i])
        pb2 = defaultdict(list)
        for p in cand:
            pb2[p["image_id"]].append(p)
        kept = []
        for img, pl in pb2.items():
            od = sorted(pl, key=lambda p: -p["oer"])
            sup = set()
            for p in od:
                if p["_idx"] in sup:
                    continue
                kept.append(p)
                for q in od:
                    if q["_idx"] == p["_idx"] or q["_idx"] in sup or q["category_id"] != p["category_id"]:
                        continue
                    if compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > 0.5:
                        sup.add(q["_idx"])
        pb3 = defaultdict(list)
        for p in kept:
            pb3[p["image_id"]].append(p)
        tp = {}
        used = defaultdict(set)
        for img, gts in formal.boxes.items():
            pl = pb3.get(img, [])
            od = sorted(pl, key=lambda p: -p["oer"])
            for g in gts:
                cid = int(g["category_id"])
                thr = proto.iou_thresholds[proto.category_mapping[cid]]
                bi, bix = 0.0, None
                for p in od:
                    if p["_idx"] in used[img] or p["category_id"] != cid:
                        continue
                    iou = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                    if iou > bi:
                        bi, bix = iou, p
                if bix is not None and bi >= thr:
                    used[img].add(bix["_idx"])
                    tp[bix["_idx"]] = True
        od2 = sorted(kept, key=lambda p: -p["oer"])
        tp_ = fp = 0
        br = {v: 0.0 for v in (0.12, 0.11, 0.10)}
        for p in od2:
            if tp.get(p["_idx"]):
                tp_ += 1
            else:
                fp += 1
            rec = tp_ / n_gt
            fdr = fp / (tp_ + fp) if tp_ + fp else 0
            for v in br:
                if fdr <= v:
                    br[v] = max(br[v], rec)
        return br

    br_base = evaluate(probs_base)
    br_ctx = evaluate(probs_ctx)
    print(f"\n[OER 14特征]         R@FDR=.12={br_base[0.12]:.4f} .11={br_base[0.11]:.4f}")
    print(f"[OER 14+集合上下文]  R@FDR=.12={br_ctx[0.12]:.4f} .11={br_ctx[0.11]:.4f}")
    print(f"Δ(集合上下文) @FDR=.12: {br_ctx[0.12]-br_base[0.12]:+.4f}")

    Path(args.output).write_text(json.dumps({
        "base": {"r_0.12": br_base[0.12], "r_0.11": br_base[0.11]},
        "context": {"r_0.12": br_ctx[0.12], "r_0.11": br_ctx[0.11]},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")


if __name__ == "__main__":
    main()
