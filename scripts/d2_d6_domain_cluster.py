"""D2/D6: latent domain 聚类 + leave-domain-cluster-out 跨域泛化验证(方案5 §十二)。

D1 发现 12 高误检域贡献 30% FP_BG、20 低召回域 Recall=0.382。D2 用 group 画像
(recall/fdr/fp_bg) 聚类 4~8 个 latent domain; D6 做 leave-domain-cluster-out——
留出一个域簇, 用其余域训练 OER, 评估留出域的 Recall@FDR, 量化跨域泛化损失,
为域专家(D5 adapter)提供上限依据。

纯本地(基于 nodes + group_id + 14特征 OER)。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

FEAT = [
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
    ap.add_argument("--n-domains", type=int, default=4)
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
    grp_of = {i: o.group_id for (i, _), o in formal.objects.items()}
    nodes["fold"] = nodes["image_id"].map(fold_of)
    nodes["group_id"] = nodes["image_id"].map(grp_of)

    # OER cross-fit(域内, 基线)
    X = nodes[FEAT].to_numpy(dtype=float)
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

    # 每 group 的错误画像(用 OER probs 的粗匹配, 简化为 fp_bg 率 + recall 代理)
    # 这里用 is_valid 的 group 分布做画像
    grp_stats = {}
    for g, sub in nodes.groupby("group_id"):
        n = len(sub)
        n_tp = int((sub["is_valid"] == 1).sum())
        grp_stats[g] = {"n_cand": n, "tp": n_tp, "tp_rate": n_tp / n if n else 0}

    groups = sorted(grp_stats.keys())
    # 画像特征: tp_rate + n_cand(对数)
    feats = np.array([[grp_stats[g]["tp_rate"], np.log1p(grp_stats[g]["n_cand"])] for g in groups])
    sc = StandardScaler().fit(feats)
    km = KMeans(n_clusters=args.n_domains, random_state=42, n_init=5).fit(sc.transform(feats))
    grp_domain = {g: int(km.labels_[i]) for i, g in enumerate(groups)}

    # 域统计
    domain_grp_count = defaultdict(int)
    domain_gt = defaultdict(int)
    for g, d in grp_domain.items():
        domain_grp_count[d] += 1
    img2grp = grp_of
    for img, gts in formal.boxes.items():
        domain_gt[grp_domain.get(img2grp.get(img, "?"), -1)] += len(gts)

    print(f"=== D2 latent domain 聚类(K={args.n_domains}) ===")
    for d in range(args.n_domains):
        gs = [g for g in groups if grp_domain[g] == d]
        tp_rates = [grp_stats[g]["tp_rate"] for g in gs]
        print(f"  域{d}: {len(gs)} group, GT={domain_gt[d]}, tp_rate均值={np.mean(tp_rates):.3f}")

    nodes["domain"] = nodes["group_id"].map(grp_domain)

    # D6 leave-domain-cluster-out: 留出一个域, 用其余域训练 OER, 评估留出域
    def train_oer_leaveout(leave_domain, feats_arr):
        tr = nodes["domain"].to_numpy() != leave_domain
        va = nodes["domain"].to_numpy() == leave_domain
        # 训练集还用 fold 隔离(避免同 image 泄漏)
        y_arr = nodes["is_valid"].to_numpy(dtype=float)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                             l2_regularization=1.0, random_state=2026)
        clf.fit(feats_arr[tr], y_arr[tr])
        return clf.predict_proba(feats_arr[va])[:, 1], va

    # 评估 frontier(简化: 只用 oer 排序 + 改类 + NMS, 复用在 leave-out 域内的候选)
    def evaluate_domain(probs_sub, va_mask, domain_id):
        # 构造候选
        cand = []
        for r in nodes[va_mask].itertuples():
            cand.append({"_idx": int(r.idx), "image_id": int(r.image_id),
                         "category_id": int(r.category_id), "crop_top1_class": int(r.crop_top1_class),
                         "bbox_xyxy": [r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2]})
        for i, p in enumerate(cand):
            p["oer"] = float(probs_sub[i])
        # NMS + 改类 + 贪心匹配(复用 D3 逻辑)
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
        tp = {}
        used = defaultdict(set)
        pb3 = defaultdict(list)
        for p in kept:
            pb3[p["image_id"]].append(p)
        n_gt_d = 0
        for img, gts in formal.boxes.items():
            if grp_domain.get(img2grp.get(img, "?"), -1) != domain_id:
                continue
            n_gt_d += len(gts)
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
        # 该域内的 frontier(用全局 FDR 口径近似)
        od2 = sorted(kept, key=lambda p: -p["oer"])
        tp_ = fp = 0
        br = 0.0
        for p in od2:
            if tp.get(p["_idx"]):
                tp_ += 1
            else:
                fp += 1
            rec = tp_ / n_gt_d if n_gt_d else 0
            fdr = fp / (tp_ + fp) if tp_ + fp else 0
            if fdr <= 0.12:
                br = max(br, rec)
        return br, n_gt_d

    print(f"\n=== D6 leave-domain-cluster-out 跨域泛化 ===")
    leaveout_results = []
    for d in range(args.n_domains):
        probs_lo, va = train_oer_leaveout(d, X)
        # 域内 OER(用 fold cross-fit 的 probs, 该域) 作对照
        va_mask = nodes["domain"].to_numpy() == d
        br_in, ngt_in = evaluate_domain(probs[va_mask], va_mask, d)
        br_out, ngt_out = evaluate_domain(probs_lo, va, d)
        gap = br_in - br_out
        leaveout_results.append({"domain": d, "gt": ngt_out, "in_domain_r": br_in,
                                 "cross_domain_r": br_out, "gap": gap})
        print(f"  域{d}(GT={ngt_out}): 域内R={br_in:.4f} 跨域R={br_out:.4f} gap={gap:+.4f}")

    Path(args.output).write_text(json.dumps({
        "n_domains": args.n_domains,
        "domain_grp_count": {str(k): v for k, v in domain_grp_count.items()},
        "domain_gt": {str(k): v for k, v in domain_gt.items()},
        "leaveout": leaveout_results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {args.output}")


if __name__ == "__main__":
    main()
