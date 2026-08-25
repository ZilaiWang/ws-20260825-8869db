"""C5: 支持面兼容(方案5 §6.3 批次1)。

核心: 候选周围环境(region 2×)是否支持该候选的粗类/细类。无支持面标注时,
用 region 特征无监督聚类得到 K 个"环境原型"(水域/跑道/道路/建筑等), 在训练折
统计每个原型的 TP 粗类分布, 构造软兼容证据:
  - compat_tp_rate:   候选所在原型中, 与候选同粗类的 TP 占比(该环境对该类的支持度)
  - compat_entropy:   候选所在原型 TP 粗类分布的熵(低=环境专一)
  - compat_coarse_rank: 候选粗类在该原型 TP 粗类分布中的排名

重点验证 C1 诊断发现的弱类(QHS/FSC/SU-24 最优尺度=region/环境)。

纯本地(基于 fpn-feats region + nodes + d4 + oto)。
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

C3, C4, C5 = 128, 256, 512
LAYOUT = {
    "core_p3": (0, C3), "ring_p3": (C3, 2 * C3), "region_p3": (2 * C3, 3 * C3),
    "core_p4": (3 * C3, 3 * C3 + C4), "ring_p4": (3 * C3 + C4, 3 * C3 + 2 * C4),
    "region_p4": (3 * C3 + 2 * C4, 3 * C3 + 3 * C4),
    "core_p5": (3 * C3 + 3 * C4, 3 * C3 + 3 * C4 + C5),
    "ring_p5": (3 * C3 + 3 * C4 + C5, 3 * C3 + 3 * C4 + 2 * C5),
    "region_p5": (3 * C3 + 3 * C4 + 2 * C5, 3 * C3 + 3 * C4 + 3 * C5),
    "scene": (3 * C3 + 3 * C4 + 3 * C5, 3 * C3 + 3 * C4 + 3 * C5 + C5),
}

FEAT_BASE = [
    "y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
    "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density",
    "d4_support", "has_oto",
]
COARSE_OF = {2: "ship", 3: "ship", 0: "ship", 1: "ship", 24: "vehicle",
             4: "aircraft", 5: "aircraft", 6: "aircraft", 7: "aircraft", 8: "aircraft",
             9: "aircraft", 10: "aircraft", 11: "aircraft", 12: "aircraft", 13: "aircraft",
             14: "aircraft", 15: "aircraft", 16: "aircraft", 17: "aircraft", 18: "aircraft",
             19: "aircraft", 20: "aircraft", 21: "aircraft", 22: "aircraft", 23: "aircraft"}


def slice_region(feat):
    parts = [feat[:, LAYOUT[b][0]:LAYOUT[b][1]]
             for b in ("region_p3", "region_p4", "region_p5")]
    return np.concatenate(parts, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--fpn-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--k-protos", type=int, default=16, help="环境原型数")
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    names = {}
    with open(args.formal_crop_manifest, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            names[int(r["class_id"])] = r["class_name"]
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
    nodes["coarse"] = nodes["crop_top1_class"].map(COARSE_OF)

    uid2feat = {}
    for f in (0, 1, 2):
        z = np.load(Path(args.fpn_dir) / f"y5_fpn_feat_fold{f}.npz", allow_pickle=True)
        for uid, feat in zip(z["uids"], z["feats"]):
            uid2feat[str(uid)] = feat
    feat_mat = np.stack([uid2feat[str(u)] for u in nodes["proposal_uid"]],
                        axis=0).astype(np.float32)
    region = slice_region(feat_mat)

    folds = nodes["fold"].to_numpy()
    is_tp = nodes["is_valid"].to_numpy() == 1
    is_fpbg = (nodes["is_valid"].to_numpy() == 0) & (nodes["fine_correct"].to_numpy() == 0)
    coarse = nodes["coarse"].to_numpy()

    # cross-fit 环境原型: 训练折 fit PCA+KMeans, 验证折 assign
    proto_id = np.zeros(len(nodes), dtype=int)
    for held in (0, 1, 2):
        tr = folds != held
        va = folds == held
        sc = StandardScaler().fit(region[tr])
        pca = PCA(n_components=32, random_state=0).fit(sc.transform(region[tr]))
        km = KMeans(n_clusters=args.k_protos, random_state=42, n_init=3)
        km.fit(pca.transform(sc.transform(region[tr])))
        proto_id[va] = km.predict(pca.transform(sc.transform(region[va])))

    # 对每个原型统计训练折的 TP 粗类分布
    compat_tp_rate = np.zeros(len(nodes))
    compat_entropy = np.zeros(len(nodes))
    compat_rank = np.zeros(len(nodes))
    proto_stats = {}
    for held in (0, 1, 2):
        tr = folds != held
        va = folds == held
        # 训练折每原型 TP 粗类计数
        for k in range(args.k_protos):
            mask = (proto_id[tr] == k)
            tp_mask = mask & is_tp[tr]
            n_proto = int(mask.sum())
            if n_proto == 0:
                continue
            tp_coarse = defaultdict(int)
            for c in coarse[tr][tp_mask]:
                tp_coarse[c] += 1
            total_tp = max(int(tp_mask.sum()), 1)
            # 该原型 TP 粗类分布(归一)
            dist = {c: n / total_tp for c, n in tp_coarse.items()}
            ent = -sum(p * np.log(p + 1e-9) for p in dist.values())
            # 应用到验证折该原型的候选
            va_mask = va & (proto_id == k)
            for i in np.where(va_mask)[0]:
                c = coarse[i]
                compat_tp_rate[i] = dist.get(c, 0.0)
                compat_entropy[i] = ent
                # rank: c 在 dist 中的排名(降序)
                ranks = {cc: r for r, (cc, _) in enumerate(
                    sorted(dist.items(), key=lambda x: -x[1]))}
                compat_rank[i] = ranks.get(c, len(dist))
            if held == 0:
                proto_stats[str(k)] = {"n": n_proto, "n_tp": int(tp_mask.sum()),
                                       "top_coarse": sorted(dist.items(), key=lambda x: -x[1])[:3]}

    # 诊断: TP vs FP_BG 的 compat 分布
    print(f"=== 支持面兼容证据判别力(K={args.k_protos}) ===")
    for nm, arr in [("compat_tp_rate", compat_tp_rate), ("compat_entropy", compat_entropy),
                    ("compat_rank", compat_rank)]:
        mt = arr[is_tp].mean()
        mb = arr[is_fpbg].mean()
        print(f"  {nm:16s} TP={mt:.4f} FP_BG={mb:.4f} |Δ|={abs(mt-mb):.4f}")

    # 重点: 弱类(QHS/FSC/MS) 的 TP vs 该类 FP_BG
    print("\n  弱类拆分(TP 该类 vs FP_BG 预测为该类):")
    for cid, nm in [(3, "MS"), (2, "QHS"), (24, "FSC"), (23, "SU-24")]:
        c_tp = is_tp & (nodes["crop_top1_class"].to_numpy() == cid)
        c_fpbg = is_fpbg & (nodes["crop_top1_class"].to_numpy() == cid)
        if c_tp.sum() < 20 or c_fpbg.sum() < 20:
            continue
        mt = compat_tp_rate[c_tp].mean()
        mb = compat_tp_rate[c_fpbg].mean()
        print(f"    {nm:8s} compat_tp_rate: TP={mt:.4f} FP_BG={mb:.4f} Δ={mt-mb:+.4f}")

    # OER 验证: 14 + compat 特征
    def train_oer(extra=None):
        X = nodes[FEAT_BASE].to_numpy(dtype=float)
        if extra is not None:
            X = np.hstack([X, extra])
        y = nodes["is_valid"].to_numpy(dtype=float)
        probs = np.zeros(len(nodes))
        for held in (0, 1, 2):
            tr = np.where(folds != held)[0]
            va = np.where(folds == held)[0]
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                                 l2_regularization=1.0, random_state=2026 + held)
            clf.fit(X[tr], y[tr])
            probs[va] = clf.predict_proba(X[va])[:, 1]
        return probs

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

    compat_feats = np.stack([compat_tp_rate, compat_entropy, compat_rank], axis=1)
    probs_base = train_oer(None)
    probs_compat = train_oer(compat_feats)

    br_base = evaluate(probs_base)
    br_compat = evaluate(probs_compat)
    print(f"\n[OER 14特征]        R@FDR=.12={br_base[0.12]:.4f} .11={br_base[0.11]:.4f}")
    print(f"[OER 14+支持面兼容] R@FDR=.12={br_compat[0.12]:.4f} .11={br_compat[0.11]:.4f} "
          f"(Δ{br_compat[0.12]-br_base[0.12]:+.4f})")

    Path(args.output).write_text(json.dumps({
        "base": {"r_0.12": br_base[0.12], "r_0.11": br_base[0.11]},
        "compat": {"r_0.12": br_compat[0.12], "r_0.11": br_compat[0.11]},
        "compat_tp_rate_tp_mean": float(compat_tp_rate[is_tp].mean()),
        "compat_tp_rate_fpbg_mean": float(compat_tp_rate[is_fpbg].mean()),
        "proto_stats": proto_stats,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")


if __name__ == "__main__":
    main()
