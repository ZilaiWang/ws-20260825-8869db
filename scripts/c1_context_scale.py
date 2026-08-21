"""C1: context 尺度搜索(方案5 §六 批次1)。

核心问题: tight(1×) / ring(1.5×) / region(2×) / scene(4×) 哪个空间尺度
对 MS/QHS/FSC 弱类的 TP vs FP_BG 判别力最强?

FPN 特征(3200维)已按 core/ring/region/scene × P3/P4/P5 分区, 直接对各尺度
做 PCA+逻辑回归判别力诊断(TP vs FP_BG), 并按弱类拆分看 AUC。随后把判别力
最强的尺度做 PCA 降维加入 OER, 验证固定风险前沿。

纯本地(基于 fpn-feats + nodes + d4 + oto)。
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
# 每个尺度在三个感受野上的区块
SCALE_BLOCKS = {
    "core(1x)": ["core_p3", "core_p4", "core_p5"],
    "ring(1.5x)": ["ring_p3", "ring_p4", "ring_p5"],
    "region(2x)": ["region_p3", "region_p4", "region_p5"],
    "scene(4x)": ["scene"],
}
WEAK_CLASSES = {"MS": 3, "QHS": 2, "FSC": 24, "SU-24": 23, "TU-160": 9, "TU-22": 15}

FEAT_BASE = [
    "y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
    "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density",
    "d4_support", "has_oto",
]


def slice_scale(feat, scale):
    blocks = SCALE_BLOCKS[scale]
    parts = [feat[:, LAYOUT[b][0]:LAYOUT[b][1]] for b in blocks]
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
    ap.add_argument("--pca-dim", type=int, default=32)
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

    # 加载 FPN 特征
    uid2feat = {}
    for f in (0, 1, 2):
        z = np.load(Path(args.fpn_dir) / f"y5_fpn_feat_fold{f}.npz", allow_pickle=True)
        for uid, feat in zip(z["uids"], z["feats"]):
            uid2feat[str(uid)] = feat
    feat_mat = np.stack([uid2feat[str(u)] for u in nodes["proposal_uid"]],
                        axis=0).astype(np.float32)

    is_tp = nodes["is_valid"].to_numpy() == 1
    is_fpbg = (nodes["is_valid"].to_numpy() == 0) & (nodes["fine_correct"].to_numpy() == 0)
    # 位置匹配 GT 的细类(crop_top1_class 是改类目标)
    pred_cls = nodes["crop_top1_class"].to_numpy()

    # ============ 尺度判别力诊断: TP vs FP_BG ============
    # 用 GT 粗类把 TP 限定到特定弱类(诊断"该类的真目标 vs 该类的背景误检")
    # 这里用 crop_top1_class 作为"预测类"代理(FP_BG 无 GT 类, 只有预测类)
    print("=" * 78)
    print("C1 尺度判别力诊断: 各尺度 FPN 特征区分 TP vs FP_BG 的 AUC")
    print("=" * 78)

    # 先对整体(所有类)做判别力
    tp_idx = np.where(is_tp)[0]
    fpbg_idx = np.where(is_fpbg)[0]
    # 采样平衡
    rng = np.random.RandomState(0)
    n_use = min(len(tp_idx), len(fpbg_idx), 8000)
    tp_s = rng.choice(tp_idx, n_use, replace=False)
    fpbg_s = rng.choice(fpbg_idx, n_use, replace=False)
    idx = np.concatenate([tp_s, fpbg_s])
    y = np.concatenate([np.ones(n_use), np.zeros(n_use)])

    scale_auc = {}
    for scale in SCALE_BLOCKS:
        X = slice_scale(feat_mat, scale)[idx]
        # 标准化 + PCA + 逻辑回归
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        pca = PCA(n_components=args.pca_dim, random_state=0).fit(Xs)
        Xp = pca.transform(Xs)
        lr = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        lr.fit(Xp, y)
        auc = roc_auc_score(y, lr.predict_proba(Xp)[:, 1])
        scale_auc[scale] = auc
        print(f"  {scale:14s} 整体 AUC={auc:.4f}  (pca {args.pca_dim}维)")

    # 按弱类拆: 该类 TP vs 该类 FP_BG(crop_top1_class == c)
    print("\n  按弱类拆分(TP 用 GT 类, FP_BG 用预测类 crop_top1_class):")
    class_auc = {}
    for nm, cid in WEAK_CLASSES.items():
        # TP: 位置匹配 GT 且 GT 粗类对应 cid —— 简化: 用 is_valid 且 crop_top1_class==cid
        # 更准确: TP 是"最终改类后正确", 这里近似用 crop_top1_class==cid 的 TP
        c_tp = np.where(is_tp & (pred_cls == cid))[0]
        c_fpbg = np.where(is_fpbg & (pred_cls == cid))[0]
        if len(c_tp) < 20 or len(c_fpbg) < 20:
            print(f"    {nm:8s} 样本不足(TP={len(c_tp)}, FP_BG={len(c_fpbg)})")
            continue
        n_use_c = min(len(c_tp), len(c_fpbg), 1500)
        tp_s = rng.choice(c_tp, n_use_c, replace=False)
        fpbg_s = rng.choice(c_fpbg, n_use_c, replace=False)
        idxc = np.concatenate([tp_s, fpbg_s])
        yc = np.concatenate([np.ones(n_use_c), np.zeros(n_use_c)])
        row = {}
        for scale in SCALE_BLOCKS:
            X = slice_scale(feat_mat, scale)[idxc]
            sc = StandardScaler().fit(X)
            Xs = sc.transform(X)
            pca = PCA(n_components=args.pca_dim, random_state=0).fit(Xs)
            Xp = pca.transform(Xs)
            lr = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
            lr.fit(Xp, yc)
            auc = roc_auc_score(yc, lr.predict_proba(Xp)[:, 1])
            row[scale] = auc
        class_auc[nm] = row
        best_scale = max(row, key=row.get)
        print(f"    {nm:8s} (TP={len(c_tp)},FP_BG={len(c_fpbg)}): " +
              "  ".join(f"{s.split('(')[0]}={a:.3f}" for s, a in row.items()) +
              f"   → 最优 {best_scale}")

    # ============ OER 验证: 各尺度 PCA 特征加入 OER ============
    print("\n" + "=" * 78)
    print("C1 OER 验证: 14特征 + 各尺度 PCA 特征 → 固定风险前沿")
    print("=" * 78)

    folds = nodes["fold"].to_numpy()

    def scale_pca_feats(scale, dim):
        """cross-fit 尺度 PCA 特征(避免泄漏)。"""
        X = slice_scale(feat_mat, scale)
        out = np.zeros((len(nodes), dim), dtype=np.float32)
        for held in (0, 1, 2):
            tr = folds != held
            va = folds == held
            sc = StandardScaler().fit(X[tr])
            pca = PCA(n_components=dim, random_state=0).fit(sc.transform(X[tr]))
            out[va] = pca.transform(sc.transform(X[va]))
        return out

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

    probs_base = train_oer(None)
    br_base = evaluate(probs_base)
    print(f"  [OER 14特征]        R@FDR=.12={br_base[0.12]:.4f} .11={br_base[0.11]:.4f}")

    oer_scale = {}
    for scale in SCALE_BLOCKS:
        extra = scale_pca_feats(scale, args.pca_dim)
        probs = train_oer(extra)
        br = evaluate(probs)
        d = br[0.12] - br_base[0.12]
        oer_scale[scale] = {"r_0.12": br[0.12], "r_0.11": br[0.11], "delta": d}
        print(f"  [OER 14+{scale:14s}] R@FDR=.12={br[0.12]:.4f} .11={br[0.11]:.4f} (Δ{d:+.4f})")

    out = {
        "scale_auc_overall": {s: float(a) for s, a in scale_auc.items()},
        "scale_auc_by_class": {nm: {s: float(a) for s, a in row.items()}
                               for nm, row in class_auc.items()},
        "oer": {
            "base": {"r_0.12": br_base[0.12], "r_0.11": br_base[0.11]},
            "per_scale": {s: {"r_0.12": v["r_0.12"], "r_0.11": v["r_0.11"],
                              "delta": v["delta"]} for s, v in oer_scale.items()},
        },
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {args.output}")


if __name__ == "__main__":
    main()
