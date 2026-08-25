"""C6: Hard Confounder Atlas(方案5 §七)—— 类别化背景原型。

核心: 当前 FP_BG 是"像 MS/QHS/FSC 的结构化背景"。按预测类(crop_top1_class)对
FP_BG 候选的 FPN 特征聚类得到每类的背景原型 μ_bg(c,k), 候选的混淆证据:
    e_conf = max_k cos(h_i, μ_bg(c,k))
这是"这个候选有多像历史上该类最危险的结构化背景"。

纯本地(基于 FPN 特征 + 标签)。
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

_C3, _C4, _C5 = 128, 256, 512
LAYOUT = {
    "core_p3": (0, _C3), "ring_p3": (_C3, 2 * _C3), "region_p3": (2 * _C3, 3 * _C3),
    "core_p4": (3 * _C3, 3 * _C3 + _C4), "ring_p4": (3 * _C3 + _C4, 3 * _C3 + 2 * _C4),
    "region_p4": (3 * _C3 + 2 * _C4, 3 * _C3 + 3 * _C4),
    "core_p5": (3 * _C3 + 3 * _C4, 3 * _C3 + 3 * _C4 + _C5),
    "ring_p5": (3 * _C3 + 3 * _C4 + _C5, 3 * _C3 + 3 * _C4 + 2 * _C5),
    "region_p5": (3 * _C3 + 3 * _C4 + 2 * _C5, 3 * _C3 + 3 * _C4 + 3 * _C5),
    "scene": (3 * _C3 + 3 * _C4 + 3 * _C5, 3 * _C3 + 3 * _C4 + 3 * _C5 + _C5),
}
_SCALES = [("p3", "core_p3", "ring_p3", "region_p3"),
           ("p4", "core_p4", "ring_p4", "region_p4"),
           ("p5", "core_p5", "ring_p5", "region_p5")]


def _cos(a, b):
    na = np.linalg.norm(a, axis=1) + 1e-8
    nb = np.linalg.norm(b, axis=1) + 1e-8
    return (a * b).sum(axis=1) / (na * nb)


def build_counterfactual(feat):
    feat = feat.astype(np.float32)
    ev = []
    for _, ck, rk, gk in _SCALES:
        core = feat[:, LAYOUT[ck][0]:LAYOUT[ck][1]]
        ring = feat[:, LAYOUT[rk][0]:LAYOUT[rk][1]]
        region = feat[:, LAYOUT[gk][0]:LAYOUT[gk][1]]
        nc = np.linalg.norm(core, axis=1) + 1e-8
        nr = np.linalg.norm(region, axis=1) + 1e-8
        nring = np.linalg.norm(ring, axis=1) + 1e-8
        ev.append((1 - _cos(core, region)))
        ev.append(_cos(core, ring))
        ev.append(nc / nr)
        ev.append(np.log1p(nc))
        ev.append(np.log1p(nring))
        ev.append(np.log1p(nr))
    scene = feat[:, LAYOUT["scene"][0]:LAYOUT["scene"][1]]
    ev.append(scene.mean(axis=1))
    ev.append(scene.std(axis=1))
    for k in range(8):
        ev.append(scene[:, k * 64:(k + 1) * 64].mean(axis=1))
    return np.stack(ev, axis=1)


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
    ap.add_argument("--fpn-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-protos", type=int, default=4, help="每类背景原型数")
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
    feat_mat = np.stack([uid2feat[str(u)] for u in nodes["proposal_uid"]], axis=0).astype(np.float32)
    ev = build_counterfactual(feat_mat)

    # PCA 降维(用于原型聚类)
    sc = StandardScaler().fit(feat_mat)
    feat_scaled = sc.transform(feat_mat)
    pca = PCA(n_components=64).fit(feat_scaled)
    feat_pca = pca.transform(feat_scaled)

    # 标签
    is_tp = nodes["is_valid"].to_numpy() == 1
    is_fpbg = (nodes["is_valid"].to_numpy() == 0) & (nodes["fine_correct"].to_numpy() == 0)
    pred_cls = nodes["crop_top1_class"].to_numpy()

    # 按 fold 构造背景原型(避免泄漏: 用训练折的 FP_BG 建原型, 应用到验证折)
    folds = nodes["fold"].to_numpy()
    conf_ev = np.zeros(len(nodes))
    proto_info = {}
    for held in (0, 1, 2):
        tr = (folds != held)
        va = (folds == held)
        # 训练折的 FP_BG, 按预测类聚类
        for c in np.unique(pred_cls[tr & is_fpbg]):
            cls_fpbg_idx = np.where(tr & is_fpbg & (pred_cls == c))[0]
            if len(cls_fpbg_idx) < args.n_protos:
                continue
            k = min(args.n_protos, len(cls_fpbg_idx))
            km = KMeans(n_clusters=k, random_state=42, n_init=2).fit(feat_pca[cls_fpbg_idx])
            # 验证折候选到该类原型的最大 cos
            va_cls_idx = np.where(va & (pred_cls == c))[0]
            if len(va_cls_idx) == 0:
                continue
            for proto_vec in km.cluster_centers_:
                pv = proto_vec / (np.linalg.norm(proto_vec) + 1e-8)
                sims = (feat_pca[va_cls_idx] * pv).sum(axis=1) / (np.linalg.norm(feat_pca[va_cls_idx], axis=1) + 1e-8)
                conf_ev[va_cls_idx] = np.maximum(conf_ev[va_cls_idx], sims)
            if held == 0 and c in (0, 1, 2, 3):
                proto_info[str(c)] = {"n_fpbg": int(len(cls_fpbg_idx)), "n_protos": int(k)}
    print(f"硬背景原型证据: 覆盖 {int((conf_ev > 0).sum())} 候选")
    print(f"原型信息(示例): {proto_info}")

    # 判别力: confounder 证据在 TP vs FP_BG 上的分布
    print("\n=== 硬背景混淆证据判别力 ===")
    print(f"  TP 均值: {conf_ev[is_tp].mean():.4f} | FP_BG 均值: {conf_ev[is_fpbg].mean():.4f}")
    print(f"  FP_BG 高混淆(>0.5)比例: {(conf_ev[is_fpbg] > 0.5).mean():.4f} | TP 高混淆比例: {(conf_ev[is_tp] > 0.5).mean():.4f}")

    # OER 训练: 14特征 vs 14+反事实 vs 14+反事实+混淆
    def train(extra):
        X = np.hstack([nodes[FEAT_BASE].to_numpy(dtype=float), extra]) if extra is not None \
            else nodes[FEAT_BASE].to_numpy(dtype=float)
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

    probs_base = train(None)
    probs_ev = train(ev)
    probs_conf = train(np.hstack([ev, conf_ev.reshape(-1, 1)]))

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
    br_ev = evaluate(probs_ev)
    br_conf = evaluate(probs_conf)
    print(f"\n[OER 14特征]              R@FDR=.12={br_base[0.12]:.4f} .11={br_base[0.11]:.4f}")
    print(f"[OER 14+反事实]           R@FDR=.12={br_ev[0.12]:.4f} .11={br_ev[0.11]:.4f}")
    print(f"[OER 14+反事实+硬背景]    R@FDR=.12={br_conf[0.12]:.4f} .11={br_conf[0.11]:.4f}")
    print(f"Δ(反事实)={br_ev[0.12]-br_base[0.12]:+.4f}  Δ(+硬背景)={br_conf[0.12]-br_ev[0.12]:+.4f}")

    Path(args.output).write_text(json.dumps({
        "base": {"r_0.12": br_base[0.12], "r_0.11": br_base[0.11]},
        "counterfactual": {"r_0.12": br_ev[0.12], "r_0.11": br_ev[0.11]},
        "confounder": {"r_0.12": br_conf[0.12], "r_0.11": br_conf[0.11]},
        "conf_tp_mean": float(conf_ev[is_tp].mean()),
        "conf_fpbg_mean": float(conf_ev[is_fpbg].mean()),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")


if __name__ == "__main__":
    main()
