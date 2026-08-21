"""C2: Core/Ring/Region 分离 vs 拼接(方案5 §六 批次1)。

核心问题: 单纯把 core+ring+region 拼接成一个整体特征, 是否不如"分区处理"
(各自独立降维后再组合)? 对比:
  - baseline: 14 特征 OER
  - concat:  14 + PCA(concat(core,ring,region), 48)   # 单纯拼接
  - split:   14 + [PCA(core,16) | PCA(ring,16) | PCA(region,16)]  # 分区
所有 PCA 用 cross-fit 避免泄漏。

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
REGION_BLOCKS = {
    "core": ["core_p3", "core_p4", "core_p5"],
    "ring": ["ring_p3", "ring_p4", "ring_p5"],
    "region": ["region_p3", "region_p4", "region_p5"],
}

FEAT_BASE = [
    "y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
    "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density",
    "d4_support", "has_oto",
]


def slice_region(feat, region):
    blocks = REGION_BLOCKS[region]
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

    uid2feat = {}
    for f in (0, 1, 2):
        z = np.load(Path(args.fpn_dir) / f"y5_fpn_feat_fold{f}.npz", allow_pickle=True)
        for uid, feat in zip(z["uids"], z["feats"]):
            uid2feat[str(uid)] = feat
    feat_mat = np.stack([uid2feat[str(u)] for u in nodes["proposal_uid"]],
                        axis=0).astype(np.float32)

    folds = nodes["fold"].to_numpy()

    def crossfit_pca(X, dim):
        out = np.zeros((len(nodes), dim), dtype=np.float32)
        for held in (0, 1, 2):
            tr = folds != held
            va = folds == held
            sc = StandardScaler().fit(X[tr])
            pca = PCA(n_components=dim, random_state=0).fit(sc.transform(X[tr]))
            out[va] = pca.transform(sc.transform(X[va]))
        return out

    core = slice_region(feat_mat, "core")
    ring = slice_region(feat_mat, "ring")
    region = slice_region(feat_mat, "region")
    concat_all = np.concatenate([core, ring, region], axis=1)

    # 拼接版: PCA(concat, 48)
    concat_feats = crossfit_pca(concat_all, 48)
    # 分区版: PCA(core,16)|PCA(ring,16)|PCA(region,16)
    split_feats = np.hstack([crossfit_pca(core, 16), crossfit_pca(ring, 16),
                             crossfit_pca(region, 16)])

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
    probs_concat = train_oer(concat_feats)
    probs_split = train_oer(split_feats)

    br_base = evaluate(probs_base)
    br_concat = evaluate(probs_concat)
    br_split = evaluate(probs_split)

    print(f"[OER 14特征]            R@FDR=.12={br_base[0.12]:.4f} .11={br_base[0.11]:.4f}")
    print(f"[OER 14+拼接(48)]       R@FDR=.12={br_concat[0.12]:.4f} .11={br_concat[0.11]:.4f} "
          f"(Δ{br_concat[0.12]-br_base[0.12]:+.4f})")
    print(f"[OER 14+分区(16|16|16)] R@FDR=.12={br_split[0.12]:.4f} .11={br_split[0.11]:.4f} "
          f"(Δ{br_split[0.12]-br_base[0.12]:+.4f})")
    print(f"分区 vs 拼接: Δ={br_split[0.12]-br_concat[0.12]:+.4f}")

    Path(args.output).write_text(json.dumps({
        "base": {"r_0.12": br_base[0.12], "r_0.11": br_base[0.11]},
        "concat": {"r_0.12": br_concat[0.12], "r_0.11": br_concat[0.11]},
        "split": {"r_0.12": br_split[0.12], "r_0.11": br_split[0.11]},
        "split_minus_concat": br_split[0.12] - br_concat[0.12],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")


if __name__ == "__main__":
    main()
