"""C4: 边界闭合/结构延续证据(方案5 §六.2)—— 从空间 grid FPN 计算。

核心: MS/QHS/FSC 的背景混淆是"跑道线/建筑长边/规则排列"等连续细长结构,
它们穿过 bbox 边界继续延伸, 而真实目标通常有闭合轮廓。

证据(每尺度 s, core/ring/region 各 5x5 grid):
1. closure: core 中心 3x3 均值 / 外圈 16 均值(闭合对象中心强边缘弱);
2. ring_cont: core 与 ring 的相似度(结构穿过边界时 core≈ring, 相似度高);
3. core_region_diff: core 与 region 的差异(对象 vs 环境);
4. core_energy: core 激活强度;
5. ring_homogeneity: ring 空间均匀性(线性结构 vs 随机纹理)。
"""
import argparse
import csv
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

# grid 布局: 每尺度 core/ring/region 各 25(5x5)
GRID_LAYOUT = {
    "core_p3": (0, 25), "ring_p3": (25, 50), "region_p3": (50, 75),
    "core_p4": (75, 100), "ring_p4": (100, 125), "region_p4": (125, 150),
    "core_p5": (150, 175), "ring_p5": (175, 200), "region_p5": (200, 225),
}
SCALES = [("p3", "core_p3", "ring_p3", "region_p3"),
          ("p4", "core_p4", "ring_p4", "region_p4"),
          ("p5", "core_p5", "ring_p5", "region_p5")]


def grid_to_55(v):
    return v.reshape(5, 5)


def build_structure_evidence(feat):
    """feat: [N, 225] → 结构证据 [N, 15]。"""
    feat = feat.astype(np.float32)
    ev = []
    for s, ck, rk, gk in SCALES:
        core = feat[:, GRID_LAYOUT[ck][0]:GRID_LAYOUT[ck][1]]   # [N, 25]
        ring = feat[:, GRID_LAYOUT[rk][0]:GRID_LAYOUT[rk][1]]
        region = feat[:, GRID_LAYOUT[gk][0]:GRID_LAYOUT[gk][1]]
        nc = np.linalg.norm(core, axis=1) + 1e-8
        nr = np.linalg.norm(ring, axis=1) + 1e-8
        # closure: 中心 3x3 均值 / 外圈 16 均值
        c55 = core.reshape(-1, 5, 5)
        center = c55[:, 1:4, 1:4].reshape(-1, 9).mean(axis=1)
        mask = np.ones((5, 5), dtype=bool)
        mask[1:4, 1:4] = False
        border = c55[:, mask].reshape(-1, 16).mean(axis=1)
        closure = center / (border + 1e-8)
        # ring_cont: core vs ring 相似度(结构穿过边界时高)
        ring_cont = (core * ring).sum(axis=1) / (nc * nr)
        # core_region_diff
        region_diff = np.linalg.norm(core - region, axis=1) / nc
        # core_energy
        core_energy = np.log1p(nc)
        # ring_homogeneity: ring 空间 std(线性结构低, 随机纹理高)
        r55 = ring.reshape(-1, 5, 5)
        ring_homo = r55.std(axis=(1, 2))
        ev += [closure, ring_cont, region_diff, core_energy, ring_homo]
    return np.stack(ev, axis=1)  # [N, 15]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--grid-dir", required=True, help="含 y5_grid_feat_fold{0,1,2}.npz")
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

    # grid 特征
    uid2feat = {}
    for f in (0, 1, 2):
        z = np.load(Path(args.grid_dir) / f"y5_grid_feat_fold{f}.npz", allow_pickle=True)
        for uid, feat in zip(z["uids"], z["feats"]):
            uid2feat[str(uid)] = feat
    print(f"grid 覆盖: {sum(1 for u in nodes['proposal_uid'] if str(u) in uid2feat)}/{len(nodes)}")
    grid_mat = np.stack([uid2feat[str(u)] for u in nodes["proposal_uid"]], axis=0)
    ev = build_structure_evidence(grid_mat)

    # 判别力
    is_tp = nodes["is_valid"].to_numpy() == 1
    is_fpbg = (nodes["is_valid"].to_numpy() == 0) & (nodes["fine_correct"].to_numpy() == 0)
    ev_names = []
    for s in ("p3", "p4", "p5"):
        ev_names += [f"{s}_closure", f"{s}_ring_cont", f"{s}_region_diff",
                     f"{s}_core_energy", f"{s}_ring_homo"]
    print(f"\n=== 结构证据判别力(TP={is_tp.sum()}, FP_BG={is_fpbg.sum()}) ===")
    seps = []
    for j, nm in enumerate(ev_names):
        mt = ev[is_tp, j].mean()
        mb = ev[is_fpbg, j].mean()
        seps.append((abs(mt - mb), nm, mt, mb))
    for sep, nm, mt, mb in sorted(seps, reverse=True)[:10]:
        print(f"  {nm:16s} TP={mt:8.3f} FP_BG={mb:8.3f} |Δ|={sep:.3f}")

    # OER 训练
    def train(extra):
        X = np.hstack([nodes[FEAT_BASE].to_numpy(dtype=float), extra]) if extra is not None \
            else nodes[FEAT_BASE].to_numpy(dtype=float)
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

    probs_base = train(None)
    probs_ev = train(ev)

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
    print(f"\n[OER 14特征]         R@FDR=.12={br_base[0.12]:.4f} .11={br_base[0.11]:.4f}")
    print(f"[OER 14+结构证据]    R@FDR=.12={br_ev[0.12]:.4f} .11={br_ev[0.11]:.4f}")
    print(f"Δ(结构证据) @FDR=.12: {br_ev[0.12]-br_base[0.12]:+.4f}")

    Path(args.output).write_text(json.dumps({
        "base": {"r_0.12": br_base[0.12], "r_0.11": br_base[0.11]},
        "structure": {"r_0.12": br_ev[0.12], "r_0.11": br_ev[0.11]},
        "top_separations": [(nm, float(mt), float(mb)) for sep, nm, mt, mb
                            in sorted(seps, reverse=True)[:10]],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")


if __name__ == "__main__":
    main()
