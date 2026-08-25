"""C3/C4/C5/C6: 对象-环境反事实证据 + 硬背景图谱(方案5 §六/§七)。

用 Y5 FPN 特征(Core/Ring/Region/Scene)计算反事实差分证据:
- e_obj: 对象核心贡献(core vs region 差异);
- e_ctx/结构延续: core vs ring 相似度(背景线性结构穿过边界时高);
- 支持面兼容: 环境特征强度;
- 硬背景原型: 按预测类聚类 FP_BG 的 FPN 特征, 算候选到原型的相似度。

先做判别力诊断(TP vs FP_BG vs FP_CLS), 再训练 OER(14+新证据)验证 frontier。
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

# FPN 特征布局(C3=128, C4=256, C5=512)
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
SCALES = [("p3", "core_p3", "ring_p3", "region_p3"),
          ("p4", "core_p4", "ring_p4", "region_p4"),
          ("p5", "core_p5", "ring_p5", "region_p5")]


def cos(a, b):
    na = np.linalg.norm(a, axis=1) + 1e-8
    nb = np.linalg.norm(b, axis=1) + 1e-8
    return (a * b).sum(axis=1) / (na * nb)


def build_counterfactual(feat):
    """feat: [N, 3200] float16 → 反事实证据矩阵 [N, D] float32。"""
    feat = feat.astype(np.float32)
    ev = []
    for _, ck, rk, gk in SCALES:
        core = feat[:, LAYOUT[ck][0]:LAYOUT[ck][1]]
        ring = feat[:, LAYOUT[rk][0]:LAYOUT[rk][1]]
        region = feat[:, LAYOUT[gk][0]:LAYOUT[gk][1]]
        nc = np.linalg.norm(core, axis=1) + 1e-8
        nr = np.linalg.norm(region, axis=1) + 1e-8
        nring = np.linalg.norm(ring, axis=1) + 1e-8
        ev.append((1 - cos(core, region)))          # obj_contrast: 对象 vs 环境差异
        ev.append(cos(core, ring))                  # ring_cont: 结构延续(高=穿过边界)
        ev.append(nc / nr)                          # core_region_ratio
        ev.append(np.log1p(nc))                     # log core norm
        ev.append(np.log1p(nring))                  # log ring norm
        ev.append(np.log1p(nr))                     # log region norm
    scene = feat[:, LAYOUT["scene"][0]:LAYOUT["scene"][1]]
    ev.append(scene.mean(axis=1))                    # scene 全局均值
    ev.append(scene.std(axis=1))                     # scene 全局 std
    # scene 分块(8 块均值)
    for k in range(8):
        ev.append(scene[:, k * 64:(k + 1) * 64].mean(axis=1))
    return np.stack(ev, axis=1)  # [N, 18 + 2 + 8 = 28]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--fpn-dir", required=True, help="含 y5_fpn_feat_fold{0,1,2}.npz")
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--folds", default="0,1,2")
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

    # 加载 FPN 特征(按 uid 对齐)
    use_folds = [int(x) for x in args.folds.split(",")]
    uid2feat = {}
    for f in use_folds:
        z = np.load(Path(args.fpn_dir) / f"y5_fpn_feat_fold{f}.npz", allow_pickle=True)
        for uid, feat in zip(z["uids"], z["feats"]):
            uid2feat[str(uid)] = feat
    print(f"FPN 特征覆盖候选: {sum(1 for u in nodes['proposal_uid'] if str(u) in uid2feat)}/{len(nodes)}")
    feat_mat = np.stack([uid2feat[str(u)] for u in nodes["proposal_uid"]], axis=0)
    ev = build_counterfactual(feat_mat)
    print(f"反事实证据维度: {ev.shape[1]}")

    # 判别力诊断
    is_tp = nodes["is_valid"].to_numpy() == 1
    is_fpbg = (nodes["is_valid"].to_numpy() == 0) & (nodes["fine_correct"].to_numpy() == 0)
    is_fpcls = (nodes["is_valid"].to_numpy() == 0) & (nodes["fine_correct"].to_numpy() == 1)
    ev_names = []
    for s in ("p3", "p4", "p5"):
        ev_names += [f"{s}_obj_contrast", f"{s}_ring_cont", f"{s}_core_ratio",
                     f"{s}_logcore", f"{s}_logring", f"{s}_logregion"]
    ev_names += ["scene_mean", "scene_std"] + [f"scene_b{k}" for k in range(8)]

    print(f"\n=== 反事实证据判别力(TP={is_tp.sum()}, FP_BG={is_fpbg.sum()}, FP_CLS={is_fpcls.sum()}) ===")
    print(f"{'证据':16s} {'TP均值':>8s} {'FP_BG均值':>9s} {'FP_CLS均值':>10s} {'|ΔTP-FPBG|':>10s}")
    separations = []
    for j, nm in enumerate(ev_names):
        mt = ev[is_tp, j].mean()
        mb = ev[is_fpbg, j].mean()
        mc = ev[is_fpcls, j].mean()
        sep = abs(mt - mb)
        separations.append((sep, nm, mt, mb, mc))
    for sep, nm, mt, mb, mc in sorted(separations, reverse=True)[:14]:
        print(f"{nm:16s} {mt:8.3f} {mb:9.3f} {mc:10.3f} {sep:10.3f}")

    # 训练 OER: 14特征 vs 14+反事实
    def train_eval(extra):
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

    probs_base = train_eval(None)
    probs_ev = train_eval(ev)

    # 评估(复用简化 frontier: 改类 oracle + NMS)
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
    print(f"\n[OER 14特征]        R@FDR=.12={br_base[0.12]:.4f} .11={br_base[0.11]:.4f} .10={br_base[0.10]:.4f}")
    print(f"[OER 14+反事实]     R@FDR=.12={br_ev[0.12]:.4f} .11={br_ev[0.11]:.4f} .10={br_ev[0.10]:.4f}")
    print(f"Δ(反事实) @FDR=.12: {br_ev[0.12]-br_base[0.12]:+.4f}")

    Path(args.output).write_text(json.dumps({
        "base": {"r_0.12": br_base[0.12], "r_0.11": br_base[0.11], "r_0.10": br_base[0.10]},
        "counterfactual": {"r_0.12": br_ev[0.12], "r_0.11": br_ev[0.11], "r_0.10": br_ev[0.10]},
        "evidence_dim": ev.shape[1],
        "top_separations": [(nm, float(mt), float(mb)) for sep, nm, mt, mb, mc
                            in sorted(separations, reverse=True)[:14]],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")


if __name__ == "__main__":
    main()
