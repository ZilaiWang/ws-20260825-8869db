"""C7: OER-v2 listwise 排序(方案5 §十一)。

核心: 保留 NMS, 用 hard-pair 加权的 HistGB 直接优化"322 个排序损失"。
- baseline OER(14特征) cross-fit 得到 probs;
- 构造 hard pair: 低分 TP(排序损失根因) vs 高分 FP_BG(侵占前沿);
- 同预测类/同域内配对, 用 sample_weight 加权重训;
- 评估 fixed-FDR frontier 与排序损失变化。

纯本地, 缓存级(基于 65,301 候选 + d4 + OTO + crop 证据)。
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
]
FEAT_FULL = FEAT_BASE + ["d4_support", "has_oto"]


def load_nodes(nodes_path, preds_d4_path, oto_dir, formal, proto):
    nodes = pd.read_csv(nodes_path)
    # d4_support
    preds4 = json.load(open(preds_d4_path))
    nodes["d4_support"] = nodes["proposal_uid"].map(
        {p["proposal_uid"]: p["d4_support"] for p in preds4}).fillna(0).astype(int)
    # has_oto (同图同细类 IoU>0.5 且 OTO score>0.5)
    oto_by_img = defaultdict(list)
    for f in (0, 1, 2):
        for p in json.load(open(Path(oto_dir) / f"a5_oto_fold{f}.json")):
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
    # fold / group_id(域)
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}
    grp_of = {i: o.group_id for (i, _), o in formal.objects.items()}
    nodes["fold"] = nodes["image_id"].map(fold_of)
    nodes["group_id"] = nodes["image_id"].map(grp_of)
    return nodes


def build_candidates(nodes, formal, proto):
    """构造候选(带 _idx/category/bbox/crop_top1_class/gt_fine)。"""
    cand = []
    for r in nodes.itertuples():
        cand.append({
            "_idx": int(r.idx), "image_id": int(r.image_id),
            "category_id": int(r.category_id), "crop_top1_class": int(r.crop_top1_class),
            "coarse_id": int(r.coarse_id),
            "bbox_xyxy": [r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2],
        })
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
    return cand, gt_fine


def train_oer(nodes, feats, sample_weight=None):
    X = nodes[feats].to_numpy(dtype=float)
    y = nodes["is_valid"].to_numpy(dtype=float)
    folds = nodes["fold"].to_numpy()
    probs = np.zeros(len(nodes))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6,
            l2_regularization=1.0, random_state=2026 + held)
        sw = None if sample_weight is None else sample_weight[tr]
        clf.fit(X[tr], y[tr], sample_weight=sw)
        probs[va] = clf.predict_proba(X[va])[:, 1]
    return probs


def evaluate(cand, probs, formal, proto, n_gt):
    """改类(oracle) + NMS + 贪心匹配 → frontier + 排序损失。"""
    newp = [dict(p) for p in cand]
    # oracle 改类: yolo 错(位置匹配 GT 但细类不在其中)→ crop top1
    gt_fine = build_gt_fine(cand, formal, proto)
    for p in newp:
        gf = gt_fine.get(p["_idx"], set())
        if gf and p["category_id"] not in gf:
            p["category_id"] = p["crop_top1_class"]
    for i, p in enumerate(newp):
        p["oer"] = float(probs[i])
    # NMS
    pb2 = defaultdict(list)
    for p in newp:
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
    # 贪心匹配(全候选 floor + 前沿)
    def greedy(preds):
        pb3 = defaultdict(list)
        for p in preds:
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
        return tp

    tp_floor = greedy(kept)
    floor = sum(tp_floor.values())
    # frontier
    od2 = sorted(kept, key=lambda p: -p["oer"])
    tp_ = fp = 0
    br = {v: 0.0 for v in (0.12, 0.11, 0.10)}
    best = (0.0, 0, 0)
    for p in od2:
        if tp_floor.get(p["_idx"]):
            tp_ += 1
        else:
            fp += 1
        rec = tp_ / n_gt
        fdr = fp / (tp_ + fp) if tp_ + fp else 0
        for v in br:
            if fdr <= v:
                br[v] = max(br[v], rec)
        if fdr <= 0.12 and rec > best[0]:
            best = (rec, tp_, fp)
    rank_loss = floor - best[1]
    return br, floor, rank_loss, best


def build_gt_fine(cand, formal, proto):
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
    return gt_fine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--hard-weight", type=float, default=3.0, help="hard pair 样本加权倍数")
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    n_gt = sum(len(gts) for gts in formal.boxes.values())

    nodes = load_nodes(args.nodes, args.preds_d4, args.oto_dir, formal, proto)
    cand, gt_fine = build_candidates(nodes, formal, proto)
    print(f"候选 {len(nodes)}, GT {n_gt}")

    # baseline OER
    probs_base = train_oer(nodes, FEAT_FULL)
    br_base, floor_base, rank_base, best_base = evaluate(cand, probs_base, formal, proto, n_gt)
    print(f"\n[baseline OER(14特征)] R@FDR=.12={br_base[0.12]:.4f} .11={br_base[0.11]:.4f} .10={br_base[0.10]:.4f}")
    print(f"  floor={floor_base} 工作点TP={best_base[1]} 排序损失={rank_base}")

    # 构造 hard pair sample_weight
    # hard_TP: is_valid=1 且 prob 低(低于该类 TP 的 median)
    # hard_FP_BG: is_valid=0 且 fine_correct=0(位置不匹配 GT) 且 prob 高
    nodes["prob_base"] = probs_base
    sw = np.ones(len(nodes))
    # 每个候选是否 FP_BG(位置不匹配任何 GT)
    is_fpbg = (nodes["is_valid"] == 0) & (nodes["fine_correct"] == 0)
    tp_med = nodes[nodes["is_valid"] == 1].groupby("category_id")["prob_base"].median()
    fp_med = nodes[is_fpbg].groupby("category_id")["prob_base"].median()
    hard_tp = (nodes["is_valid"] == 1) & (
        nodes["prob_base"] < nodes["category_id"].map(tp_med))
    hard_fpbg = is_fpbg & (
        nodes["prob_base"] > nodes["category_id"].map(fp_med))
    sw[hard_tp.to_numpy()] = args.hard_weight
    sw[hard_fpbg.to_numpy()] = args.hard_weight
    print(f"\nhard 样本: hard_TP={int(hard_tp.sum())}, hard_FP_BG={int(hard_fpbg.sum())} "
          f"(加权 {args.hard_weight}x)")

    # 加权重训
    probs_hard = train_oer(nodes, FEAT_FULL, sample_weight=sw)
    br_hard, floor_hard, rank_hard, best_hard = evaluate(cand, probs_hard, formal, proto, n_gt)
    print(f"\n[OER + hard-pair 加权] R@FDR=.12={br_hard[0.12]:.4f} .11={br_hard[0.11]:.4f} .10={br_hard[0.10]:.4f}")
    print(f"  floor={floor_hard} 工作点TP={best_hard[1]} 排序损失={rank_hard}")

    print(f"\nΔ(加权−基线) @FDR=.12: {br_hard[0.12]-br_base[0.12]:+.4f}")
    print(f"排序损失: {rank_base} → {rank_hard} (Δ {rank_hard-rank_base:+d})")

    Path(args.output).write_text(json.dumps({
        "baseline": {"r_fdr_0.12": br_base[0.12], "r_fdr_0.11": br_base[0.11],
                     "r_fdr_0.10": br_base[0.10], "floor": floor_base,
                     "workpoint_tp": best_base[1], "rank_loss": rank_base},
        "hard_pair": {"r_fdr_0.12": br_hard[0.12], "r_fdr_0.11": br_hard[0.11],
                      "r_fdr_0.10": br_hard[0.10], "floor": floor_hard,
                      "workpoint_tp": best_hard[1], "rank_loss": rank_hard},
        "hard_tp": int(hard_tp.sum()), "hard_fpbg": int(hard_fpbg.sum()),
        "hard_weight": args.hard_weight,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")


if __name__ == "__main__":
    main()
