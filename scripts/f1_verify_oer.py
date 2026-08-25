"""F1 验证: foreground rejector logit 加入 OER 后评估 frontier。

foreground logit = P(真实目标), 是压 FP_BG 的独立证据(与 y5_score 互补?需验证)。
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
    ap.add_argument("--fg-logits", required=True, help="F1 foreground logit JSON")
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

    # F1 foreground logit
    fg = json.load(open(args.fg_logits))
    nodes["fg_logit"] = nodes["proposal_uid"].map(fg).fillna(0.0)
    print(f"fg_logit 覆盖: {(nodes['fg_logit'] != 0).sum()}/{len(nodes)}")

    # 判别力诊断
    is_tp = nodes["is_valid"].to_numpy() == 1
    is_fpbg = (nodes["is_valid"].to_numpy() == 0) & (nodes["fine_correct"].to_numpy() == 0)
    print(f"\n=== fg_logit 判别力 ===")
    print(f"  TP 均值: {nodes['fg_logit'][is_tp].mean():.4f} | FP_BG 均值: {nodes['fg_logit'][is_fpbg].mean():.4f}")
    # fg_logit 与 y5_score 的相关性
    corr = nodes["fg_logit"].corr(nodes["y5_score"])
    print(f"  fg_logit vs y5_score 相关: {corr:.4f}")

    # OER 训练对比
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
    probs_fg = train(FEAT_BASE + ["fg_logit"])

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
    br_fg = evaluate(probs_fg)
    print(f"\n[OER 14特征]         R@FDR=.12={br_base[0.12]:.4f} .11={br_base[0.11]:.4f} .10={br_base[0.10]:.4f}")
    print(f"[OER 14+fg_logit]    R@FDR=.12={br_fg[0.12]:.4f} .11={br_fg[0.11]:.4f} .10={br_fg[0.10]:.4f}")
    print(f"Δ(fg_logit) @FDR=.12: {br_fg[0.12]-br_base[0.12]:+.4f}")

    Path(args.output).write_text(json.dumps({
        "base": {"r_0.12": br_base[0.12], "r_0.11": br_base[0.11]},
        "fg_logit": {"r_0.12": br_fg[0.12], "r_0.11": br_fg[0.11]},
        "fg_tp_mean": float(nodes["fg_logit"][is_tp].mean()),
        "fg_fpbg_mean": float(nodes["fg_logit"][is_fpbg].mean()),
        "corr_with_y5": float(corr),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")


if __name__ == "__main__":
    main()
