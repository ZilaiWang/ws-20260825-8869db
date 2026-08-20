#!/usr/bin/env python3
"""OER-v1 验证: D4 旋转一致性特征对 OER node_validity 的增量。

对比: OER 基础(12 特征) vs OER + d4_support(13 特征), 同样改类 + NMS。
d4_support 从 y5-all-preds-d4.json 对齐到 nodes(通过 proposal_uid)。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

FEAT_BASE = ["y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
             "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}
    n_gt = sum(len(gts) for gts in formal.boxes.values())

    nodes = pd.read_csv(args.nodes)
    preds = json.loads(Path(args.preds_d4).read_text(encoding="utf-8"))
    uid2d4 = {p["proposal_uid"]: p["d4_support"] for p in preds}
    nodes["d4_support"] = nodes["proposal_uid"].map(uid2d4).fillna(0).astype(int)
    nodes["fold"] = nodes["image_id"].map(fold_of)

    from sklearn.ensemble import HistGradientBoostingClassifier

    def train_oer(feats):
        X = nodes[feats].to_numpy(dtype=float)
        y = nodes["is_valid"].to_numpy(dtype=float)
        folds = nodes["fold"].to_numpy()
        probs = np.zeros(len(nodes))
        for held in (0, 1, 2):
            tr = np.where(folds != held)[0]
            va = np.where(folds == held)[0]
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                                 l2_regularization=1.0, class_weight="balanced",
                                                 random_state=2026 + held)
            clf.fit(X[tr], y[tr])
            probs[va] = clf.predict_proba(X[va])[:, 1]
        return probs

    node_map = {int(r.idx): r for r in nodes.itertuples()}
    # 候选(改类 + NMS)
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
        if p["_idx"] not in gt_fine:
            continue
        if p["category_id"] in gt_fine[p["_idx"]]:
            continue
        p["category_id"] = p["crop_top1_class"]

    def evaluate(oer_probs):
        newp = [dict(p) for p in cand]
        for i, p in enumerate(newp):
            p["oer"] = float(oer_probs[i])
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
        tp = {}
        used = defaultdict(set)
        pb3 = defaultdict(list)
        for p in kept:
            pb3[p["image_id"]].append(p)
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
        br = {v: 0.0 for v in (0.15, 0.12, 0.11, 0.10)}
        for p in od2:
            if tp.get(p["_idx"]):
                tp_ += 1
            else:
                fp += 1
            rec = tp_ / n_gt
            fdr = fp / (tp_ + fp) if tp_ + fp else 0.0
            for v in br:
                if fdr <= v:
                    br[v] = max(br[v], rec)
        return br

    pb_base = train_oer(FEAT_BASE)
    pb_d4 = train_oer(FEAT_BASE + ["d4_support"])
    br_base = evaluate(pb_base)
    br_d4 = evaluate(pb_d4)
    print(f"[OER 基础 12特征]  R@FDR=.12={br_base[0.12]:.4f} .11={br_base[0.11]:.4f} .10={br_base[0.10]:.4f}")
    print(f"[OER +d4 13特征]   R@FDR=.12={br_d4[0.12]:.4f} .11={br_d4[0.11]:.4f} .10={br_d4[0.10]:.4f}")
    print(f"Δ R@FDR=.12: {br_d4[0.12]-br_base[0.12]:+.4f}")

    Path(args.output).write_text(json.dumps({
        "base": {"r_at_fdr_0.12": br_base[0.12], "r_at_fdr_0.11": br_base[0.11]},
        "d4": {"r_at_fdr_0.12": br_d4[0.12], "r_at_fdr_0.11": br_d4[0.11]},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
