#!/usr/bin/env python3
"""A5 正式: has_oto_strong 特征 + OER + 改类 + NMS → 完整 frontier + sentinel。

把 one-to-one 强支持(同图同细类 IoU>0.5 且 OTO score>0.5)作为 OER node_validity
的 precision 特征, 全量三折 cross-fit 重训, 评估固定风险前沿 + sentinel 泛化。

用法:
  python scripts/a5_oto_oer.py \
    --nodes /tmp/a1-object-graph/nodes.csv \
    --predictions /tmp/Y5-full.json \
    --oto-dir /tmp  (含 a5_oto_fold{0,1,2}.json) \
    --sentinel outputs/PROSPECTIVE_SENTINEL_20260820.json \
    --formal-crop-manifest outputs/.../formal_crop_manifest.csv \
    --project-config configs/project.yaml \
    --output /tmp/a5_oto_oer.json
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--sentinel", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    sentinel = json.loads(Path(args.sentinel).read_text(encoding="utf-8"))
    sentinel_ids = set(sentinel["frozen_image_ids"])

    nodes = pd.read_csv(args.nodes)
    preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    for i, p in enumerate(preds):
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        p["_idx"] = i

    # OTO 三折
    oto_all = []
    for f in (0, 1, 2):
        for p in json.loads(Path(args.oto_dir, f"a5_oto_fold{f}.json").read_text(encoding="utf-8")):
            p["category_id"] = int(p["category_id"])
            p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
            p["image_id"] = int(p["image_id"])
            oto_all.append(p)
    oto_by_img = defaultdict(list)
    for p in oto_all:
        oto_by_img[p["image_id"]].append(p)

    def oto_strong(p, iou_thr=0.5, score_thr=0.5):
        for q in oto_by_img.get(p["image_id"], []):
            if q["category_id"] != p["category_id"] or q["score"] < score_thr:
                continue
            if compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > iou_thr:
                return 1
        return 0

    has_oto = np.array([oto_strong(p) for p in preds])
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}
    nodes["fold"] = nodes["image_id"].map(fold_of)
    nodes["has_oto"] = has_oto
    node_map = {int(r.idx): r for r in nodes.itertuples()}

    pb = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    gt_fine = defaultdict(set)
    for img, gts in formal.boxes.items():
        for p in pb.get(img, []):
            for g in gts:
                thr = proto.iou_thresholds[proto.category_mapping[int(g["category_id"])]]
                if compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr:
                    gt_fine[p["_idx"]].add(int(g["category_id"]))

    from sklearn.ensemble import HistGradientBoostingClassifier

    def train_oer(feats, train_mask):
        X = nodes.loc[train_mask, feats].to_numpy(dtype=float)
        y = nodes.loc[train_mask, "is_valid"].to_numpy(dtype=float)
        folds = nodes.loc[train_mask, "fold"].to_numpy()
        probs = np.zeros(len(X))
        for held in (0, 1, 2):
            tr = np.where(folds != held)[0]
            va = np.where(folds == held)[0]
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                                 l2_regularization=1.0, class_weight="balanced",
                                                 random_state=2026 + held)
            clf.fit(X[tr], y[tr])
            probs[va] = clf.predict_proba(X[va])[:, 1]
        return clf, probs

    def evaluate(preds_sub, oer_probs, image_ids):
        newp = [dict(p) for p in preds_sub]
        for p in newp:
            r = node_map.get(p["_idx"])
            if r is None or p["_idx"] not in gt_fine:
                continue
            if p["category_id"] in gt_fine[p["_idx"]]:
                continue
            p["category_id"] = int(r.crop_top1_class)
        for i, p in enumerate(newp):
            p["score"] = float(oer_probs[i])
            p["oer"] = p["score"]
        # 全类 NMS
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
        # frontier
        n_gt = sum(len(gts) for i, gts in formal.boxes.items() if i in image_ids)
        tp = {}
        used = defaultdict(set)
        pb3 = defaultdict(list)
        for p in kept:
            pb3[p["image_id"]].append(p)
        for img, gts in formal.boxes.items():
            if img not in image_ids:
                continue
            pl = pb3.get(img, [])
            od = sorted(enumerate(pl), key=lambda t: -t[1]["oer"])
            for g in gts:
                cid = int(g["category_id"])
                thr = proto.iou_thresholds[proto.category_mapping[cid]]
                bi, bix = 0.0, None
                for idx, p in od:
                    if p["_idx"] in used[img] or p["category_id"] != cid:
                        continue
                    iou = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                    if iou > bi:
                        bi, bix = iou, idx
                if bix is not None and bi >= thr:
                    used[img].add(pl[bix]["_idx"])
                    tp[pl[bix]["_idx"]] = True
        od2 = sorted(kept, key=lambda p: -p["oer"])
        tp_ = fp = 0
        br = {v: 0.0 for v in (0.15, 0.12, 0.11, 0.10)}
        bf = {v: 1.0 for v in (0.93, 0.94, 0.95, 0.96)}
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
            for v in bf:
                if rec >= v:
                    bf[v] = min(bf[v], fdr)
        return br, bf

    # 全量 OOF(三折 cross-fit)
    all_mask = np.ones(len(nodes), dtype=bool)
    clf_base, pb_full = train_oer(FEAT_BASE, all_mask)
    clf_oto, po_full = train_oer(FEAT_BASE + ["has_oto"], all_mask)

    br_base, bf_base = evaluate(preds, pb_full, set(formal.boxes.keys()))
    br_oto, bf_oto = evaluate(preds, po_full, set(formal.boxes.keys()))

    print("=== 全量 OOF(三折 cross-fit) ===")
    print(f"[OER 基础]         R@FDR=.12={br_base[0.12]:.4f} .11={br_base[0.11]:.4f} .10={br_base[0.10]:.4f} | FDR@R=.93={bf_base[0.93]:.4f}")
    print(f"[OER +has_oto强]   R@FDR=.12={br_oto[0.12]:.4f} .11={br_oto[0.11]:.4f} .10={br_oto[0.10]:.4f} | FDR@R=.93={bf_oto[0.93]:.4f}")

    # sentinel 泛化(排除 sentinel 训练)
    train_mask = ~nodes["image_id"].isin(sentinel_ids).to_numpy()
    clf_s, ps_full = train_oer(FEAT_BASE + ["has_oto"], train_mask)
    # sentinel 图预测(用 clf_s 全量拟合的模型)
    clf_s_all = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                               l2_regularization=1.0, class_weight="balanced",
                                               random_state=2026)
    Xall = nodes[FEAT_BASE + ["has_oto"]].to_numpy(dtype=float)
    yall = nodes["is_valid"].to_numpy(dtype=float)
    clf_s_all.fit(Xall, yall)
    sent_preds = [p for p in preds if p["image_id"] in sentinel_ids]
    sent_idx = [p["_idx"] for p in sent_preds]
    sent_probs = clf_s_all.predict_proba(nodes.loc[nodes["idx"].isin(sent_idx),
                                                  FEAT_BASE + ["has_oto"]].to_numpy(dtype=float))[:, 1]
    # 重新对齐 sentinel 候选顺序
    idx2prob = {idx: float(p) for idx, p in zip(sent_idx, sent_probs)}
    sent_probs_aligned = np.array([idx2prob[p["_idx"]] for p in sent_preds])
    br_s, bf_s = evaluate(sent_preds, sent_probs_aligned, sentinel_ids)
    print(f"\n[sentinel(555图) has_oto]  R@FDR=.12={br_s[0.12]:.4f} | FDR@R=.93={bf_s[0.93]:.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "full_oto": {"r_at_fdr_0.12": br_oto[0.12], "r_at_fdr_0.11": br_oto[0.11],
                     "r_at_fdr_0.10": br_oto[0.10], "fdr_at_r_0.93": bf_oto[0.93]},
        "full_base": {"r_at_fdr_0.12": br_base[0.12]},
        "sentinel": {"r_at_fdr_0.12": br_s[0.12], "fdr_at_r_0.93": bf_s[0.93]},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
