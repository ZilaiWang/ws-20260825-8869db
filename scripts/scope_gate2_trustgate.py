"""SCOPE 改类链路 v2：trust_label 路由器 + 同类抑制门控。

机制（已确认）：
- oracle 改类收益 97% 来自 delta_fp<0（改类后被同 image 同类候选 NMS 抑制 → 减少 FP）
- 改类候选 = 位置匹配 GT 但类别错（trust_label，AUC 0.97 可学）
- 门控 = 同 image 有类别==crop_top1_class 且 IoU>0.5 的候选（deploy 可观测）

本脚本三折严格 OOF：
1. trust_label 路由器（学"位置匹配 GT 但类别错"）
2. 同类抑制门控
3. 联合改类 + frontier 评估 vs base vs oracle
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
from rsdet.scope.official_scorer import CandidateView
from rsdet.scope.incremental_scorer import IncrementalFrontierScorer
from sklearn.ensemble import HistGradientBoostingClassifier


FEAT_BASE = ["y5_score", "crop_top1", "crop_margin", "crop_entropy",
             "crop_top1_class", "detector_crop_agree", "w", "h", "area",
             "short_edge", "aspect", "local_density"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}

    nodes = pd.read_csv(args.nodes)
    nodes["fold"] = nodes["image_id"].map(fold_of)
    node_map = {int(r.idx): r for r in nodes.itertuples()}

    preds = json.load(open(args.preds_d4))
    for i, p in enumerate(preds):
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        p["_idx"] = i

    oto_by_img = defaultdict(list)
    for f in (0, 1, 2):
        fp = Path(args.oto_dir) / f"a5_oto_fold{f}.json"
        if fp.exists():
            for p in json.load(open(fp)):
                oto_by_img[int(p["image_id"])].append(p)
    has_oto = np.zeros(len(preds), dtype=np.int8)
    for i, p in enumerate(preds):
        for q in oto_by_img.get(p["image_id"], []):
            if int(q["category_id"]) == p["category_id"] and q["score"] > 0.5 and \
                    compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > 0.5:
                has_oto[i] = 1
                break
    nodes["has_oto"] = has_oto

    F = FEAT_BASE + ["has_oto"]
    X = nodes[F].to_numpy(float)
    y = nodes["is_valid"].to_numpy(float)
    folds = nodes["fold"].to_numpy()
    oer = np.zeros(len(nodes))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=6, l2_regularization=1.0,
                                             class_weight="balanced", random_state=2026 + held)
        clf.fit(X[tr], y[tr])
        oer[va] = clf.predict_proba(X[va])[:, 1]

    image_ids = set(formal.boxes.keys())

    # gt_fine（仅离线标签）
    gt_fine = defaultdict(set)
    pb = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    for img, gts in formal.boxes.items():
        for p in pb.get(img, []):
            for g in gts:
                thr = proto.iou_thresholds[proto.category_mapping[int(g["category_id"])]]
                if compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr:
                    gt_fine[p["_idx"]].add(int(g["category_id"]))

    all_cands = [CandidateView(_idx=i, image_id=p["image_id"], category_id=p["category_id"],
                               score=float(oer[i]), bbox_xyxy=p["bbox_xyxy"])
                 for i, p in enumerate(preds) if p["image_id"] in image_ids]
    isc = IncrementalFrontierScorer(proto, formal.boxes, image_ids, all_cands)
    base_u, base_tp, base_fp = isc.score(), isc.n_tp, isc.n_fp

    img_cands = defaultdict(list)
    for c in all_cands:
        img_cands[c.image_id].append(c)

    def frontier_relabel(idx_set):
        edits = {i: ("relabel", int(node_map[i].crop_top1_class)) for i in idx_set}
        return isc.score_after_multi(edits)[0]

    # oracle 上界
    relabel_idx = [p["_idx"] for p in preds if p["_idx"] in gt_fine and p["category_id"] not in gt_fine[p["_idx"]]]
    u_oracle = frontier_relabel(set(relabel_idx))
    print(f"三折 base={base_u:.4f}  oracle(改{len(relabel_idx)})={u_oracle:.4f} (Δ{u_oracle-base_u:+.4f})")

    # trust_label 路由器（严格 OOF）：学"位置匹配 GT 但类别错"
    trust = np.array([1 if (i in gt_fine and preds[i]["category_id"] not in gt_fine[i]) else 0
                      for i in range(len(preds))], dtype=int)
    trust_prob = np.zeros(len(preds))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=5,
                                           class_weight="balanced", random_state=42 + held)
        c.fit(X[tr], trust[tr])
        trust_prob[va] = c.predict_proba(X[va])[:, 1]
    print(f"trust_label 路由器: 正样本={trust.sum()}, AUC 见 gate1b(0.97)")

    # 组合：trust_prob >= T 且 同 image 有同类 IoU>0.5 候选
    print("\n=== trust_label 路由器 + 同类抑制门控 ===")
    best = (0.0, base_u, 0)
    for tt in (0.5, 0.7, 0.8, 0.9):
        sel = set()
        for i, p in enumerate(preds):
            if p["image_id"] not in image_ids:
                continue
            if trust_prob[i] < tt:
                continue
            if int(node_map[i].crop_top1_class) == p["category_id"]:
                continue
            alt = int(node_map[i].crop_top1_class)
            bi = p["bbox_xyxy"]
            hit = False
            for c in img_cands[p["image_id"]]:
                if c.category_id == alt and c._idx != i and compute_iou(bi, c.bbox_xyxy) > 0.5:
                    hit = True
                    break
            if hit:
                sel.add(i)
        u = frontier_relabel(sel)
        print(f"  trust>={tt}: 改{len(sel)} → {u:.4f} (Δ{u-base_u:+.4f})")
        if u > best[1]:
            best = (tt, u, len(sel))
    print(f"\n最优: trust>={best[0]} → {best[1]:.4f} (Δ{best[1]-base_u:+.4f}) 改{best[2]}")
    print(f"oracle 增益恢复率: {(best[1]-base_u)/(u_oracle-base_u):.1%}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "base": base_u, "oracle": u_oracle,
        "best_frontier": best[1], "best_thr": best[0], "best_n": best[2],
        "recovery": (best[1] - base_u) / (u_oracle - base_u),
    }, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已写: {args.output}")


if __name__ == "__main__":
    main()
