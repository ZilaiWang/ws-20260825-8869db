#!/usr/bin/env python3
"""P0-1: GT-blind 改类审计(冻结真实可部署基线)。

当前主提交 0.9620 的改类部分依赖 gt_fine(位置匹配 GT 的细类集合)判断"yolo 是否错",
是 oracle relabel upper bound。本脚本:
1. 训练纯预测路由器(特征不含任何 GT 字段), 学习 P(改类有益) = trust_crop;
2. 对比 oracle 改类 vs deploy 路由器改类的 fixed-risk frontier;
3. 输出 predicted_change / corrected / broken 等 GT-blind 统计。

路由器训练用 GT 标签(标准监督学习), 推理只用纯预测特征(可部署)。
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

# 路由器特征(纯预测, 不含 GT 字段)
ROUTER_FEAT = ["y5_score", "crop_top1", "crop_margin", "crop_entropy",
               "detector_crop_agree", "short_edge", "area", "aspect", "local_density",
               "d4_support", "has_oto"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}
    n_gt = sum(len(gts) for gts in formal.boxes.values())

    nodes = pd.read_csv(args.nodes)
    nodes["fold"] = nodes["image_id"].map(fold_of)
    preds4 = json.loads(Path(args.preds_d4).read_text(encoding="utf-8"))
    nodes["d4_support"] = nodes["proposal_uid"].map(
        {p["proposal_uid"]: p["d4_support"] for p in preds4}).fillna(0).astype(int)
    oto_by_img = defaultdict(list)
    for f in (0, 1, 2):
        for p in json.loads(Path(args.oto_dir, f"a5_oto_fold{f}.json").read_text(encoding="utf-8")):
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

    from sklearn.ensemble import HistGradientBoostingClassifier

    # ============ 候选 + GT 匹配标签 ============
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

    # 路由器标签: yolo_wrong = 位置匹配 GT 且 yolo 错(即 oracle 全改会改的候选)
    trust_label = {}
    for p in cand:
        gf = gt_fine.get(p["_idx"], set())
        if not gf:
            trust_label[p["_idx"]] = 0
            continue
        trust_label[p["_idx"]] = 1 if (p["category_id"] not in gf) else 0

    # ============ OER(14特征, 纯预测) ============
    def train_oer(feats):
        X = nodes[feats].to_numpy(dtype=float)
        y = nodes["is_valid"].to_numpy(dtype=float)
        folds = nodes["fold"].to_numpy()
        probs = np.zeros(len(nodes))
        for held in (0, 1, 2):
            tr = np.where(folds != held)[0]; va = np.where(folds == held)[0]
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                                 l2_regularization=1.0, class_weight="balanced",
                                                 random_state=2026 + held)
            clf.fit(X[tr], y[tr]); probs[va] = clf.predict_proba(X[va])[:, 1]
        return probs
    oer_probs = train_oer(FEAT_BASE + ["d4_support", "has_oto"])

    # ============ 路由器(纯预测改类决策, cross-fit) ============
    def train_router():
        Xr = nodes[ROUTER_FEAT].to_numpy(dtype=float)
        yr = np.array([trust_label[int(r.idx)] for r in nodes.itertuples()], dtype=float)
        folds = nodes["fold"].to_numpy()
        probs = np.zeros(len(nodes))
        for held in (0, 1, 2):
            tr = np.where(folds != held)[0]; va = np.where(folds == held)[0]
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=5,
                                                 class_weight="balanced", random_state=42 + held)
            clf.fit(Xr[tr], yr[tr]); probs[va] = clf.predict_proba(Xr[va])[:, 1]
        return probs, yr
    router_probs, yr = train_router()
    from sklearn.metrics import roc_auc_score
    print(f"路由器(预测 trust_crop) AUC(三折 cross-fit): {roc_auc_score(yr, router_probs):.4f} "
          f"(trust_crop 正样本率 {yr.mean():.3f})")

    # ============ 评估: 给定改类方式, 算 frontier + corrected/broken ============
    def evaluate(relabel_mode, relabel_decision, thr=0.5):
        """relabel_mode: 'oracle' | 'router'; relabel_decision: router 预测概率数组(router 模式)"""
        newp = [dict(p) for p in cand]
        changed = set()
        for i, p in enumerate(newp):
            if relabel_mode == "oracle":
                gf = gt_fine.get(p["_idx"], set())
                if gf and p["category_id"] not in gf:
                    p["category_id"] = p["crop_top1_class"]
                    changed.add(p["_idx"])
            else:  # router
                if relabel_decision[i] > thr:
                    p["category_id"] = p["crop_top1_class"]
                    changed.add(p["_idx"])
            p["oer"] = float(oer_probs[i])
        # corrected / broken(对象级)
        corrected = broken = 0
        for idx in changed:
            gf = gt_fine.get(idx, set())
            orig_wrong = cand[idx]["category_id"] not in gf
            new_right = cand[idx]["crop_top1_class"] in gf
            if orig_wrong and new_right:
                corrected += 1
            elif (not orig_wrong) and (not new_right):
                broken += 1
        pb2 = defaultdict(list)
        for p in newp:
            pb2[p["image_id"]].append(p)
        kept = []
        for img, pl in pb2.items():
            od = sorted(pl, key=lambda p: -p["oer"]); sup = set()
            for p in od:
                if p["_idx"] in sup:
                    continue
                kept.append(p)
                for q in od:
                    if q["_idx"] == p["_idx"] or q["_idx"] in sup or q["category_id"] != p["category_id"]:
                        continue
                    if compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > 0.5:
                        sup.add(q["_idx"])
        tp = {}; used = defaultdict(set)
        pb3 = defaultdict(list)
        for p in kept:
            pb3[p["image_id"]].append(p)
        for img, gts in formal.boxes.items():
            pl = pb3.get(img, []); od = sorted(pl, key=lambda p: -p["oer"])
            for g in gts:
                cid = int(g["category_id"]); thr = proto.iou_thresholds[proto.category_mapping[cid]]
                bi, bix = 0.0, None
                for p in od:
                    if p["_idx"] in used[img] or p["category_id"] != cid:
                        continue
                    iou = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                    if iou > bi:
                        bi, bix = iou, p
                if bix is not None and bi >= thr:
                    used[img].add(bix["_idx"]); tp[bix["_idx"]] = True
        od2 = sorted(kept, key=lambda p: -p["oer"])
        tp_ = fp = 0; br = {v: 0.0 for v in (0.12, 0.11, 0.10)}
        for p in od2:
            if tp.get(p["_idx"]):
                tp_ += 1
            else:
                fp += 1
            rec = tp_ / n_gt; fdr = fp / (tp_ + fp) if tp_ + fp else 0
            for v in br:
                if fdr <= v:
                    br[v] = max(br[v], rec)
        return br, corrected, broken, len(changed)

    br_oracle, corr_o, brk_o, nchg_o = evaluate("oracle", None)
    print(f"\n[oracle 改类(GT 依赖)] R@FDR=.12={br_oracle[0.12]:.4f} .11={br_oracle[0.11]:.4f} .10={br_oracle[0.10]:.4f} "
          f"(改 {nchg_o}, corrected {corr_o}, broken {brk_o})")

    best = None
    print(f"[deploy 路由器(纯预测, 阈值扫描)]")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        br, corr, brk, nchg = evaluate("router", router_probs, thr)
        print(f"  thr={thr}: R@FDR=.12={br[0.12]:.4f} .11={br[0.11]:.4f} .10={br[0.10]:.4f} "
              f"(改 {nchg}, corrected {corr}, broken {brk})")
        if best is None or br[0.12] > best[0]:
            best = (br[0.12], thr, br, corr, brk)

    print(f"\n最优 deploy: thr={best[1]} R@FDR=.12={best[2][0.12]:.4f} (oracle {br_oracle[0.12]:.4f}, "
          f"Δ {best[2][0.12]-br_oracle[0.12]:+.4f}, corrected {best[3]}, broken {best[4]})")

    Path(args.output).write_text(json.dumps({
        "oracle": {"r_at_fdr_0.12": br_oracle[0.12], "r_at_fdr_0.11": br_oracle[0.11]},
        "deploy_best": {"thr": best[1], "r_at_fdr_0.12": best[2][0.12],
                        "corrected": best[3], "broken": best[4]},
        "router_auc": float(roc_auc_score(yr, router_probs)),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
