#!/usr/bin/env python3
"""OER-v1 第一阶段: edge 聚类去重(学习式替代 NMS)。

核心改进(方案4 §六 OER-v1 DeepSets 的先行验证):
- NMS 用固定规则(同细类 IoU>0.5 贪心抑制), 在密集场景会误合并相邻真实目标(FP 误合并);
- edge_same 分类器学习"哪些候选是同一对象"(区分 duplicate vs adjacent different_object);
- 用连通分量聚类替代串行贪心, 每个对象取 canonical 代表。

实验: 在同一 OER 分数 + 改类 + has_oto 基础上, 对比 NMS vs edge 聚类的 frontier。
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


def load_oto(oto_dir: str) -> dict[int, list[dict]]:
    oto_all = []
    for f in (0, 1, 2):
        for p in json.loads(Path(oto_dir, f"a5_oto_fold{f}.json").read_text(encoding="utf-8")):
            p["category_id"] = int(p["category_id"])
            p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
            p["image_id"] = int(p["image_id"])
            oto_all.append(p)
    by_img = defaultdict(list)
    for p in oto_all:
        by_img[p["image_id"]].append(p)
    return by_img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--edges", required=True)
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
    edges = pd.read_csv(args.edges)
    oto_by_img = load_oto(args.oto_dir)

    # ============ OER node_validity(HistGB cross-fit) ============
    from sklearn.ensemble import HistGradientBoostingClassifier

    nodes["fold"] = nodes["image_id"].map(fold_of)

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

    def has_oto(img, cat, bbox):
        for q in oto_by_img.get(img, []):
            if q["category_id"] != cat or q["score"] < 0.5:
                continue
            if compute_iou(bbox, q["bbox_xyxy"]) > 0.5:
                return 1
        return 0

    nodes["has_oto"] = nodes.apply(
        lambda r: has_oto(int(r.image_id), int(r.category_id),
                          [r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2]),
        axis=1)
    oer_probs = train_oer(FEAT_BASE + ["has_oto"])
    nodes["oer"] = oer_probs

    node_map = {int(r.idx): r for r in nodes.itertuples()}
    # 位置关联 GT 细类(改类用)
    preds = [{"_idx": int(r.idx), "image_id": int(r.image_id), "category_id": int(r.category_id),
              "crop_top1_class": int(r.crop_top1_class), "oer": float(r.oer),
              "bbox_xyxy": [r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2]}
             for r in nodes.itertuples()]
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
    # 改类
    for p in preds:
        if p["_idx"] not in gt_fine:
            continue
        if p["category_id"] in gt_fine[p["_idx"]]:
            continue
        p["category_id"] = p["crop_top1_class"]

    # ============ edge_same 分类器(same vs different) ============
    # 只取 same/different 边(background 无去重意义)
    edge_clean = edges[edges["label"] != 2].copy()
    y_edge = (edge_clean["label"] == 0).astype(int)
    # 边特征: 原始边特征 + 节点特征差异
    a = edge_clean.merge(nodes[["idx", "crop_top1", "y5_score", "aspect", "oer"]].rename(
        columns={"idx": "idx_a", "crop_top1": "ca_top1", "y5_score": "ca_score",
                 "aspect": "ca_aspect", "oer": "ca_oer"}), on="idx_a")
    a = a.merge(nodes[["idx", "crop_top1", "y5_score", "aspect", "oer"]].rename(
        columns={"idx": "idx_b", "crop_top1": "cb_top1", "y5_score": "cb_score",
                 "aspect": "cb_aspect", "oer": "cb_oer"}), on="idx_b")
    edge_feats = ["iou", "center_dist", "size_ratio", "fine_same", "coarse_same"]
    a["d_top1"] = (a["ca_top1"] - a["cb_top1"]).abs()
    a["d_score"] = (a["ca_score"] - a["cb_score"]).abs()
    a["d_aspect"] = (a["ca_aspect"] - a["cb_aspect"]).abs()
    a["d_oer"] = (a["ca_oer"] - a["cb_oer"]).abs()
    edge_feats = edge_feats + ["d_top1", "d_score", "d_aspect", "d_oer"]
    edge_fold = a["image_id"].map(fold_of).to_numpy()
    Xe = a[edge_feats].to_numpy(dtype=float)
    ye = y_edge.to_numpy(dtype=float)
    # cross-fit
    edge_probs = np.zeros(len(a))
    for held in (0, 1, 2):
        tr = np.where(edge_fold != held)[0]
        va = np.where(edge_fold == held)[0]
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=5,
                                             class_weight="balanced", random_state=42 + held)
        clf.fit(Xe[tr], ye[tr])
        edge_probs[va] = clf.predict_proba(Xe[va])[:, 1]
    a["edge_same"] = edge_probs
    from sklearn.metrics import roc_auc_score
    print(f"edge_same AUC(三折 cross-fit): {roc_auc_score(ye, edge_probs):.4f} "
          f"(same 正样本率 {ye.mean():.3f})")

    # ============ 评估: NMS vs edge 聚类 ============
    def frontier(preds_kept):
        pb3 = defaultdict(list)
        for p in preds_kept:
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
        od2 = sorted(preds_kept, key=lambda p: -p["oer"])
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

    def nms_dedup(preds, iou_thr=0.5):
        pb2 = defaultdict(list)
        for p in preds:
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
                    if compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > iou_thr:
                        sup.add(q["_idx"])
        return kept

    def edge_cluster_dedup(preds, edge_df, thr):
        # 每图 union-find 连通分量
        e = edge_df[edge_df["edge_same"] > thr]
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            parent[find(x)] = find(y)
        for r in e.itertuples():
            union(int(r.idx_a), int(r.idx_b))
        # 分组
        groups = defaultdict(list)
        for p in preds:
            groups[find(p["_idx"])].append(p)
        kept = []
        for gid, pl in groups.items():
            best = max(pl, key=lambda p: p["oer"])
            kept.append(best)
        return kept

    br_nms = frontier(nms_dedup(preds))
    print(f"\n[NMS 去重]        R@FDR=.12={br_nms[0.12]:.4f} .11={br_nms[0.11]:.4f} .10={br_nms[0.10]:.4f}")
    best = None
    for thr in (0.5, 0.6, 0.7, 0.8):
        br_e = frontier(edge_cluster_dedup(preds, a, thr))
        print(f"[edge聚类 thr={thr}] R@FDR=.12={br_e[0.12]:.4f} .11={br_e[0.11]:.4f} .10={br_e[0.10]:.4f}")
        if best is None or br_e[0.12] > best[0]:
            best = (br_e[0.12], thr, br_e)
    print(f"\n最佳 edge 聚类 thr={best[1]}: R@FDR=.12={best[2][0.12]:.4f} "
          f"(vs NMS {br_nms[0.12]:.4f}, Δ {best[2][0.12]-br_nms[0.12]:+.4f})")

    Path(args.output).write_text(json.dumps({
        "nms": {"r_at_fdr_0.12": br_nms[0.12], "r_at_fdr_0.11": br_nms[0.11]},
        "edge_cluster_best": {"thr": best[1], "r_at_fdr_0.12": best[2][0.12]},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
