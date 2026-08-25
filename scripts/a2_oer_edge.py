#!/usr/bin/env python3
"""A2 第二阶段: edge_same_object 去重(替代 NMS)。

训练 edge 模型判断候选对是否同一对象(区别于"相邻真实对象"), 然后用
OER node_validity 分数排序 + edge 抑制去重, 评估固定风险前沿。

用法:
  python scripts/a2_oer_edge.py \
    --scores /tmp/oer_scores.csv \
    --edges /tmp/a1-object-graph/edges.csv \
    --nodes /tmp/a1-object-graph/nodes.csv \
    --formal-crop-manifest outputs/.../formal_crop_manifest.csv \
    --project-config configs/project.yaml \
    --output /tmp/a2_oer_edge.json
"""
from __future__ import annotations

import argparse
import json
import math
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

EDGE_FEATURES = ["iou", "center_dist", "size_ratio", "fine_same", "coarse_same"]


def greedy_match(preds, formal, proto) -> dict:
    pb = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    tp_flags = {}
    used = defaultdict(set)
    for img_id, gts in formal.boxes.items():
        plist = pb.get(img_id, [])
        ordered = sorted(enumerate(plist), key=lambda t: -t[1]["score"])
        for g in gts:
            cid = int(g["category_id"])
            thr = proto.iou_thresholds[proto.category_mapping[cid]]
            bi, bix = 0.0, None
            for idx, p in ordered:
                if p["_idx"] in used[img_id] or p["category_id"] != cid:
                    continue
                iou = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if iou > bi:
                    bi, bix = iou, idx
            if bix is not None and bi >= thr:
                used[img_id].add(plist[bix]["_idx"])
                tp_flags[plist[bix]["_idx"]] = True
    return tp_flags


def frontier(preds, score_key, tp_flags, n_gt) -> dict:
    ordered = sorted(preds, key=lambda p: -p[score_key])
    tp = fp = 0
    br = {v: 0.0 for v in (0.10, 0.12, 0.15)}
    bf = {v: 1.0 for v in (0.94, 0.95, 0.96)}
    for p in ordered:
        tp += 1 if tp_flags.get(p["_idx"]) else 0
        fp += 0 if tp_flags.get(p["_idx"]) else 1
        rec = tp / n_gt
        fdr = fp / (tp + fp) if tp + fp else 0.0
        for v in br:
            if fdr <= v:
                br[v] = max(br[v], rec)
        for v in bf:
            if rec >= v:
                bf[v] = min(bf[v], fdr)
    return {"recall_at_fdr": {f"fdr_{v}": round(br[v], 4) for v in (0.10, 0.12, 0.15)},
            "fdr_at_recall": {f"rec_{v}": round(bf[v], 4) for v in (0.94, 0.95, 0.96)}}


def edge_feat(a: dict, b: dict) -> list[float]:
    iou = compute_iou(a["bbox_xyxy"], b["bbox_xyxy"])
    ax = (a["bbox_xyxy"][0] + a["bbox_xyxy"][2]) / 2
    ay = (a["bbox_xyxy"][1] + a["bbox_xyxy"][3]) / 2
    bx = (b["bbox_xyxy"][0] + b["bbox_xyxy"][2]) / 2
    by = (b["bbox_xyxy"][1] + b["bbox_xyxy"][3]) / 2
    cdist = math.hypot(ax - bx, ay - by)
    sa = max(a["bbox_xyxy"][2] - a["bbox_xyxy"][0], 1e-6) * max(a["bbox_xyxy"][3] - a["bbox_xyxy"][1], 1e-6)
    sb = max(b["bbox_xyxy"][2] - b["bbox_xyxy"][0], 1e-6) * max(b["bbox_xyxy"][3] - b["bbox_xyxy"][1], 1e-6)
    sr = min(sa, sb) / max(sa, sb)
    fine_same = 1 if a["category_id"] == b["category_id"] else 0
    coarse_same = 1 if proto.category_mapping[a["category_id"]] == proto.category_mapping[b["category_id"]] else 0
    return [iou, cdist, sr, fine_same, coarse_same]


proto = None


def main() -> int:
    global proto
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--same-threshold", type=float, default=0.5,
                    help="edge P(same) 超过该阈值即抑制")
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    n_gt = sum(len(gts) for gts in formal.boxes.values())

    # 候选(从 nodes 还原 bbox/category, 从 scores 取各分数)
    nodes = pd.read_csv(args.nodes)
    scores = pd.read_csv(args.scores)
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}
    nodes["fold"] = nodes["image_id"].map(fold_of)
    merged = nodes.merge(scores[["idx", "oer_score", "softrisk", "y5_score"]].rename(
        columns={"y5_score": "y5s"}), on="idx")
    preds = []
    for r in merged.itertuples():
        preds.append({
            "image_id": int(r.image_id), "category_id": int(r.category_id),
            "bbox_xyxy": [r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2],
            "_idx": int(r.idx), "fold": int(r.fold),
            "score": float(r.y5s), "oer": float(r.oer_score),
            "softrisk": float(r.softrisk),
        })
    tp_flags = greedy_match(preds, formal, proto)

    # 训练 edge_same 二分类(下采样 background 边 + 逻辑回归, 快速)
    edges = pd.read_csv(args.edges)
    y_edge = (edges["label"] == 0).astype(int)  # same=1, 其他=0
    # 下采样: 保留全部 same/different, background 下采样到 same 的 3 倍
    bg_idx = edges.index[edges["label"] == 2].to_numpy()
    fg_idx = edges.index[edges["label"] != 2].to_numpy()
    n_same = int(y_edge.sum())
    rng = np.random.default_rng(42)
    bg_keep = rng.choice(bg_idx, size=min(len(bg_idx), n_same * 3), replace=False)
    sel_idx = np.concatenate([fg_idx, bg_keep])
    edges_sel = edges.loc[sel_idx]
    X_edge = edges_sel[EDGE_FEATURES].to_numpy(dtype=float)
    y_edge_sel = (edges_sel["label"] == 0).astype(int).to_numpy()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
    clf.fit(X_edge, y_edge_sel)
    print(f"edge_same 二分类(下采样 {len(edges_sel)}/{len(edges)} 条): "
          f"AUC={roc_auc_score(y_edge_sel, clf.predict_proba(X_edge)[:, 1]):.4f} "
          f"(same 正样本率 {y_edge_sel.mean():.3f})")

    # 去重: 按 oer 分数降序, edge 抑制(空间剪枝: 只对 IoU>0 或中心距离<50 的候选对算 edge)
    def dedup(preds, score_key, thr):
        pb = defaultdict(list)
        for p in preds:
            pb[p["image_id"]].append(p)
        kept = []
        for img_id, plist in pb.items():
            ordered = sorted(plist, key=lambda p: -p[score_key])
            local_kept = []
            for p in ordered:
                is_dup = False
                for q in local_kept:
                    f = edge_feat(p, q)
                    if f[0] <= 0 and f[1] >= 50.0:
                        continue  # 空间不相邻, 必非 same
                    p_same = clf.predict_proba(np.array([f]))[0, 1]
                    if p_same > thr:
                        is_dup = True
                        break
                if not is_dup:
                    local_kept.append(p)
            kept.extend(local_kept)
        return kept

    res = {}
    for name, key in [("oer", "oer"), ("oer+edge去重", "oer"), ("softrisk", "softrisk"),
                      ("softrisk+edge去重", "softrisk")]:
        if "edge去重" in name:
            k = dedup(preds, key, args.same_threshold)
            res[name] = frontier(k, key, tp_flags, n_gt)
            res[name]["n_kept"] = len(k)
        else:
            res[name] = frontier(preds, key, tp_flags, n_gt)
            res[name]["n_kept"] = len(preds)

    print("\n=== A2 第二阶段: edge 去重 ===")
    for name, r in res.items():
        print(f"[{name}] n={r.get('n_kept', 0)}")
        for k, v in r["recall_at_fdr"].items():
            print(f"  Recall@{k.split('_')[1]}: {v}")
        for k, v in r["fdr_at_recall"].items():
            print(f"  FDR@{k.split('_')[1]}: {v}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_gt": n_gt, "models": res}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
