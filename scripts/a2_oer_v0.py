#!/usr/bin/env python3
"""A2: OER-v0 对象证据解析器(第一阶段: node_validity 重排序)。

在 Y5 全量候选(65K)上用更丰富的对象级特征训练 node_validity 分类器,
替代 SoftRisk 的简单逻辑回归, 判断"候选能否成为官方 TP"。

对比三档分数在固定风险前沿上的表现:
  1. Y5 原始 score(基线)
  2. SoftRisk v0(位置标签逻辑回归)
  3. OER node_validity(HistGB + 丰富特征)

用法:
  python scripts/a2_oer_v0.py \
    --nodes /tmp/a1-object-graph/nodes.csv \
    --formal-crop-manifest outputs/.../formal_crop_manifest.csv \
    --project-config configs/project.yaml \
    --output /tmp/a2_oer_v0.json
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

FEATURES = ["y5_score", "crop_top1", "crop_margin", "crop_entropy",
            "crop_top1_class", "detector_crop_agree",
            "w", "h", "area", "short_edge", "aspect", "local_density",
            "coarse_id"]
LABEL = "is_valid"


def greedy_match(preds: list[dict], formal, proto) -> dict:
    pb: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    tp_flags: dict[int, bool] = {}
    used: dict[int, set] = defaultdict(set)
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


def frontier_from_scores(preds: list[dict], score_key: str, tp_flags: dict, n_gt: int) -> dict:
    """按某分数列重排序 → PR frontier。"""
    ordered = sorted(preds, key=lambda p: -p[score_key])
    tp = fp = 0
    best_r_at_fdr = {v: 0.0 for v in (0.10, 0.12, 0.15)}
    best_f_at_rec = {v: 1.0 for v in (0.94, 0.95, 0.96)}
    for p in ordered:
        if tp_flags.get(p["_idx"]):
            tp += 1
        else:
            fp += 1
        rec = tp / n_gt
        fdr = fp / (tp + fp) if tp + fp else 0.0
        for v in best_r_at_fdr:
            if fdr <= v:
                best_r_at_fdr[v] = max(best_r_at_fdr[v], rec)
        for v in best_f_at_rec:
            if rec >= v:
                best_f_at_rec[v] = min(best_f_at_rec[v], fdr)
    return {
        "recall_at_fdr": {f"fdr_{v}": round(best_r_at_fdr[v], 4) for v in (0.10, 0.12, 0.15)},
        "fdr_at_recall": {f"rec_{v}": round(best_f_at_rec[v], 4) for v in (0.94, 0.95, 0.96)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    n_gt = sum(len(gts) for gts in formal.boxes.values())

    df = pd.read_csv(args.nodes)
    print(f"节点: {len(df)}, 正样本(is_valid=1): {df[LABEL].sum()}")

    # fold 映射(image_id -> fold)
    fold_of = {}
    for (img_id, _), obj in formal.objects.items():
        fold_of[img_id] = obj.fold
    df["fold"] = df["image_id"].map(fold_of)

    # 还原预测对象(带 bbox, 用于重排序后评估)
    preds = []
    for _, row in df.iterrows():
        preds.append({
            "image_id": int(row["image_id"]),
            "category_id": int(row["category_id"]),
            "bbox_xyxy": [row["cx"] - row["w"] / 2, row["cy"] - row["h"] / 2,
                          row["cx"] + row["w"] / 2, row["cy"] + row["h"] / 2],
            "_idx": int(row["idx"]),
            "score": float(row["y5_score"]),
            "softrisk": float(row["y5_score"]),  # 占位, 下面覆盖
            "oer": float(row["y5_score"]),
        })
    # 用 nodes 里的 bbox 精确还原(而非中心+wh)
    for i, row in enumerate(df.itertuples()):
        preds[i]["bbox_xyxy"] = [row.cx - row.w / 2, row.cy - row.h / 2,
                                 row.cx + row.w / 2, row.cy + row.h / 2]

    tp_flags = greedy_match(preds, formal, proto)
    print(f"官方 TP 数: {sum(tp_flags.values())}")

    # 1) Y5 基线
    res = {"y5_baseline": frontier_from_scores(preds, "score", tp_flags, n_gt)}

    # 2) SoftRisk v0(位置标签逻辑回归, 三折 cross-fit)
    from sklearn.linear_model import LogisticRegression
    sr_feats = ["y5_score", "aspect", "local_density", "coarse_id"]
    X_sr = df[sr_feats].to_numpy(dtype=float)
    y = df[LABEL].to_numpy(dtype=float)
    folds = df["fold"].to_numpy()
    probs_sr = np.zeros(len(df))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=2026 + held)
        clf.fit(X_sr[tr], y[tr])
        probs_sr[va] = clf.predict_proba(X_sr[va])[:, 1]
    for i in range(len(preds)):
        preds[i]["softrisk"] = float(probs_sr[i])
    res["softrisk_v0"] = frontier_from_scores(preds, "softrisk", tp_flags, n_gt)

    # 3) OER node_validity(HistGB + 丰富特征, 三折 cross-fit)
    from sklearn.ensemble import HistGradientBoostingClassifier
    # coarse_id one-hot
    X = df[FEATURES].copy()
    for c in (0, 1, 2):
        X[f"coarse_{c}"] = (df["coarse_id"] == c).astype(float)
    X = X.drop(columns=["coarse_id"]).to_numpy(dtype=float)
    probs_oer = np.zeros(len(df))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=6, l2_regularization=1.0,
                                             class_weight="balanced", random_state=2026 + held)
        clf.fit(X[tr], y[tr])
        probs_oer[va] = clf.predict_proba(X[va])[:, 1]
    for i in range(len(preds)):
        preds[i]["oer"] = float(probs_oer[i])
    res["oer_node_validity"] = frontier_from_scores(preds, "oer", tp_flags, n_gt)

    # 输出
    print("\n=== A2 OER-v0 固定风险前沿对比 ===")
    for name in ("y5_baseline", "softrisk_v0", "oer_node_validity"):
        r = res[name]
        print(f"[{name}]")
        for k, v in r["recall_at_fdr"].items():
            print(f"  Recall@{k.split('_')[1]}: {v}")
        for k, v in r["fdr_at_recall"].items():
            print(f"  FDR@{k.split('_')[1]}: {v}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_gt": n_gt, "models": res},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {out}")
    # 保存 OER 分数(供下游)
    pd.DataFrame({"idx": df["idx"], "proposal_uid": df["proposal_uid"],
                  "image_id": df["image_id"], "category_id": df["category_id"],
                  "oer_score": [p["oer"] for p in preds],
                  "softrisk": [p["softrisk"] for p in preds],
                  "y5_score": df["y5_score"]}).to_csv(
        out.with_name("oer_scores.csv"), index=False)
    print(f"OER 分数: {out.with_name('oer_scores.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
