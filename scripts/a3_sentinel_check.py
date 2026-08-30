#!/usr/bin/env python3
"""A3 泛化验证: 排除 sentinel 图训练 OER+路由器, 在 555 冻结图上评估。

确认"OER 排序 + 改类"的收益不是对旧 OOF 的记忆(方案4 §12.3)。
- 训练集 = 4481 − 555(非 sentinel), cross-fit;
- 评估集 = 555 sentinel 图, 只评一次;
- 对比: sentinel 上 OER+NMS(不改类) vs OER+NMS+全改。

用法:
  python scripts/a3_sentinel_check.py \
    --nodes /tmp/a1-object-graph/nodes.csv \
    --predictions /tmp/Y5-full.json \
    --sentinel outputs/PROSPECTIVE_SENTINEL_20260820.json \
    --formal-crop-manifest outputs/.../formal_crop_manifest.csv \
    --project-config configs/project.yaml \
    --output /tmp/a3_sentinel.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_frontier import (
    min_fdr_at_recall,
    official_fixed_risk_frontier,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

FEATURES = ["y5_score", "crop_top1", "crop_margin", "crop_entropy",
            "crop_top1_class", "detector_crop_agree",
            "w", "h", "area", "short_edge", "aspect", "local_density"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--sentinel", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    sentinel = json.loads(Path(args.sentinel).read_text(encoding="utf-8"))
    sentinel_ids = set(sentinel["frozen_image_ids"])
    all_ids = set(formal.boxes.keys())
    train_ids = all_ids - sentinel_ids
    n_gt = sum(len(gts) for i, gts in formal.boxes.items() if i in sentinel_ids)

    nodes = pd.read_csv(args.nodes)
    preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    for i, p in enumerate(preds):
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        p["_idx"] = i

    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}
    nodes["fold"] = nodes["image_id"].map(fold_of)
    nodes["is_train"] = nodes["image_id"].isin(train_ids)

    # 训练 OER(只在 train_ids 上 cross-fit)
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr_df = nodes[nodes["is_train"]]
    X = tr_df[FEATURES].to_numpy(dtype=float)
    y = tr_df["is_valid"].to_numpy(dtype=float)
    folds = tr_df["fold"].to_numpy()
    oer = np.zeros(len(tr_df))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                             l2_regularization=1.0, class_weight="balanced",
                                             random_state=2026 + held)
        clf.fit(X[tr], y[tr])
        oer[va] = clf.predict_proba(X[va])[:, 1]
    # 用训练好的模型对 sentinel 图预测(用最后一折模型)
    clf_all = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                             l2_regularization=1.0, class_weight="balanced",
                                             random_state=2026)
    clf_all.fit(X, y)
    sent_df = nodes[~nodes["is_train"]]
    Xs = sent_df[FEATURES].to_numpy(dtype=float)
    sent_oer = clf_all.predict_proba(Xs)[:, 1]

    node_map = {int(r.idx): r for r in nodes.itertuples()}
    oer_map = {}
    for j, (_, r) in enumerate(tr_df.iterrows()):
        oer_map[int(r.idx)] = float(oer[j])
    for j, (_, r) in enumerate(sent_df.iterrows()):
        oer_map[int(r.idx)] = float(sent_oer[j])

    # 构建 sentinel 图候选(带 oer 分数)
    sent_preds = [dict(p) for p in preds if p["image_id"] in sentinel_ids]
    for p in sent_preds:
        p["score"] = oer_map.get(p["_idx"], p["score"])
        p["oer"] = p["score"]
        p["source_prediction_index"] = p["_idx"]

    def evaluate(records):
        result = official_fixed_risk_frontier(
            gt_boxes=formal.boxes,
            predictions=records,
            category_mapping=proto.category_mapping,
            iou_thresholds=proto.iou_thresholds,
            image_ids=sentinel_ids,
            fdr_levels=(0.10, 0.12, 0.15),
            nms_iou=0.50,
        )
        br = {level: result.points[level].recall for level in (0.10, 0.12, 0.15)}
        bf = min_fdr_at_recall(result, recall_levels=(0.94, 0.95, 0.96))
        return br, bf

    # 基线: 不改类
    br0, bf0 = evaluate(sent_preds)

    # 可部署全改：所有 detector/crop 分歧均使用 crop top1，不查 GT。
    changed = 0
    sent_preds2 = [dict(p) for p in sent_preds]
    for p in sent_preds2:
        r = node_map.get(p["_idx"])
        if r is None:
            continue
        top1 = int(r.crop_top1_class)
        if top1 >= 0 and top1 != p["category_id"]:
            p["category_id"] = top1
            changed += 1
    br1, bf1 = evaluate(sent_preds2)

    print(f"sentinel 图: {len(sentinel_ids)}, GT: {n_gt}")
    print(f"全改 {changed} 个（部署字段决定，无 GT oracle）")
    print(f"\n[基线 OER+NMS]  R@FDR=.12={br0[0.12]:.4f} | FDR@R=.94={bf0[0.94]:.4f}")
    print(f"[全改 OER+NMS]  R@FDR=.12={br1[0.12]:.4f} | FDR@R=.94={bf1[0.94]:.4f}")
    print(f"Δ: R@FDR=.12 {br1[0.12]-br0[0.12]:+.4f} | FDR@R=.94 {bf1[0.94]-bf0[0.94]:+.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "sentinel": {"n_images": len(sentinel_ids), "n_gt": n_gt},
        "baseline": {"r_at_fdr_0.12": br0[0.12], "fdr_at_r_0.94": bf0[0.94]},
        "full_relabel": {"r_at_fdr_0.12": br1[0.12], "fdr_at_r_0.94": bf1[0.94],
                         "changed": changed},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
