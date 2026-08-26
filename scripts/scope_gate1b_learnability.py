#!/usr/bin/env python3
"""Gate 1b: 验证「动作价值」监督目标是否可学习。

读反事实标签（delta_utility / delta_tp / delta_fp），join 候选节点特征，
训练 LightGBM 预测动作价值，回答方案6 Gate 1 的核心问题：

> 新监督目标「动作的官方分数收益」是否比「预测改类正确性」更接近官方评分？

验证三个目标的可学习性：
  1. delta_tp > 0        （动作是否增加 TP，平滑，可直接学）
  2. delta_utility > 0   （动作是否有正 frontier 收益，稀疏，最终目标）
  3. delta_utility 回归  （动作收益量级）

严格 OOF：按 fold cross-fit（若标签含多折）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FEAT_BASE = ["y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
             "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    labels = [json.loads(l) for l in open(args.labels, encoding="utf-8")]
    df = pd.DataFrame(labels)
    nodes = pd.read_csv(args.nodes)
    node_map = {int(r.idx): r for r in nodes.itertuples()}

    # join 特征
    feats = FEAT_BASE + ["has_oto"] if "has_oto" in nodes.columns else FEAT_BASE
    feat_cols = [c for c in feats if c in nodes.columns]
    df["_idx"] = df["candidate_id"].astype(int)

    X = np.zeros((len(df), len(feat_cols)))
    for j, c in enumerate(feat_cols):
        X[:, j] = df["_idx"].map(lambda i: float(node_map[i].__getattribute__(c))).to_numpy()

    y_tp = (df["delta_tp"].to_numpy() > 0).astype(int)          # 增加 TP 的动作
    y_pos = (df["delta_utility"].to_numpy() > 0).astype(int)    # 正 frontier 收益的动作
    y_du = df["delta_utility"].to_numpy()                        # 收益量级

    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import roc_auc_score, average_precision_score

    print(f"标签数: {len(df)}, 特征数: {len(feat_cols)}")
    print(f"delta_tp>0 占比: {y_tp.mean():.3f}, delta_utility>0 占比: {y_pos.mean():.4f}")

    out = {"n_labels": len(df), "n_features": len(feat_cols)}

    # 1) delta_tp > 0 分类（平滑，主可学习信号）
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs, aps = [], []
    for tr, va in skf.split(X, y_tp):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                             class_weight="balanced", random_state=42)
        clf.fit(X[tr], y_tp[tr])
        p = clf.predict_proba(X[va])[:, 1]
        if y_tp[va].sum() > 0 and (y_tp[va] == 0).sum() > 0:
            aucs.append(roc_auc_score(y_tp[va], p))
            aps.append(average_precision_score(y_tp[va], p))
    out["tp_auc"] = float(np.mean(aucs)) if aucs else None
    out["tp_ap"] = float(np.mean(aps)) if aps else None
    print(f"[delta_tp>0] AUC={out['tp_auc']:.4f} AP={out['tp_ap']:.4f}" if out["tp_auc"] else f"[delta_tp>0] 样本不足")

    # 2) delta_utility > 0 分类（稀疏，最终目标）
    if y_pos.sum() >= 10:
        aucs2, aps2 = [], []
        for tr, va in skf.split(X, y_pos):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                                 class_weight="balanced", random_state=42)
            clf.fit(X[tr], y_pos[tr])
            p = clf.predict_proba(X[va])[:, 1]
            aucs2.append(roc_auc_score(y_pos[va], p))
            aps2.append(average_precision_score(y_pos[va], p))
        out["du_auc"] = float(np.mean(aucs2))
        out["du_ap"] = float(np.mean(aps2))
        print(f"[delta_utility>0] AUC={out['du_auc']:.4f} AP={out['du_ap']:.4f}")
    else:
        out["du_auc"] = None
        print(f"[delta_utility>0] 正样本 {y_pos.sum()} 不足，跳过")

    # 3) delta_utility 回归（量级）
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    r2s = []
    for tr, va in kf.split(X):
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=6, random_state=42)
        reg.fit(X[tr], y_du[tr])
        p = reg.predict(X[va])
        ss_res = ((y_du[va] - p) ** 2).sum()
        ss_tot = ((y_du[va] - y_du[va].mean()) ** 2).sum()
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    out["du_r2"] = float(np.mean(r2s))
    print(f"[delta_utility 回归] R²={out['du_r2']:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
