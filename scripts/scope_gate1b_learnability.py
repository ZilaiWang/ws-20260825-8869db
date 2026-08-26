#!/usr/bin/env python3
"""Gate 1b: 验证「动作价值」监督目标是否可学习。

读反事实标签（delta_utility / delta_tp / delta_fp），join 候选节点特征，
训练 LightGBM 预测动作价值，回答方案6 Gate 1 的核心问题：

> 新监督目标「动作的官方分数收益」是否比「预测改类正确性」更接近官方评分？

验证三个目标的可学习性：
  1. delta_tp > 0        （动作是否增加 TP，平滑，可直接学）
  2. delta_utility > 0   （动作是否有正 frontier 收益，稀疏，最终目标）
  3. delta_utility 回归  （动作收益量级）

严格 OOF：只使用 formal CV3 外层折；同一候选的 DROP/RELABEL 动作不可拆折，
同一 source group 不可跨训练/验证。禁止随机 StratifiedKFold/KFold。
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
from rsdet.evaluation.grouped_oof import GroupedLedger, iter_outer_splits, split_audit

FEAT_BASE = ["y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
             "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with Path(args.labels).open(encoding="utf-8") as handle:
        labels = [json.loads(line) for line in handle]
    df = pd.DataFrame(labels)
    nodes = pd.read_csv(args.nodes)
    node_map = {int(r.idx): r for r in nodes.itertuples()}
    formal = load_formal_ground_truth(args.formal_crop_manifest)

    image_ledger: dict[int, tuple[int, str]] = {}
    for obj in formal.objects.values():
        current = (int(obj.fold), str(obj.group_id))
        previous = image_ledger.setdefault(int(obj.image_id), current)
        if previous != current:
            raise ValueError(f"formal image assignment is inconsistent: {obj.image_id}")

    # join 特征
    feats = FEAT_BASE + ["has_oto"] if "has_oto" in nodes.columns else FEAT_BASE
    feat_cols = [c for c in feats if c in nodes.columns]
    df["_idx"] = df["candidate_id"].astype(int)
    missing_candidates = sorted(set(df["_idx"]) - set(node_map))
    if missing_candidates:
        raise ValueError(f"labels reference missing candidate IDs: {missing_candidates[:5]}")

    df["image_id"] = df["_idx"].map(lambda value: int(node_map[value].image_id))
    missing_images = sorted(set(df["image_id"]) - set(image_ledger))
    if missing_images:
        raise ValueError(f"candidate images missing from formal CV3: {missing_images[:5]}")
    df["outer_fold"] = df["image_id"].map(lambda value: image_ledger[value][0])
    df["group_id"] = df["image_id"].map(lambda value: image_ledger[value][1])

    ledger = GroupedLedger(
        candidate_ids=df["candidate_id"].astype(str).to_numpy(),
        image_ids=df["image_id"].to_numpy(),
        outer_folds=df["outer_fold"].to_numpy(),
        group_ids=df["group_id"].astype(str).to_numpy(),
    )
    outer_splits = iter_outer_splits(ledger)

    X = np.zeros((len(df), len(feat_cols)))
    for j, c in enumerate(feat_cols):
        X[:, j] = df["_idx"].map(lambda i: float(node_map[i].__getattribute__(c))).to_numpy()

    y_tp = (df["delta_tp"].to_numpy() > 0).astype(int)          # 增加 TP 的动作
    y_pos = (df["delta_utility"].to_numpy() > 0).astype(int)    # 正 frontier 收益的动作
    y_du = df["delta_utility"].to_numpy()                        # 收益量级

    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import average_precision_score, roc_auc_score

    print(f"标签数: {len(df)}, 特征数: {len(feat_cols)}")
    print(f"delta_tp>0 占比: {y_tp.mean():.3f}, delta_utility>0 占比: {y_pos.mean():.4f}")

    out = {
        "protocol": "formal_cv3_outer_oof_v1",
        "n_labels": len(df),
        "n_features": len(feat_cols),
        "features": feat_cols,
        "split_audit": split_audit(ledger, outer=outer_splits),
    }

    def classify_outer(y_target: np.ndarray) -> tuple[float | None, float | None, list[dict]]:
        probabilities = np.full(len(y_target), np.nan, dtype=float)
        fold_rows: list[dict] = []
        for split in outer_splits:
            tr, va = split.train_indices, split.validation_indices
            if len(np.unique(y_target[tr])) < 2:
                raise ValueError(f"outer fold {split.held_out_fold} training labels have one class")
            clf = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.08,
                max_depth=6,
                class_weight="balanced",
                random_state=2026 + split.held_out_fold,
            )
            clf.fit(X[tr], y_target[tr])
            probabilities[va] = clf.predict_proba(X[va])[:, 1]
            fold_auc = fold_ap = None
            if len(np.unique(y_target[va])) == 2:
                fold_auc = float(roc_auc_score(y_target[va], probabilities[va]))
                fold_ap = float(average_precision_score(y_target[va], probabilities[va]))
            fold_rows.append(
                {
                    "held_out_fold": split.held_out_fold,
                    "n_train": len(tr),
                    "n_validation": len(va),
                    "positive_rate": float(y_target[va].mean()),
                    "auc": fold_auc,
                    "ap": fold_ap,
                }
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError("outer OOF probabilities are incomplete")
        if len(np.unique(y_target)) < 2:
            return None, None, fold_rows
        return (
            float(roc_auc_score(y_target, probabilities)),
            float(average_precision_score(y_target, probabilities)),
            fold_rows,
        )

    # 1) delta_tp > 0 分类（平滑，主可学习信号）
    out["tp_auc"], out["tp_ap"], out["tp_by_fold"] = classify_outer(y_tp)
    print(f"[delta_tp>0] AUC={out['tp_auc']:.4f} AP={out['tp_ap']:.4f}" if out["tp_auc"] else "[delta_tp>0] 样本不足")

    # 2) delta_utility > 0 分类（稀疏，最终目标）
    if y_pos.sum() >= 10:
        out["du_auc"], out["du_ap"], out["du_by_fold"] = classify_outer(y_pos)
        print(f"[delta_utility>0] AUC={out['du_auc']:.4f} AP={out['du_ap']:.4f}")
    else:
        out["du_auc"] = None
        print(f"[delta_utility>0] 正样本 {y_pos.sum()} 不足，跳过")

    # 3) delta_utility 回归（量级）
    predictions = np.full(len(y_du), np.nan, dtype=float)
    regression_folds = []
    for split in outer_splits:
        tr, va = split.train_indices, split.validation_indices
        reg = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.08,
            max_depth=6,
            random_state=2026 + split.held_out_fold,
        )
        reg.fit(X[tr], y_du[tr])
        predictions[va] = reg.predict(X[va])
        ss_res = ((y_du[va] - predictions[va]) ** 2).sum()
        ss_tot = ((y_du[va] - y_du[va].mean()) ** 2).sum()
        regression_folds.append(
            {
                "held_out_fold": split.held_out_fold,
                "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
            }
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("outer OOF regression predictions are incomplete")
    ss_res = float(((y_du - predictions) ** 2).sum())
    ss_tot = float(((y_du - y_du.mean()) ** 2).sum())
    out["du_r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else None
    out["du_regression_by_fold"] = regression_folds
    print(f"[delta_utility 回归] R²={out['du_r2']:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
