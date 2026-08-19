#!/usr/bin/env python3
"""E3: Soft-Risk v0 —— 用位置匹配概率做有界残差重排(替代 N2 硬 gate)。

对应 DCR²-YOLO 总纲 §6(模块 C RCR 的 v0 版)与实验队列 E3。
目标: 对 Y5 全量候选学习 P(该框与任一 GT 在官方 IoU 下匹配 | 证据),
用有界残差重排分数, 让低分真阳被上调、高分背景被下调——而不是硬删。

协议: 三折 cross-fit(每折用另外两折拟合逻辑回归, 本折应用), 与 N2 一致。
评估: 重排后多档阈值截断 + 官方细类匹配口径, 输出阈值曲线对比 Y5 基线。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

THRESHOLDS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
FEATURE_COLS = ["score", "log_se", "log_ar", "density", "fold",
                "is_air", "is_ship", "is_veh"]
FOLD_COL_IDX = 4  # fold 是特征, 但训练时不用 fold 预测(避免泄漏), 仅用于 split


def build_features(preds: list[dict[str, Any]],
                   formal: Any, proto: Any) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """构建特征矩阵 + 位置匹配标签(类别无关官方 IoU)。"""
    pb: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)

    def feat(p: dict[str, Any], coarse: str, density: int, fold: int) -> list[float]:
        x0, y0, x1, y1 = p["bbox_xyxy"]
        se = min(x1 - x0, y1 - y0)
        w, h = x1 - x0, y1 - y0
        ar = math.log(max(w / h, h / w))
        return [p["score"], math.log(max(se, 1.0)), ar, float(density), float(fold),
                1.0 if coarse == "aircraft" else 0.0,
                1.0 if coarse == "ship" else 0.0,
                1.0 if coarse == "vehicle" else 0.0]

    def loc_label(p: dict[str, Any], gts: list[dict[str, Any]]) -> float:
        for gt in gts:
            thr = proto.iou_thresholds[proto.category_mapping[int(gt["category_id"])]]
            if compute_iou(p["bbox_xyxy"], gt["bbox_xyxy"]) >= thr:
                return 1.0
        return 0.0

    rows: list[list[float]] = []
    labels: list[float] = []
    folds: list[int] = []
    for img_id, plist in pb.items():
        gts = formal.boxes.get(img_id, [])
        obj0 = formal.objects.get((img_id, 0))
        fold = obj0.fold if obj0 else 0
        coarse_density: dict[str, int] = defaultdict(int)
        for p in plist:
            coarse_density[proto.category_mapping[int(p["category_id"])]] += 1
        for p in plist:
            coarse = proto.category_mapping[int(p["category_id"])]
            rows.append(feat(p, coarse, coarse_density[coarse], fold))
            labels.append(loc_label(p, gts))
            folds.append(fold)
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.float64), folds


def crossfit(feats: np.ndarray, labels: np.ndarray, folds: list[int],
             seed: int = 20260820) -> list[np.ndarray]:
    """三折 cross-fit, 返回每折该折样本的 P(loc)。"""
    feats_no_fold = np.delete(feats, FOLD_COL_IDX, axis=1)
    probs = np.zeros(len(feats))
    for held in (0, 1, 2):
        tr_idx = [i for i, f in enumerate(folds) if f != held]
        va_idx = [i for i, f in enumerate(folds) if f == held]
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                 random_state=seed + held)
        clf.fit(feats_no_fold[tr_idx], labels[tr_idx])
        probs[va_idx] = clf.predict_proba(feats_no_fold[va_idx])[:, 1]
    return probs


def rerank_score(orig_score: np.ndarray, p_loc: np.ndarray, beta: float,
                 clip_c: float) -> np.ndarray:
    """z_submit = logit(s) + clip(beta * logit(p_loc), -c, c), 再 sigmoid。"""
    def logit(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))
    z = logit(orig_score) + np.clip(beta * logit(p_loc), -clip_c, clip_c)
    return 1.0 / (1.0 + np.exp(-z))


def evaluate_threshold(preds: list[dict[str, Any]], formal: Any, proto: Any) -> dict:
    pb: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    used_by_img: dict[int, set[int]] = defaultdict(set)
    tp = fn = 0
    for img_id, gts in formal.boxes.items():
        plist = pb.get(img_id, [])
        used = used_by_img[img_id]
        ordered = sorted(enumerate(plist), key=lambda t: -t[1]["score"])
        for gt in gts:
            cid = int(gt["category_id"])
            thr = proto.iou_thresholds[proto.category_mapping[cid]]
            best_iou, best_idx = 0.0, None
            for idx, p in ordered:
                if idx in used or p["category_id"] != cid:
                    continue
                iou = compute_iou(p["bbox_xyxy"], gt["bbox_xyxy"])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx is not None and best_iou >= thr:
                used.add(best_idx)
                tp += 1
            else:
                fn += 1
    fp = 0
    for img_id, plist in pb.items():
        for i, p in enumerate(plist):
            if i not in used_by_img[img_id]:
                fp += 1
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fdr = fp / (tp + fp) if (tp + fp) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "recall": recall, "fdr": fdr}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, help="Y5 全量候选 JSON")
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--beta", type=float, default=0.5, help="残差权重")
    ap.add_argument("--clip-c", type=float, default=3.0, help="残差裁剪")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))

    feats, labels, folds = build_features(preds, formal, proto)
    print(f"特征: {feats.shape}, 位置匹配率: {labels.mean():.3f}")
    p_loc = crossfit(feats, labels, folds)
    print(f"P(loc) 分布: mean={p_loc.mean():.3f} P10={np.percentile(p_loc,10):.3f} "
          f"P90={np.percentile(p_loc,90):.3f}")

    # AUC 诊断
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, p_loc)
    print(f"P(loc) AUC: {auc:.4f}")

    scores = np.array([p["score"] for p in preds], dtype=np.float64)
    new_scores = rerank_score(scores, p_loc, args.beta, args.clip_c)

    rows_out: dict[str, Any] = {
        "experiment": "E3-SOFT-RISK-V0",
        "protocol": "三折 cross-fit 逻辑回归 + 有界残差重排(替代 N2 硬 gate)",
        "n_candidates": len(preds),
        "loc_match_rate": float(labels.mean()),
        "p_loc_auc": float(auc),
        "beta": args.beta,
        "clip_c": args.clip_c,
        "threshold_curves": {},
    }
    for t in THRESHOLDS:
        base = evaluate_threshold(
            [p for p in preds if p["score"] >= t], formal, proto)
        new = evaluate_threshold(
            [dict(p, score=new_scores[i]) for i, p in enumerate(preds)
             if new_scores[i] >= t], formal, proto)
        rows_out["threshold_curves"][str(t)] = {
            "baseline": base, "softrisk": new,
            "d_recall": round(new["recall"] - base["recall"], 4),
            "d_fdr": round(new["fdr"] - base["fdr"], 4),
        }
        print(f"t={t:5} base R={base['recall']:.4f}/F={base['fdr']:.4f} | "
              f"softR R={new['recall']:.4f}/F={new['fdr']:.4f} | "
              f"ΔR={new['recall']-base['recall']:+.4f} ΔF={new['fdr']-base['fdr']:+.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows_out, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\n已写: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
