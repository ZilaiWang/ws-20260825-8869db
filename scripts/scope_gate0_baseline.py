#!/usr/bin/env python3
"""Gate 0: 底座冻结——复现 OER+NMS 基线（deploy 口径，不含改类）。

目标：
1. 训练 OER（FEAT_BASE + has_oto，三折 cross-fit）得到每个候选的排序分数；
2. 候选 category_id 用 Y5 原始预测（不改类），score 用 OER 概率；
3. 同类 NMS + 贪心匹配 + fixed-risk frontier；
4. 输出三折 + 单折的 Recall@FDR，验证与 0.9415 deploy 冠军一致。

这是 SCOPE 的「不可破坏安全基线」。后续所有动作价值都相对此 baseline 计算。
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
from rsdet.scope.official_scorer import FrontierScorer, CandidateView

FEAT_BASE = ["y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
             "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density"]


def build_has_oto(preds, oto_by_img):
    arr = np.zeros(len(preds), dtype=np.int8)
    for i, p in enumerate(preds):
        for q in oto_by_img.get(p["image_id"], []):
            if q["category_id"] != p["category_id"] or q["score"] < 0.5:
                continue
            if compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > 0.5:
                arr[i] = 1
                break
    return arr


def train_oer(nodes: pd.DataFrame, feats: list[str]):
    """三折 cross-fit OER，返回全量 oer 概率（与 a5 完全一致）。"""
    X = nodes[feats].to_numpy(dtype=float)
    y = nodes["is_valid"].to_numpy(dtype=float)
    folds = nodes["fold"].to_numpy()
    from sklearn.ensemble import HistGradientBoostingClassifier
    probs = np.zeros(len(nodes))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6,
            l2_regularization=1.0, class_weight="balanced",
            random_state=2026 + held,
        )
        clf.fit(X[tr], y[tr])
        probs[va] = clf.predict_proba(X[va])[:, 1]
    return probs


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

    nodes = pd.read_csv(args.nodes)
    nodes["fold"] = nodes["image_id"].map(fold_of)

    preds = json.loads(Path(args.preds_d4).read_text(encoding="utf-8"))
    for i, p in enumerate(preds):
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        p["_idx"] = i

    oto_by_img = defaultdict(list)
    for f in (0, 1, 2):
        for p in json.loads(Path(args.oto_dir, f"a5_oto_fold{f}.json").read_text(encoding="utf-8")):
            p["category_id"] = int(p["category_id"])
            p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
            p["image_id"] = int(p["image_id"])
            oto_by_img[p["image_id"]].append(p)

    has_oto = build_has_oto(preds, oto_by_img)
    nodes["has_oto"] = has_oto

    # OER 训练（三折 cross-fit）
    oer = train_oer(nodes, FEAT_BASE + ["has_oto"])

    # base 候选（deploy 口径：category=Y5 原始，score=OER，不改类）
    cand = []
    for i, p in enumerate(preds):
        cand.append(CandidateView(
            _idx=i,
            image_id=p["image_id"],
            category_id=p["category_id"],
            score=float(oer[i]),
            bbox_xyxy=p["bbox_xyxy"],
        ))

    # 三折全局 frontier
    scorer_all = FrontierScorer(proto, formal.boxes)
    fr_all = scorer_all.frontier(cand)
    # 单折
    fr_folds = {}
    for f in (0, 1, 2):
        ids = {i for i, gts in formal.boxes.items() if i in {img for (img, _), o in formal.objects.items() if o.fold == f}}
        sf = FrontierScorer(proto, formal.boxes, ids)
        fr_folds[f] = sf.frontier(cand)

    out = {
        "baseline_note": "OER(12feat+has_oto) + 同类NMS，不含改类（deploy 冠军口径）",
        "full_recall_at_fdr_0.12": fr_all.recall_at_fdr[0.12],
        "full_recall_at_fdr_0.11": fr_all.recall_at_fdr[0.11],
        "full_recall_at_fdr_0.10": fr_all.recall_at_fdr[0.10],
        "n_gt": fr_all.n_gt,
        "n_kept": fr_all.n_kept,
        "n_tp": fr_all.n_tp,
        "n_fp": fr_all.n_fp,
        "per_fold": {str(f): {"r0.12": fr_folds[f].recall_at_fdr[0.12], "n_kept": fr_folds[f].n_kept} for f in (0, 1, 2)},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
