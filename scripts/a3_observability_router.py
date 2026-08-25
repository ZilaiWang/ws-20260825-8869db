#!/usr/bin/env python3
"""A3: 可观测性路由器(机制验证版)。

核心(方案4 §五.3): 预测"细类是否可观测"(g_obs = P(视觉证据足以支持细类判断)),
再决定相信 crop 还是 YOLO, 替代 E5 规则改类。

机制验证(本地可做, 用已有特征):
  对位置匹配 GT 的候选, 在 YOLO 错细类的子集上, 训练二分类"crop 是否正确",
  看 crop 对错是否可由可观测性特征预测(AUC)。若可预测, 则路由器有信号,
  可据此只改"crop 可信"的候选, 压低 broken TP。

完整版(待 GPU): 补充 local contrast / blur / edge energy / YOLO logits margin /
D4 稳定性等视觉质量特征。

用法:
  python scripts/a3_observability_router.py \
    --nodes /tmp/a1-object-graph/nodes.csv \
    --predictions /tmp/Y5-full.json \
    --formal-crop-manifest outputs/.../formal_crop_manifest.csv \
    --project-config configs/project.yaml \
    --output /tmp/a3_router.json
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

OBS_FEATURES = ["short_edge", "area", "aspect", "crop_margin", "crop_entropy",
                "crop_top1", "detector_crop_agree", "local_density"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))

    nodes = pd.read_csv(args.nodes)
    # 还原候选 bbox(用于位置关联)
    preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    for i, p in enumerate(preds):
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        p["_idx"] = i

    # 位置关联(类无关): 每个候选 -> 位置覆盖的 GT 细类集合
    pb = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    gt_fine_of: dict[int, set[int]] = defaultdict(set)
    for img_id, gts in formal.boxes.items():
        for p in pb.get(img_id, []):
            for g in gts:
                thr = proto.iou_thresholds[proto.category_mapping[int(g["category_id"])]]
                if compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr:
                    gt_fine_of[p["_idx"]].add(int(g["category_id"]))

    # 合并 nodes(带 crop 特征)
    node_map = {int(r.idx): r for r in nodes.itertuples()}
    rows = []
    for p in preds:
        if p["_idx"] not in gt_fine_of:
            continue  # 只关心位置匹配 GT 的候选
        gt_fines = gt_fine_of[p["_idx"]]
        r = node_map.get(p["_idx"])
        if r is None:
            continue
        yolo_fine = p["category_id"]
        crop_top1 = int(r.crop_top1_class)
        yolo_correct = 1 if yolo_fine in gt_fines else 0
        crop_correct = 1 if crop_top1 in gt_fines else 0
        # 可观测性特征
        feat = [float(r.short_edge), float(r.area), float(r.aspect),
                float(r.crop_margin), float(r.crop_entropy), float(r.crop_top1),
                float(r.detector_crop_agree), float(r.local_density)]
        # 标签分类: trust_crop(yolo错crop对) / trust_detector(yolo对crop错) / agree
        if yolo_correct and not crop_correct:
            label = "trust_detector"
        elif (not yolo_correct) and crop_correct:
            label = "trust_crop"
        else:
            label = "agreement"
        rows.append(feat + [yolo_correct, crop_correct, label])

    cols = OBS_FEATURES + ["yolo_correct", "crop_correct", "label"]
    df = pd.DataFrame(rows, columns=cols)
    print(f"位置匹配 GT 的候选: {len(df)}")
    print(df["label"].value_counts().to_dict())

    # 机制验证: 在 yolo 错的候选上, 训练"crop 是否正确"二分类
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    sub = df[df["yolo_correct"] == 0]
    print(f"  YOLO 错细类的候选(可改类对象): {len(sub)}, 其中 crop 对: {sub['crop_correct'].sum()}")
    if len(sub) > 50 and sub["crop_correct"].nunique() > 1:
        X = sub[OBS_FEATURES].to_numpy(dtype=float)
        y = sub["crop_correct"].to_numpy()
        # 5 折 CV
        aucs = []
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(sub))
        splits = np.array_split(idx, 5)
        for fi, va in enumerate(splits):
            tr = np.concatenate([splits[j] for j in range(5) if j != fi])
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                                 max_depth=4, random_state=fi)
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1]))
        print(f"  路由器信号: 预测'crop 是否正确' 5折 AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    else:
        print("  样本不足或单一类, 跳过 AUC")

    # 特征区分度: trust_crop vs trust_detector 的可观测性特征均值
    print("\n=== 可观测性特征区分度(trust_crop vs trust_detector) ===")
    for f in OBS_FEATURES:
        tc = df[df["label"] == "trust_crop"][f].mean()
        td = df[df["label"] == "trust_detector"][f].mean()
        ag = df[df["label"] == "agreement"][f].mean()
        print(f"  {f}: trust_crop={tc:.3f} trust_detector={td:.3f} agreement={ag:.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_matched": len(df),
        "label_dist": df["label"].value_counts().to_dict(),
        "n_yolo_wrong": int((df["yolo_correct"] == 0).sum()),
        "n_crop_right_among_wrong": int(sub["crop_correct"].sum()),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
