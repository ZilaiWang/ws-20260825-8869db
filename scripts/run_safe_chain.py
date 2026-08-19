#!/usr/bin/env python3
"""Safe 链正式 pipeline(E11 组合的基础组件, DCR²-YOLO)。

输入: E1 重排后预测 JSON(e1_y5_rerank_screen.py 输出)或任意 predictions。
流程:
  1. E2: aircraft 同细类 post-rerank NMS@0.5(按重排后 score 排序, 零 TP 损失);
  2. E3: Soft-Risk 有界残差重排(可选, 用 sklearn 逻辑回归三折 cross-fit);
  3. 输出最终预测 + 阈值曲线评估。

用法示例:
  python scripts/run_safe_chain.py \
    --predictions e1_reranked_predictions.json \
    --formal-crop-manifest outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv \
    --project-config configs/project.yaml \
    --nms-iou 0.5 --soft-risk --beta 0.5 --clip-c 3.0 \
    --output /tmp/safe_chain.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

AIRCRAFT_CIDS = list(range(4, 24))
THRESHOLDS = (0.05, 0.1, 0.2)


def post_rerank_nms(preds: list[dict], iou_thr: float = 0.5) -> list[dict]:
    """aircraft 同细类 NMS(按 score 降序贪心; 只抑制 aircraft 同细类高 IoU 框)。"""
    pb: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    kept: list[dict] = []
    for img_id, plist in pb.items():
        ordered = sorted(enumerate(plist), key=lambda t: -t[1]["score"])
        suppressed: set[int] = set()
        for i, p in ordered:
            if i in suppressed:
                continue
            kept.append(p)
            if p["category_id"] not in AIRCRAFT_CIDS:
                continue
            for j, q in ordered:
                if j == i or j in suppressed or q["category_id"] != p["category_id"]:
                    continue
                if compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > iou_thr:
                    suppressed.add(j)
    return kept


def evaluate(preds: list[dict], formal: Any, proto: Any, threshold: float) -> dict:
    pb: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    used: dict[int, set[int]] = defaultdict(set)
    tp = fn = 0
    for img_id, gts in formal.boxes.items():
        plist = [p for p in pb.get(img_id, []) if p["score"] >= threshold]
        ordered = sorted(enumerate(plist), key=lambda t: -t[1]["score"])
        for g in gts:
            cid = int(g["category_id"])
            thr = proto.iou_thresholds[proto.category_mapping[cid]]
            bi, bix = 0.0, None
            for idx, p in ordered:
                if idx in used[img_id] or p["category_id"] != cid:
                    continue
                iou = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if iou > bi:
                    bi, bix = iou, idx
            if bix is not None and bi >= thr:
                used[img_id].add(bix)
                tp += 1
            else:
                fn += 1
    fp = 0
    for img_id, plist in pb.items():
        for i, p in enumerate(plist):
            if p["score"] >= threshold and i not in used[img_id]:
                fp += 1
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fdr = fp / (tp + fp) if (tp + fp) else 0.0
    return {"recall": round(recall, 4), "fdr": round(fdr, 4),
            "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--soft-risk", action="store_true", help="启用 E3 Soft-Risk 重排")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--clip-c", type=float, default=3.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from rsdet.analysis.oof_detection import load_formal_ground_truth
    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))

    preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    print(f"输入候选: {len(preds)}")

    # E2 NMS
    preds = post_rerank_nms(preds, args.nms_iou)
    print(f"E2 NMS 后: {len(preds)}")

    # E3 Soft-Risk(可选)
    if args.soft_risk:
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        pb: dict[int, list[dict]] = defaultdict(list)
        for p in preds:
            pb[p["image_id"]].append(p)

        def feats(p: dict, coarse: str, density: int, fold: int) -> list[float]:
            x0, y0, x1, y1 = p["bbox_xyxy"]
            se = min(x1 - x0, y1 - y0)
            w, h = x1 - x0, y1 - y0
            return [p["score"], math.log(max(se, 1.0)), math.log(max(w / h, h / w)),
                    float(density), float(fold),
                    1.0 if coarse == "aircraft" else 0.0,
                    1.0 if coarse == "ship" else 0.0,
                    1.0 if coarse == "vehicle" else 0.0]

        rows, labels, folds = [], [], []
        for img_id, plist in pb.items():
            gts = formal.boxes.get(img_id, [])
            obj0 = formal.objects.get((img_id, 0))
            fold = obj0.fold if obj0 else 0
            density: dict[str, int] = defaultdict(int)
            for p in plist:
                density[proto.category_mapping[int(p["category_id"])]] += 1
            for p in plist:
                coarse = proto.category_mapping[int(p["category_id"])]
                label = 0.0
                for g in gts:
                    thr = proto.iou_thresholds[proto.category_mapping[int(g["category_id"])]]
                    if compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr:
                        label = 1.0
                        break
                rows.append(feats(p, coarse, density[coarse], fold))
                labels.append(label)
                folds.append(fold)
        X = np.array(rows, dtype=np.float64)
        y = np.array(labels, dtype=np.float64)
        X_nf = np.delete(X, 4, axis=1)
        probs = np.zeros(len(preds))
        for held in (0, 1, 2):
            tr = [i for i, f in enumerate(folds) if f != held]
            va = [i for i, f in enumerate(folds) if f == held]
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                     random_state=20260820 + held)
            clf.fit(X_nf[tr], y[tr])
            probs[va] = clf.predict_proba(X_nf[va])[:, 1]

        def logit(x: np.ndarray) -> np.ndarray:
            x = np.clip(x, 1e-6, 1 - 1e-6)
            return np.log(x / (1 - x))

        scores = np.array([p["score"] for p in preds])
        new_scores = 1.0 / (1.0 + np.exp(-(logit(scores) + np.clip(
            args.beta * logit(probs), -args.clip_c, args.clip_c))))
        for i, p in enumerate(preds):
            p["score"] = float(new_scores[i])
        print("E3 Soft-Risk 已应用")

    # 阈值曲线
    curves = {}
    for t in THRESHOLDS:
        curves[str(t)] = evaluate(preds, formal, proto, t)
        m = curves[str(t)]
        print(f"t={t}: R={m['recall']:.4f} F={m['fdr']:.4f} tp={m['tp']} fp={m['fp']} fn={m['fn']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "SAFE-CHAIN",
        "n_input": len(json.loads(Path(args.predictions).read_text(encoding="utf-8"))),
        "n_after_nms": len(preds),
        "threshold_curves": curves,
        "components": ["R1-R3-fuse", "post-rerank-NMS", "soft-risk" if args.soft_risk else None],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
