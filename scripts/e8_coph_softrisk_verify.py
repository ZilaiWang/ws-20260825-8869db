#!/usr/bin/env python3
"""E8 COPH fold0 组合验证: SoftRisk(位置标签, 内部5折CV) + aircraft NMS。

公平对比 Y5 fold0 vs COPH fold0(两者都用同样的内部 CV 方法),
回答: COPH 的 Recall 提升在风险过滤后是否保持, FP 是否被压回。

用法:
  python scripts/e8_coph_softrisk_verify.py \
    --coph /tmp/COPH-fold0-std.json \
    --y5 /tmp/Y5-full.json \
    --formal-crop-manifest outputs/.../formal_crop_manifest.csv \
    --project-config configs/project.yaml \
    --output /tmp/e8-verify.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

AIRCRAFT_CIDS = list(range(4, 24))
FEAT_NAMES = ["score", "log_se", "log_ar", "density", "is_air", "is_ship", "is_veh"]


def load_std(path: str) -> list[dict[str, Any]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for p in d:
        p["score"] = float(p["score"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["category_id"] = int(p["category_id"])
    return d


def build_feats(preds: list[dict], formal: Any, proto: Any) -> tuple[np.ndarray, np.ndarray]:
    pb: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        pb[int(p["image_id"])].append(p)
    rows: list[list[float]] = []
    labels: list[float] = []
    for img_id, plist in pb.items():
        gts = formal.boxes.get(img_id, [])
        coarse_density: dict[str, int] = defaultdict(int)
        for p in plist:
            coarse_density[proto.category_mapping[p["category_id"]]] += 1
        for p in plist:
            c = proto.category_mapping[p["category_id"]]
            se = min(p["bbox_xyxy"][2] - p["bbox_xyxy"][0], p["bbox_xyxy"][3] - p["bbox_xyxy"][1])
            w = p["bbox_xyxy"][2] - p["bbox_xyxy"][0]
            h = p["bbox_xyxy"][3] - p["bbox_xyxy"][1]
            ar = math.log(max(w / h, h / w))
            # 位置匹配标签(类别无关)
            lab = 0.0
            for gt in gts:
                thr = proto.iou_thresholds[proto.category_mapping[int(gt["category_id"])]]
                if compute_iou(p["bbox_xyxy"], gt["bbox_xyxy"]) >= thr:
                    lab = 1.0
                    break
            rows.append([p["score"], math.log(max(se, 1.0)), ar, float(coarse_density[c]),
                         1.0 if c == "aircraft" else 0.0,
                         1.0 if c == "ship" else 0.0,
                         1.0 if c == "vehicle" else 0.0])
            labels.append(lab)
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.float64)


def crossfit_5fold(feats: np.ndarray, labels: np.ndarray, seed: int = 2026) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    n = len(feats)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)
    probs = np.zeros(n)
    for fi, va in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(5) if j != fi])
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=2026 + fi)
        clf.fit(feats[tr], labels[tr])
        probs[va] = clf.predict_proba(feats[va])[:, 1]
    return probs


def rerank(preds: list[dict], probs: np.ndarray, beta: float = 0.5, clip_c: float = 3.0) -> list[dict]:
    def _logit(x: float) -> float:
        x = max(min(x, 1 - 1e-6), 1e-6)
        return math.log(x / (1 - x))

    out = []
    for p, pr in zip(preds, probs):
        z = _logit(p["score"]) + max(-clip_c, min(clip_c, beta * _logit(float(pr))))
        q = dict(p)
        q["score"] = 1.0 / (1.0 + math.exp(-z))
        out.append(q)
    return out


def aircraft_nms(preds: list[dict], iou_thr: float = 0.5) -> list[dict]:
    """全细类同类别 NMS(不只 aircraft)——COPH 候选 ship/vehicle 重复框同样需要压。"""
    pb: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        pb[int(p["image_id"])].append(p)
    kept: list[dict] = []
    for img_id, plist in pb.items():
        ordered = sorted(enumerate(plist), key=lambda t: -t[1]["score"])
        suppressed: set[int] = set()
        for i, p in ordered:
            if i in suppressed:
                continue
            kept.append(p)
            for j, q in ordered:
                if j == i or j in suppressed or q["category_id"] != p["category_id"]:
                    continue
                if compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > iou_thr:
                    suppressed.add(j)
    return kept


def evaluate(preds: list[dict], formal: Any, proto: Any, threshold: float, img_filter: set[int] | None = None):
    pb: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        if img_filter is not None and int(p["image_id"]) not in img_filter:
            continue
        pb[int(p["image_id"])].append(p)
    used: dict[int, set[int]] = defaultdict(set)
    tp = fn = 0
    for img_id, gts in formal.boxes.items():
        if img_filter is not None and img_id not in img_filter:
            continue
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
    return tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coph", required=True)
    ap.add_argument("--y5", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))

    coph = load_std(args.coph)
    y5_all = load_std(args.y5)

    # fold0 图集合
    fold0_img = set()
    for (img_id, _), obj in formal.objects.items():
        if obj.fold == 0:
            fold0_img.add(img_id)
    y5_f0 = [p for p in y5_all if int(p["image_id"]) in fold0_img]
    print(f"Y5 fold0 候选: {len(y5_f0)} | COPH fold0 候选: {len(coph)}")

    def _chain(preds: list[dict], label: str) -> dict:
        feats, labs = build_feats(preds, formal, proto)
        probs = crossfit_5fold(feats, labs)
        rr = rerank(preds, probs)
        rr = aircraft_nms(rr)
        out: dict[str, Any] = {"n_raw": len(preds), "n_after": len(rr), "tables": {}}
        for t in (0.01, 0.05, 0.1, 0.2):
            tp, fp, fn = evaluate(rr, formal, proto, t, fold0_img)
            out["tables"][str(t)] = {
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "fdr": fp / (tp + fp) if tp + fp else 0.0,
                "tp": tp, "fp": fp, "fn": fn,
            }
        print(f"\n[{label}] raw={len(preds)} -> 链后={len(rr)}")
        for t in ("0.01", "0.05", "0.1", "0.2"):
            c = out["tables"][t]
            print(f"  t={t}: R={c['recall']:.4f} F={c['fdr']:.4f} (TP={c['tp']} FP={c['fp']} FN={c['fn']})")
        return out

    y5_res = _chain(y5_f0, "Y5 fold0 + SoftRisk+NMS")
    coph_res = _chain(coph, "COPH fold0 + SoftRisk+NMS")

    result = {"y5_fold0": y5_res, "coph_fold0": coph_res, "n_y5": len(y5_f0), "n_coph": len(coph)}
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
