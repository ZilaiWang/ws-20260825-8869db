#!/usr/bin/env python3
"""A0: 完整 PR frontier 重算(固定风险前沿)。

对照 HERA-YOLO 方案4 §十二: 主指标不再用固定阈值, 改为
  - Recall@FDR=0.10/0.12/0.15
  - FDR@Recall=0.94/0.95/0.96
  - Macro Recall / 重点类 Recall / FP 分解

用法:
  python scripts/a0_pr_frontier.py \
    --predictions /tmp/Y5-3fold-fullchain_predictions.json --label "Y5链" \
    --predictions /tmp/COPH-3fold-safechain_predictions.json --label "COPH链" \
    --formal-crop-manifest outputs/.../formal_crop_manifest.csv \
    --project-config configs/project.yaml \
    --output /tmp/a0_frontier.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

# 25 细类名称(来自 formal manifest class_id/class_name)
FINE_NAMES = {
    0: "HM", 1: "LQS", 2: "QHS", 3: "MS", 4: "A1_SU-35", 5: "A2_C-130",
    6: "A3_C-17", 7: "A4_C-5", 8: "A5_F-16", 9: "A6_TU-160", 10: "A7_E-3",
    11: "A8_B-52", 12: "A9_P-3C", 13: "A10_B-1B", 14: "A11_E-8", 15: "A12_TU-22",
    16: "A13_F-15", 17: "A14_KC-135", 18: "A15_F-22", 19: "A16_FA-18",
    20: "A17_TU-95", 21: "A18_KC-10", 22: "A19_SU-34", 23: "A20_SU-24", 24: "FSC",
}
FOCUS_CLASSES = (0, 1, 9, 18, 24)  # HM/LQS/TU-160/F-22/FSC


def load_predictions(path: str) -> list[dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for i, p in enumerate(d):
        p["score"] = float(p["score"])
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        p["_idx"] = i
    return d


def greedy_match(preds: list[dict], formal: Any, proto: Any) -> dict:
    """贪心匹配(按 score 降序, 同细类, 粗类 IoU 阈值)。返回 {候选 _idx: True(TP)}。

    同时返回每类 TP/FN 计数用于 macro/focus recall。
    """
    pb: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    tp_flags: dict[int, bool] = {}
    class_tp: dict[int, int] = defaultdict(int)
    class_fn: dict[int, int] = defaultdict(int)
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
                class_tp[cid] += 1
            else:
                class_fn[cid] += 1
    return tp_flags, class_tp, class_fn


def build_curve(preds: list[dict], tp_flags: dict, n_gt: int) -> list[tuple]:
    """按 score 降序累计 TP/FP → (score, recall, fdr, tp, fp) 曲线。"""
    ordered = sorted(preds, key=lambda p: -p["score"])
    tp = fp = 0
    curve = []
    for p in ordered:
        if tp_flags.get(p["_idx"]):
            tp += 1
        else:
            fp += 1
        curve.append((p["score"], tp / n_gt, fp / (tp + fp) if tp + fp else 0.0, tp, fp))
    return curve


def recall_at_fdr(curve: list[tuple], fdr_target: float) -> float:
    """Recall@FDR=target: 满足 fdr<=target 的最大 recall(曲线单调性下取最后达标点)。"""
    best = 0.0
    for _, r, f, _, _ in curve:
        if f <= fdr_target:
            best = max(best, r)
    return best


def fdr_at_recall(curve: list[tuple], recall_target: float) -> float:
    """FDR@Recall=target: 满足 recall>=target 的最小 fdr。"""
    best = 1.0
    for _, r, f, _, _ in curve:
        if r >= recall_target:
            best = min(best, f)
    return best


def metrics_at_fdr(curve: list[tuple], fdr_target: float,
                   class_tp: dict, class_fn: dict) -> dict:
    """在 FDR<=target 的最大 recall 工作点, 报告 macro/focus recall。"""
    best_recall = 0.0
    best_idx = -1
    for i, (_, r, f, _, _) in enumerate(curve):
        if f <= fdr_target and r >= best_recall:
            best_recall = r
            best_idx = i
    # 该工作点的 tp/fp(用曲线索引)
    if best_idx < 0:
        return {}
    _, r, f, tp, fp = curve[best_idx]
    # class recall 是"所有 GT"口径(候选 floor 级, 与阈值无关)
    macros = []
    for coarse in ("aircraft", "ship", "vehicle"):
        cls = [c for c in FINE_NAMES if proto_mapping[c] == coarse]
        mtp = sum(class_tp.get(c, 0) for c in cls)
        mfn = sum(class_fn.get(c, 0) for c in cls)
        macros.append(mtp / (mtp + mfn) if mtp + mfn else 0.0)
    focus = {FINE_NAMES[c]: (class_tp.get(c, 0), class_fn.get(c, 0)) for c in FOCUS_CLASSES}
    return {"recall": r, "fdr": f, "tp": tp, "fp": fp,
            "macro_recall": sum(macros) / len(macros),
            "focus": {k: {"tp": v[0], "fn": v[1],
                          "recall": v[0] / (v[0] + v[1]) if v[0] + v[1] else 0.0}
                      for k, v in focus.items()}}


proto_mapping: dict[int, str] = {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", action="append", nargs=2, metavar=("PATH", "LABEL"),
                    required=True, help="可重复: --predictions 路径 标签")
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    global proto_mapping
    proto_mapping = {c: proto.category_mapping[c] for c in range(25)}

    n_gt = sum(len(gts) for gts in formal.boxes.values())
    print(f"GT 总数: {n_gt}")

    results = {}
    for path, label in args.predictions:
        preds = load_predictions(path)
        tp_flags, class_tp, class_fn = greedy_match(preds, formal, proto)
        curve = build_curve(preds, tp_flags, n_gt)
        r = {
            "n_candidates": len(preds),
            "candidate_floor_recall": len(tp_flags) / n_gt,
            "recall_at_fdr": {f"fdr_{v}": recall_at_fdr(curve, v) for v in (0.10, 0.12, 0.15)},
            "fdr_at_recall": {f"rec_{v}": fdr_at_recall(curve, v) for v in (0.94, 0.95, 0.96)},
            "at_fdr_0.12": metrics_at_fdr(curve, 0.12, class_tp, class_fn),
            "at_fdr_0.15": metrics_at_fdr(curve, 0.15, class_tp, class_fn),
        }
        results[label] = r
        print(f"\n=== {label} ===")
        print(f"  候选: {len(preds)} | candidate-floor Recall: {r['candidate_floor_recall']:.4f}")
        for k, v in r["recall_at_fdr"].items():
            print(f"  Recall@{k.split('_')[1]}: {v:.4f}")
        for k, v in r["fdr_at_recall"].items():
            print(f"  FDR@{k.split('_')[1]}: {v:.4f}")
        m = r["at_fdr_0.12"]
        if m:
            print(f"  @FDR=0.12: Recall={m['recall']:.4f} FDR={m['fdr']:.4f} "
                  f"Macro={m['macro_recall']:.4f} TP={m['tp']} FP={m['fp']}")
            focus_str = " ".join(f"{k}={v['recall']:.3f}" for k, v in m["focus"].items())
            print(f"    focus: {focus_str}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_gt": n_gt, "models": results},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
