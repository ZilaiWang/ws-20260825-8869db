#!/usr/bin/env python3
"""E1: 在 Y5 proposals 上重放 R1 提案重排变体(DCR²-YOLO 实验 E1)。

复用 R1-0 的变体网格(R1/R2 gate/R3 fuse/R4 gate+fuse), 但输入换成 Y5
全量候选 + Y5 crop logits(P03-F 教师推理), 不做 M1 的 C2 分数校准
(Y5 分数域需单独重拟合, 后续做)。

协议: 三折 cross-fit(两折选变体+阈值, 第三折评估), 官方细类口径。
输出: 各变体在 Y5 上的 pooled/macro/aircraft Recall·FDR + 配对转移账本。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.proposal_reranking import (
    apply_variant,
    build_variants,
    load_logits,
    load_proposal_manifest,
    logits_to_probabilities,
)
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

THRESHOLD_START, THRESHOLD_STOP, THRESHOLD_STEP = 0.001, 0.301, 0.01


def scan_threshold(gt: dict[int, list[dict]], preds: dict[int, list[dict]],
                   proto: Any) -> tuple[float, dict]:
    """全局阈值扫描(官方细类口径), 返回 (最优阈值, 指标)。"""
    best = None
    best_t = 0.001
    t = THRESHOLD_START
    while t <= THRESHOLD_STOP + 1e-9:
        m = evaluate(preds, gt, proto, t)
        if m["recall"] >= proto.recall_min and m["fdr"] <= proto.fdr_max:
            key = (m["macro_recall"], m["recall"], -m["fdr"])
            if best is None or key > best[0]:
                best = (key, m)
                best_t = t
        t = round(t + THRESHOLD_STEP, 3)
    if best is None:
        # 无达标点: 取 recall 最高且 fdr 尽量低
        cands = []
        t = THRESHOLD_START
        while t <= THRESHOLD_STOP + 1e-9:
            m = evaluate(preds, gt, proto, t)
            cands.append((m["recall"], -m["fdr"], t, m))
            t = round(t + THRESHOLD_STEP, 3)
        cands.sort(reverse=True)
        best_t, best = cands[0][2], (None, cands[0][3])
    return best_t, best[1]


def evaluate(preds: dict[int, list[dict]], gt: dict[int, list[dict]],
             proto: Any, threshold: float) -> dict:
    """官方细类口径: 只保留 score>=threshold 的预测参与贪心匹配。"""
    used: dict[int, set[int]] = defaultdict(set)
    tp = fn = 0
    macro_tp: dict[str, int] = defaultdict(int)
    macro_fn: dict[str, int] = defaultdict(int)
    for img_id, gts in gt.items():
        plist = [p for p in preds.get(img_id, []) if p["score"] >= threshold]
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
                macro_tp[proto.category_mapping[cid]] += 1
            else:
                fn += 1
                macro_fn[proto.category_mapping[cid]] += 1
    fp = 0
    for img_id, plist in preds.items():
        for i, p in enumerate(plist):
            if p["score"] >= threshold and i not in used[img_id]:
                fp += 1
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fdr = fp / (tp + fp) if (tp + fp) else 0.0
    macros = []
    for coarse in ("aircraft", "ship", "vehicle"):
        cls = [c for c in range(25) if proto.category_mapping[c] == coarse]
        m_tp = sum(macro_tp[proto.category_mapping[c]] for c in cls)
        m_fn = sum(macro_fn[proto.category_mapping[c]] for c in cls)
        macros.append(m_tp / (m_tp + m_fn) if (m_tp + m_fn) else 0.0)
    return {"recall": recall, "fdr": fdr, "tp": tp, "fp": fp, "fn": fn,
            "macro_recall": sum(macros) / len(macros), "aircraft_recall": macros[0],
            "ship_recall": macros[1], "vehicle_recall": macros[2]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--logits-dir", required=True)
    ap.add_argument("--config", default="configs/experiments/r1_proposal_reranking_v1.yaml")
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--folds", default="0,1,2", help="加载 logits 的折(逗号分隔, 单折场景如 COPH 可传 0)")
    ap.add_argument("--variant", default=None,
                    help="跳过 cross-fit, 直接应用指定变体(如 R3_fuse_a0.40)——单折/诊断场景")
    args = ap.parse_args()

    folds = tuple(int(f) for f in args.folds.split(",") if f.strip())

    config = load_config(args.config)
    records = load_proposal_manifest(args.manifest)
    print(f"proposals: {len(records)}")
    logits = load_logits(args.logits_dir, records, folds=folds)
    probabilities = logits_to_probabilities(logits)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    variants = build_variants(config["search"])

    # fold 映射: 从 manifest
    image_folds = {r.image_id: r.fold for r in records}
    all_ids = sorted(image_folds)
    transformed: dict[str, dict[int, list[dict]]] = {}
    for v in variants:
        transformed[v.name], _ = apply_variant(records, probabilities, v, image_ids=all_ids)

    # GT
    from rsdet.analysis.proposal_reranking import load_gt_from_formal_crop_manifest
    gt = load_gt_from_formal_crop_manifest(args.formal_crop_manifest,
                                           expected_images=4481, expected_annotations=20933)

    # 指定变体: 跳过 cross-fit, 直接应用(单折/诊断场景)
    if args.variant:
        if args.variant not in transformed:
            raise ValueError(f"未知变体 {args.variant}, 可选: {sorted(transformed)}")
        merged_best = transformed[args.variant]
        gt_subset = {i: gt[i] for i in all_ids}  # 只评估 manifest 覆盖的图
        t_all, m_all = scan_threshold(gt_subset, merged_best, proto)
        print(f"\n=== 指定变体 {args.variant}: R={m_all['recall']:.4f} F={m_all['fdr']:.4f} "
              f"macroR={m_all['macro_recall']:.4f} aircraftR={m_all['aircraft_recall']:.4f} ===")
        per_fold: list[dict] = []

    # 三折 cross-fit: 两折选变体+阈值, 评估第三折
    else:
        per_fold = []
        merged_best = {}
        for held in (0, 1, 2):
            sel_ids = [i for i in all_ids if image_folds[i] != held]
            val_ids = [i for i in all_ids if image_folds[i] == held]
            sel_gt = {i: gt[i] for i in sel_ids}
            candidates = []
            for v in variants:
                t, m = scan_threshold(sel_gt, {i: transformed[v.name][i] for i in sel_ids}, proto)
                candidates.append({"variant": v.name, "threshold": t, "selection": m})
            # 选: 达标且 recall drop <=0.005, 按 macro_recall 优先
            base = next(c for c in candidates if c["variant"] == "D0_detector")
            best = None
            for c in candidates:
                eligible = (c["selection"]["recall"] >= base["selection"]["recall"] - 0.005
                            and c["selection"]["fdr"] <= base["selection"]["fdr"] + 1e-12
                            and c["selection"]["recall"] >= proto.recall_min
                            and c["selection"]["fdr"] <= proto.fdr_max)
                if eligible:
                    key = c["selection"]["macro_recall"]
                    if best is None or key > best[0]:
                        best = (key, c)
            chosen = best[1] if best else base
            val_preds = {i: transformed[chosen["variant"]][i] for i in val_ids}
            t, m = scan_threshold({i: gt[i] for i in val_ids}, val_preds, proto)
            per_fold.append({"held_out": held, "variant": chosen["variant"],
                             "threshold": chosen["threshold"], "val_metric": m})
            # 记录 chosen 变体在 held-out 上的预测(用于 merged)
            for i in val_ids:
                merged_best[i] = transformed[chosen["variant"]][i]
            print(f"fold{held}: variant={chosen['variant']} t={chosen['threshold']:.3f} "
                  f"R={m['recall']:.4f} F={m['fdr']:.4f} macroR={m['macro_recall']:.4f} "
                  f"airR={m['aircraft_recall']:.4f}")

        # 合并三折(各折用其 chosen 变体)
        t_all, m_all = scan_threshold(gt, merged_best, proto)
        print(f"\n=== E1 合并(各折选变体): R={m_all['recall']:.4f} F={m_all['fdr']:.4f} "
              f"macroR={m_all['macro_recall']:.4f} aircraftR={m_all['aircraft_recall']:.4f} ===")

    # D0 基线(全量, 无重排)
    gt_baseline = gt_subset if args.variant else gt
    _, m_d0 = scan_threshold(gt_baseline, transformed["D0_detector"], proto)
    print(f"D0 基线: R={m_d0['recall']:.4f} F={m_d0['fdr']:.4f} "
          f"macroR={m_d0['macro_recall']:.4f} aircraftR={m_d0['aircraft_recall']:.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "E1-Y5-R1-REPLAY",
        "n_proposals": len(records),
        "per_fold": per_fold,
        "merged_e1": m_all, "d0_baseline": m_d0,
        "delta": {k: round(m_all[k] - m_d0[k], 4) for k in
                  ("recall", "fdr", "macro_recall", "aircraft_recall")},
        "note": "未套用 M1 的 C2 分数校准(Y5 分数域需单独重拟合); "
                "R1 变体仅 aircraft/ship 同粗类重标, vehicle 旁路(目标粗类默认)或全粗类",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {out}")

    # 输出各折 chosen 变体的合并预测(供 E2 post-rerank NMS 消费)
    pred_out = out.parent / "e1_reranked_predictions.json"
    predictions = []
    for image_id in sorted(merged_best):
        for record in merged_best[image_id]:
            predictions.append({
                "image_id": image_id,
                "category_id": int(record["category_id"]),
                "score": float(record["score"]),
                "bbox_xyxy": [float(v) for v in record["bbox_xyxy"]],
                "proposal_uid": record["proposal_uid"],
            })
    pred_out.write_text(json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"重排后预测(供 E2): {pred_out} ({len(predictions)} 条)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
