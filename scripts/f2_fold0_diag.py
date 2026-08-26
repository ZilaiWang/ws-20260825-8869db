#!/usr/bin/env python3
"""F2 fold0 快速诊断: 对比 Y5 vs F2 的 candidate-floor + NO_CAND + 粗类 Recall。

快速判断 F2(family 三层辅助损失)是否改变了候选分布——尤其 FP_CLS(兄弟混淆)。
Y5 基线候选从 nodes.csv(fold=0) 读; F2 候选从推理 preds.json 读。

本地运行(无需 GPU)。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def candidate_floor_and_nocand(preds, formal, proto, fold_img):
    pb = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    used = defaultdict(set)
    tp = 0
    no_cand = defaultdict(int)
    total = defaultdict(int)
    for img, gts in formal.boxes.items():
        if img not in fold_img:
            continue
        pl = pb.get(img, [])
        ordered = sorted(enumerate(pl), key=lambda t: -t[1]["score"])
        for g in gts:
            cid = int(g["category_id"])
            coarse = proto.category_mapping[cid]
            thr = proto.iou_thresholds[coarse]
            total[coarse] += 1
            matched = any(compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr for p in pl)
            if not matched:
                no_cand[coarse] += 1
            bi, bix = 0.0, None
            for idx, p in ordered:
                if idx in used[img] or p["category_id"] != cid:
                    continue
                iou = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if iou > bi:
                    bi, bix = iou, idx
            if bix is not None and bi >= thr:
                used[img].add(bix)
                tp += 1
    return tp, dict(no_cand), dict(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="Y5 nodes.csv")
    ap.add_argument("--f2-preds", required=True, help="F2 推理 preds.json")
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    fold_img = {i for (i, _), o in formal.objects.items() if o.fold == args.fold}
    n_gt = sum(1 for img, gts in formal.boxes.items() if img in fold_img for _ in gts)

    # Y5 候选(从 nodes.csv fold=0)
    nodes = pd.read_csv(args.nodes)
    y5 = nodes[nodes["fold"] == args.fold] if "fold" in nodes.columns else nodes
    y5_preds = []
    for r in y5.itertuples():
        y5_preds.append({
            "image_id": int(r.image_id), "category_id": int(r.category_id),
            "score": float(r.y5_score),
            "bbox_xyxy": [r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2],
        })

    # F2 候选
    f2_raw = json.load(open(args.f2_preds))
    f2_preds = [{"image_id": int(p["image_id"]), "category_id": int(p["category_id"]),
                 "score": float(p["score"]),
                 "bbox_xyxy": [float(v) for v in p["bbox_xyxy"]]} for p in f2_raw]

    tp_y5, nc_y5, tot_y5 = candidate_floor_and_nocand(y5_preds, formal, proto, fold_img)
    tp_f2, nc_f2, tot_f2 = candidate_floor_and_nocand(f2_preds, formal, proto, fold_img)

    print(f"fold{args.fold} GT 数: {n_gt}\n")
    print(f"{'模型':6s} {'候选':>8s} {'cand-floor':>11s} {'aircraft_NO_CAND':>16s} "
          f"{'ship_NO_CAND':>13s} {'vehicle_NO_CAND':>15s}")
    for name, preds, tp, nc, tot in [("Y5", y5_preds, tp_y5, nc_y5, tot_y5),
                                     ("F2", f2_preds, tp_f2, nc_f2, tot_f2)]:
        floor = tp / n_gt
        a = f"{nc.get('aircraft',0)}/{tot.get('aircraft',0)}"
        s = f"{nc.get('ship',0)}/{tot.get('ship',0)}"
        v = f"{nc.get('vehicle',0)}/{tot.get('vehicle',0)}"
        print(f"{name:6s} {len(preds):8d} {floor:11.4f} {a:>16s} {s:>13s} {v:>15s}")

    print(f"\ncand-floor Δ(F2-Y5) = {tp_f2/n_gt - tp_y5/n_gt:+.4f}")

    Path(args.output).write_text(json.dumps({
        "fold": args.fold, "n_gt": n_gt,
        "y5": {"n": len(y5_preds), "floor": tp_y5 / n_gt, "no_cand": nc_y5},
        "f2": {"n": len(f2_preds), "floor": tp_f2 / n_gt, "no_cand": nc_f2},
        "floor_delta": tp_f2 / n_gt - tp_y5 / n_gt,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output}")


if __name__ == "__main__":
    main()
