"""批量 frontier 分析: 对已完成实验的每个 fold 算 Recall@FDR=0.12, 对比 Y5 基线。

纯 score 排序(无改类无 OER), 用于 L2 漏斗的双折同向判定。
用法:
  python scripts/frontier_batch.py \
    --formal-manifest /workspace/N1A/formal_crop_manifest.csv \
    --project-config configs/project.yaml \
    --y5-dir /workspace/results/Y5-ROT90-CV3-OOF \
    --exps "F2-FAMILY:0,1,2 F3-CONFPAIR:0,1,2 V2-VEHICLESEED:0,1 D4-WORSTGROUPLOSS:0,1"
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def load_preds(path: str, fmt: str = "xyxy") -> list[dict]:
    d = json.load(open(path))
    out = []
    for p in d:
        if fmt == "xywh":
            x, y, w, h = p["bbox"]
            b = [x, y, x + w, y + h]
        else:
            b = p["bbox_xyxy"]
        out.append({
            "image_id": int(p["image_id"]),
            "category_id": int(p["category_id"]),
            "score": float(p["score"]),
            "bbox_xyxy": b,
        })
    return out


def frontier(preds, formal, proto, fold):
    fold_img = {i for (i, _), o in formal.objects.items() if o.fold == fold}
    n_gt = sum(1 for img, gts in formal.boxes.items() if img in fold_img for _ in gts)
    pb = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    kept = []
    for img, pl in pb.items():
        od = sorted(pl, key=lambda p: -p["score"])
        sup = set()
        for p in od:
            if id(p) in sup:
                continue
            kept.append(p)
            for q in od:
                if q is p or id(q) in sup or q["category_id"] != p["category_id"]:
                    continue
                if compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > 0.5:
                    sup.add(id(q))
    pb2 = defaultdict(list)
    for p in kept:
        pb2[p["image_id"]].append(p)
    used = defaultdict(set)
    ts = set()
    for img, gts in formal.boxes.items():
        if img not in fold_img:
            continue
        od = sorted(pb2.get(img, []), key=lambda p: -p["score"])
        for g in gts:
            cid = int(g["category_id"])
            thr = proto.iou_thresholds[proto.category_mapping[cid]]
            bi, bix = 0.0, None
            for p in od:
                if id(p) in used[img] or p["category_id"] != cid:
                    continue
                iou = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if iou > bi:
                    bi, bix = iou, p
            if bix is not None and bi >= thr:
                used[img].add(id(bix))
                ts.add(id(bix))
    od2 = sorted(kept, key=lambda p: -p["score"])
    tp_ = fp = 0
    br = {0.12: 0.0, 0.11: 0.0, 0.10: 0.0}
    for p in od2:
        if id(p) in ts:
            tp_ += 1
        else:
            fp += 1
        rec = tp_ / n_gt
        fdr = fp / (tp_ + fp) if tp_ + fp else 0
        for v in br:
            if fdr <= v:
                br[v] = max(br[v], rec)
    return br, n_gt, len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--y5-dir", required=True)
    ap.add_argument("--exps", required=True, help="空格分隔 'EXP:fold1,fold2'")
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))

    # Y5 基线
    print("=== Y5 基线(纯score frontier@FDR0.12) ===")
    y5_base = {}
    for f in (0, 1, 2):
        preds = load_preds(f"{args.y5_dir}/fold_{f}/predictions_low.json", fmt="xywh")
        br, n_gt, n_kept = frontier(preds, formal, proto, f)
        y5_base[f] = br[0.12]
        print(f"  fold{f}: frontier={br[0.12]:.4f} (GT={n_gt}, 候选={n_kept})")
    print()

    # 各实验
    print("=== 各实验 frontier@FDR0.12 vs Y5 ===")
    print(f"{'实验':22s} {'fold':4s} {'frontier':>9s} {'候选数':>8s} {'ΔvsY5':>8s}")
    for spec in args.exps.split():
        exp, folds = spec.split(":")
        for f in map(int, folds.split(",")):
            preds_path = f"/workspace/results/{exp}-FOLD{f}-40EP/preds.json"
            try:
                preds = load_preds(preds_path)
            except FileNotFoundError:
                print(f"{exp:22s} {f:<4d} {'(preds缺失)':>9s}")
                continue
            br, n_gt, n_kept = frontier(preds, formal, proto, f)
            delta = br[0.12] - y5_base[f]
            print(f"{exp:22s} {f:<4d} {br[0.12]:>9.4f} {n_kept:>8d} {delta:>+8.4f}")


if __name__ == "__main__":
    main()
