"""V1 车辆种子诊断: vehicle-only DFD 是否受控地降低车辆无候选率。

对比 Y5 vs V1 的 fold0 vehicle NO_CAND / candidate-floor / 候选量(受控性)。
方案5 §九.5 门槛: 无候选率 17.9%→≤10%, 新增候选 ≤Y5 1.15x, 车辆 Recall +3pp。
"""
import argparse
import csv
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def load_preds(path, fold_img):
    out = []
    for p in json.load(open(path)):
        img = int(p["image_id"])
        if img not in fold_img:
            continue
        out.append({"image_id": img, "category_id": int(p["category_id"]),
                    "score": float(p["score"]),
                    "bbox_xyxy": [float(v) for v in p["bbox_xyxy"]]})
    return out


def candidate_floor(preds, formal, proto, fold_img):
    pb = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    used = defaultdict(set)
    tp = 0
    for img, gts in formal.boxes.items():
        if img not in fold_img:
            continue
        pl = pb.get(img, [])
        ordered = sorted(enumerate(pl), key=lambda t: -t[1]["score"])
        for g in gts:
            cid = int(g["category_id"])
            thr = proto.iou_thresholds[proto.category_mapping[cid]]
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
    return tp


def no_cand_by_coarse(preds, formal, proto, fold_img):
    pb = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    no_cand = defaultdict(int)
    total = defaultdict(int)
    for img, gts in formal.boxes.items():
        if img not in fold_img:
            continue
        pl = pb.get(img, [])
        for g in gts:
            cid = int(g["category_id"])
            coarse = proto.category_mapping[cid]
            thr = proto.iou_thresholds[coarse]
            total[coarse] += 1
            if not any(compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr for p in pl):
                no_cand[coarse] += 1
    return dict(no_cand), dict(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--preds", action="append", required=True,
                    help="name:path 形式, 可多次")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    fold_img = {i for (i, _), o in formal.objects.items() if o.fold == args.fold}
    n_gt = sum(1 for img, gts in formal.boxes.items() if img in fold_img for _ in gts)

    print(f"fold{args.fold} GT 数: {n_gt}\n")
    print(f"{'模型':12s} {'候选':>8s} {'cand-floor':>11s} {'aircraft_NC':>12s} {'ship_NC':>9s} {'vehicle_NC':>12s}")
    rows = []
    for spec in args.preds:
        name, path = spec.split(":", 1)
        preds = load_preds(path, fold_img)
        floor = candidate_floor(preds, formal, proto, fold_img) / n_gt
        no_cand, total = no_cand_by_coarse(preds, formal, proto, fold_img)
        rows.append((name, len(preds), floor, no_cand, total))
        print(f"{name:12s} {len(preds):8d} {floor:11.4f} "
              f"{no_cand.get('aircraft',0)}/{total.get('aircraft',0):>10s} "
              f"{no_cand.get('ship',0)}/{total.get('ship',0):>7s} "
              f"{no_cand.get('vehicle',0)}/{total.get('vehicle',0):>10s}")

    # vehicle 专项(如果有多模型对比)
    if len(rows) >= 2:
        base_name, base_n, base_floor, base_nc, base_tot = rows[0]
        cand_name, cand_n, cand_floor, cand_nc, cand_tot = rows[-1]
        v_b = base_nc.get("vehicle", 0)
        v_c = cand_nc.get("vehicle", 0)
        print(f"\n=== vehicle 专项对比({cand_name} vs {base_name}) ===")
        print(f"  vehicle NO_CAND: {v_b} → {v_c} ({'↓' if v_c <= v_b else '↑'} {abs(v_c-v_b)})")
        print(f"  候选量比: {cand_n/base_n:.2f}x (门槛 ≤1.15x)")
        print(f"  candidate-floor: {base_floor:.4f} → {cand_floor:.4f} (Δ {cand_floor-base_floor:+.4f})")

    Path(args.output).write_text(json.dumps({
        "fold": args.fold, "n_gt": n_gt,
        "models": [{"name": r[0], "n_pred": r[1], "floor": r[2],
                    "no_cand": dict(r[3]), "total": dict(r[4])} for r in rows],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {args.output}")


if __name__ == "__main__":
    main()
