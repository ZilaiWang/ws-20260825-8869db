"""P0-4: 类别/域固定 FDR 前沿(方案5 §二 批次0)。

在整体工作点(FDR=0.12)下, 拆解每粗类/每细类/每 source_group 的 Recall 与 FDR,
定位拖后腿的类与域。纯本地(基于 65,301 候选 + 14 特征 OER)。
"""
import argparse
import csv
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

FEAT = [
    "y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
    "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density",
    "d4_support", "has_oto",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))

    names = {}
    with open(args.formal_crop_manifest, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            names[int(r["class_id"])] = r["class_name"]

    nodes = pd.read_csv(args.nodes)
    preds4 = json.load(open(args.preds_d4))
    nodes["d4_support"] = nodes["proposal_uid"].map(
        {p["proposal_uid"]: p["d4_support"] for p in preds4}).fillna(0).astype(int)

    oto_by_img = defaultdict(list)
    for fold in (0, 1, 2):
        for p in json.load(open(Path(args.oto_dir) / f"a5_oto_fold{fold}.json")):
            p["category_id"] = int(p["category_id"])
            p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
            p["image_id"] = int(p["image_id"])
            oto_by_img[p["image_id"]].append(p)

    def has_oto(r):
        bb = [r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2]
        for q in oto_by_img.get(int(r.image_id), []):
            if q["category_id"] != int(r.category_id) or q["score"] < 0.5:
                continue
            if compute_iou(bb, q["bbox_xyxy"]) > 0.5:
                return 1
        return 0

    nodes["has_oto"] = nodes.apply(has_oto, axis=1)
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}
    grp_of = {i: o.group_id for (i, _), o in formal.objects.items()}
    nodes["fold"] = nodes["image_id"].map(fold_of)
    nodes["group_id"] = nodes["image_id"].map(grp_of)

    # OER cross-fit
    X = nodes[FEAT].to_numpy(dtype=float)
    y = nodes["is_valid"].to_numpy(dtype=float)
    folds = nodes["fold"].to_numpy()
    probs = np.zeros(len(nodes))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                             l2_regularization=1.0, random_state=2026 + held)
        clf.fit(X[tr], y[tr])
        probs[va] = clf.predict_proba(X[va])[:, 1]

    # 候选 + oracle 改类
    cand = []
    for r in nodes.itertuples():
        cand.append({"_idx": int(r.idx), "image_id": int(r.image_id),
                     "category_id": int(r.category_id), "crop_top1_class": int(r.crop_top1_class),
                     "coarse_id": int(r.coarse_id), "group_id": r.group_id,
                     "bbox_xyxy": [r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2]})
    pb = defaultdict(list)
    for p in cand:
        pb[p["image_id"]].append(p)
    gt_fine = defaultdict(set)
    for img, gts in formal.boxes.items():
        for p in pb.get(img, []):
            for g in gts:
                thr = proto.iou_thresholds[proto.category_mapping[int(g["category_id"])]]
                if compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr:
                    gt_fine[p["_idx"]].add(int(g["category_id"]))
    for p in cand:
        gf = gt_fine.get(p["_idx"], set())
        if gf and p["category_id"] not in gf:
            p["category_id"] = p["crop_top1_class"]
    for i, p in enumerate(cand):
        p["oer"] = float(probs[i])

    # NMS
    pb2 = defaultdict(list)
    for p in cand:
        pb2[p["image_id"]].append(p)
    kept = []
    for img, pl in pb2.items():
        od = sorted(pl, key=lambda p: -p["oer"])
        sup = set()
        for p in od:
            if p["_idx"] in sup:
                continue
            kept.append(p)
            for q in od:
                if q["_idx"] == p["_idx"] or q["_idx"] in sup or q["category_id"] != p["category_id"]:
                    continue
                if compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > 0.5:
                    sup.add(q["_idx"])

    # 贪心匹配(全候选 floor)
    def greedy(preds):
        pb3 = defaultdict(list)
        for p in preds:
            pb3[p["image_id"]].append(p)
        tp = {}
        used = defaultdict(set)
        for img, gts in formal.boxes.items():
            pl = pb3.get(img, [])
            od = sorted(pl, key=lambda p: -p["oer"])
            for g in gts:
                cid = int(g["category_id"])
                thr = proto.iou_thresholds[proto.category_mapping[cid]]
                bi, bix = 0.0, None
                for p in od:
                    if p["_idx"] in used[img] or p["category_id"] != cid:
                        continue
                    iou = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                    if iou > bi:
                        bi, bix = iou, p
                if bix is not None and bi >= thr:
                    used[img].add(bix["_idx"])
                    tp[bix["_idx"]] = True
        return tp

    tp_floor = greedy(kept)
    n_gt = sum(len(gts) for gts in formal.boxes.values())

    # 工作点: 整体 FDR<=0.12 的最大 Recall
    od2 = sorted(kept, key=lambda p: -p["oer"])
    tp_ = fp = 0
    work_thr = None
    best = (0.0, 0, 0)
    for p in od2:
        if tp_floor.get(p["_idx"]):
            tp_ += 1
        else:
            fp += 1
        fdr = fp / (tp_ + fp) if tp_ + fp else 0
        if fdr <= 0.12 and tp_ / n_gt > best[0]:
            best = (tp_ / n_gt, tp_, fp)
            work_thr = p["oer"]
    print(f"整体: R@FDR=.12 = {best[0]:.4f} (TP={best[1]}, FP={best[2]}, 阈值={work_thr:.4f})")

    # 工作点候选(score >= work_thr)
    work_preds = [p for p in kept if p["oer"] >= work_thr]

    # 每粗类 Recall(工作点口径, 用全候选 floor 做上限对照)
    def class_recall(preds_sub):
        pb3 = defaultdict(list)
        for p in preds_sub:
            pb3[p["image_id"]].append(p)
        used = defaultdict(set)
        class_tp = Counter()
        for img, gts in formal.boxes.items():
            pl = pb3.get(img, [])
            od = sorted(pl, key=lambda p: -p["oer"])
            for g in gts:
                cid = int(g["category_id"])
                thr = proto.iou_thresholds[proto.category_mapping[cid]]
                bi, bix = 0.0, None
                for p in od:
                    if p["_idx"] in used[img] or p["category_id"] != cid:
                        continue
                    iou = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                    if iou > bi:
                        bi, bix = iou, p
                if bix is not None and bi >= thr:
                    used[img].add(bix["_idx"])
                    class_tp[cid] += 1
        return class_tp

    work_tp = class_recall(work_preds)
    floor_tp = class_recall(kept)
    gt_count = Counter()
    for gts in formal.boxes.values():
        for g in gts:
            gt_count[int(g["category_id"])] += 1

    # 每粗类
    print(f"\n=== 每粗类 Recall(工作点 vs floor) ===")
    for coarse in ("aircraft", "ship", "vehicle"):
        cls = [c for c in range(25) if proto.category_mapping[c] == coarse]
        tg = sum(gt_count[c] for c in cls)
        tw = sum(work_tp[c] for c in cls)
        tf = sum(floor_tp[c] for c in cls)
        print(f"  {coarse:10s}: GT={tg:5d} 工作点={tw/tg:.4f} floor={tf/tg:.4f} 排序损失={tf-tw}")

    # 每细类(工作点 Recall, 排序)
    print(f"\n=== 每细类 Recall(工作点口径, 升序 = 最弱类在前) ===")
    rows = []
    for c in range(25):
        tg = gt_count[c]
        if tg == 0:
            continue
        rows.append((work_tp[c] / tg, names[c], tg, work_tp[c], floor_tp[c], proto.category_mapping[c]))
    rows.sort()
    for r, nm, tg, tw, tf, coarse in rows:
        flag = " <== 弱类" if r < 0.90 else ""
        print(f"  {nm:10s}({coarse:8s}) GT={tg:5d} Recall={r:.4f} floor={tf/tg:.4f}{flag}")

    # 每域(group_id)
    print(f"\n=== 每 source_group Recall/FDR(工作点口径) ===")
    # 每域的工作点候选 + GT
    grp_preds = defaultdict(list)
    for p in work_preds:
        grp_preds[p["group_id"]].append(p)
    grp_gt = defaultdict(int)
    for gts in formal.boxes.values():
        for g in gts:
            img = int(g.get("image_id") or 0)
            pass
    # 用 objects 拿每 image 的 group
    img2grp = {i: o.group_id for (i, _), o in formal.objects.items()}
    grp_gt = defaultdict(int)
    for img, gts in formal.boxes.items():
        g = img2grp.get(img, "?")
        grp_gt[g] += len(gts)
    grp_rows = []
    for grp, n_gt_g in grp_gt.items():
        pl = grp_preds.get(grp, [])
        # 该域工作点 TP/FP(在整体阈值下)
        # 用全候选 floor 匹配限定该域
        pb3 = defaultdict(list)
        for p in pl:
            pb3[p["image_id"]].append(p)
        used = defaultdict(set)
        tp_g = 0
        for img, gts in formal.boxes.items():
            if img2grp.get(img) != grp:
                continue
            plist = pb3.get(img, [])
            od = sorted(plist, key=lambda p: -p["oer"])
            for gt in gts:
                cid = int(gt["category_id"])
                thr = proto.iou_thresholds[proto.category_mapping[cid]]
                bi, bix = 0.0, None
                for p in od:
                    if p["_idx"] in used[img] or p["category_id"] != cid:
                        continue
                    iou = compute_iou(p["bbox_xyxy"], gt["bbox_xyxy"])
                    if iou > bi:
                        bi, bix = iou, p
                if bix is not None and bi >= thr:
                    used[img].add(bix["_idx"])
                    tp_g += 1
        n_pred_g = len(pl)
        fdr_g = (n_pred_g - tp_g) / n_pred_g if n_pred_g else 0
        grp_rows.append((tp_g / n_gt_g if n_gt_g else 0, grp, n_gt_g, tp_g, n_pred_g, fdr_g))
    grp_rows.sort()
    # 最差 15 域
    print("  最差 15 域(Recall 升序):")
    for r, grp, ng, tp, np_, fdr in grp_rows[:15]:
        print(f"    {grp}: GT={ng} Recall={r:.3f} FDR={fdr:.3f} (pred={np_})")
    # 最好 5 域
    print("  最好 5 域:")
    for r, grp, ng, tp, np_, fdr in grp_rows[-5:]:
        print(f"    {grp}: GT={ng} Recall={r:.3f} FDR={fdr:.3f}")
    recs = [r for r, *_ in grp_rows]
    print(f"  域 Recall P10={np.percentile(recs, 10):.4f} 中位={np.percentile(recs, 50):.4f} P90={np.percentile(recs, 90):.4f}")

    # 输出 JSON
    out = {
        "overall": {"r_fdr_0.12": best[0], "tp": best[1], "fp": best[2], "threshold": work_thr},
        "coarse": {c: {"gt": sum(gt_count[k] for k in range(25) if proto.category_mapping[k] == c),
                        "workpoint_tp": sum(work_tp[k] for k in range(25) if proto.category_mapping[k] == c),
                        "floor_tp": sum(floor_tp[k] for k in range(25) if proto.category_mapping[k] == c)}
                   for c in ("aircraft", "ship", "vehicle")},
        "fine_class": {str(c): {"name": names[c], "gt": gt_count[c], "workpoint_tp": work_tp[c],
                                "floor_tp": floor_tp[c], "coarse": proto.category_mapping[c]}
                       for c in range(25) if gt_count[c] > 0},
        "domain_p10_p50_p90": [float(np.percentile(recs, 10)), float(np.percentile(recs, 50)),
                               float(np.percentile(recs, 90))],
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {args.output}")


if __name__ == "__main__":
    main()
