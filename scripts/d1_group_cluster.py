"""D1: source-group 错误聚类(方案5 §十二 批次4)。

核心: 255 个 source_group 平均三折结果会隐藏少数高误检机场(跑道结构导致
MS/FSC 幻觉、尺度导致车辆消失)。本脚本对每个 group_id 统计工作点口径的
Recall / FDR / FP_BG / FP_CLS, 并按错误画像聚类, 输出高误检域清单。

纯本地(基于 nodes + group_id + 14特征 OER)。
"""
import argparse
import csv
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

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
    n_gt = sum(len(gts) for gts in formal.boxes.values())

    nodes = pd.read_csv(args.nodes)
    preds4 = json.load(open(args.preds_d4))
    nodes["d4_support"] = nodes["proposal_uid"].map(
        {p["proposal_uid"]: p["d4_support"] for p in preds4}).fillna(0).astype(int)
    oto_by_img = defaultdict(list)
    for f in (0, 1, 2):
        for p in json.load(open(Path(args.oto_dir) / f"a5_oto_fold{f}.json")):
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

    # 候选 + oracle 改类 + NMS
    cand = []
    for r in nodes.itertuples():
        cand.append({"_idx": int(r.idx), "image_id": int(r.image_id),
                     "category_id": int(r.category_id), "crop_top1_class": int(r.crop_top1_class),
                     "group_id": r.group_id,
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

    # 工作点
    tp_floor = {}
    pb3 = defaultdict(list)
    for p in kept:
        pb3[p["image_id"]].append(p)
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
                tp_floor[bix["_idx"]] = True

    od2 = sorted(kept, key=lambda p: -p["oer"])
    tp_ = fp = 0
    work_thr = None
    best = 0.0
    for p in od2:
        if tp_floor.get(p["_idx"]):
            tp_ += 1
        else:
            fp += 1
        fdr = fp / (tp_ + fp) if tp_ + fp else 0
        if fdr <= 0.12 and tp_ / n_gt > best:
            best = tp_ / n_gt
            work_thr = p["oer"]
    print(f"整体工作点: R@FDR=.12={best:.4f} (阈值 {work_thr:.4f})")

    work_preds = [p for p in kept if p["oer"] >= work_thr]
    # 工作点候选的 idx
    work_idx = {p["_idx"] for p in work_preds}

    # 对每个 group 统计: GT / 工作点TP / FP_BG / FP_CLS
    img2grp = {i: o.group_id for (i, _), o in formal.objects.items()}
    grp_gt = defaultdict(int)
    for img, gts in formal.boxes.items():
        grp_gt[img2grp.get(img, "?")] += len(gts)

    # 工作点 TP(每 group)
    idx2p = {p["_idx"]: p for p in kept}
    grp_tp = defaultdict(int)
    for idx in work_idx:
        if tp_floor.get(idx):
            grp_tp[idx2p[idx]["group_id"]] += 1

    # FP 细分: 工作点里非 TP 的候选 → FP_BG(细类也错) vs FP_CLS(细类对但没匹配上GT)
    grp_fpbg = defaultdict(int)
    grp_fpcls = defaultdict(int)
    for p in work_preds:
        if tp_floor.get(p["_idx"]):
            continue
        # 判断 FP_BG vs FP_CLS: 位置匹配 GT 且细类对 = FP_CLS, 否则 FP_BG
        gf = gt_fine.get(p["_idx"], set())
        is_cls_correct = (p["category_id"] in gf)
        if is_cls_correct:
            grp_fpcls[p["group_id"]] += 1
        else:
            grp_fpbg[p["group_id"]] += 1

    # 每 group 汇总
    rows = []
    for grp, ng in grp_gt.items():
        tp = grp_tp.get(grp, 0)
        fbg = grp_fpbg.get(grp, 0)
        fcls = grp_fpcls.get(grp, 0)
        npred = tp + fbg + fcls
        rec = tp / ng if ng else 0
        fdr = (fbg + fcls) / npred if npred else 0
        rows.append({"group_id": grp, "gt": ng, "tp": tp, "fp_bg": fbg, "fp_cls": fcls,
                     "recall": rec, "fdr": fdr})

    df = pd.DataFrame(rows)
    df = df.sort_values("recall")

    # 高误检域(FP_BG 集中)
    print(f"\n=== 域级统计(共 {len(df)} 个 source_group) ===")
    print(f"  域 Recall: P10={np.percentile(df['recall'],10):.3f} 中位={np.percentile(df['recall'],50):.3f} "
          f"P90={np.percentile(df['recall'],90):.3f}")
    print(f"  域 FDR:    中位={np.percentile(df['fdr'],50):.3f} P90={np.percentile(df['fdr'],90):.3f}")
    total_fbg = df["fp_bg"].sum()
    print(f"  FP_BG 总量={total_fbg}, 集中在 top-20 域: {df.nlargest(20,'fp_bg')['fp_bg'].sum()/total_fbg:.1%}")

    # 最差域(低 Recall)
    print(f"\n=== 最差 15 域(Recall 升序) ===")
    for _, r in df.head(15).iterrows():
        print(f"  {r['group_id']:24s} GT={r['gt']:4d} Recall={r['recall']:.3f} "
              f"FDR={r['fdr']:.3f} FP_BG={r['fp_bg']:4d} FP_CLS={r['fp_cls']:3d}")

    # 高误检域(FP_BG 最多)
    print(f"\n=== FP_BG 最多 15 域(高误检背景域) ===")
    top_fbg = df.nlargest(15, "fp_bg")
    for _, r in top_fbg.iterrows():
        print(f"  {r['group_id']:24s} GT={r['gt']:4d} FP_BG={r['fp_bg']:4d} "
              f"Recall={r['recall']:.3f} FDR={r['fdr']:.3f}")

    # 域聚类(按 recall/fdr/fp_bg 画像, K=4)
    feats = StandardScaler().fit_transform(
        df[["recall", "fdr", "fp_bg", "fp_cls"]].to_numpy(dtype=float))
    km = KMeans(n_clusters=4, random_state=42, n_init=5).fit(feats)
    df["cluster"] = km.labels_
    print(f"\n=== 域聚类(4 簇, 按 recall/fdr/fp_bg 画像) ===")
    for k in range(4):
        sub = df[df["cluster"] == k]
        print(f"  簇{k}: {len(sub)} 域, Recall均值={sub['recall'].mean():.3f} "
              f"FDR均值={sub['fdr'].mean():.3f} FP_BG合计={sub['fp_bg'].sum()}")

    out = {
        "n_groups": int(len(df)),
        "domain_recall_p10_p50_p90": [float(np.percentile(df["recall"], 10)),
                                       float(np.percentile(df["recall"], 50)),
                                       float(np.percentile(df["recall"], 90))],
        "domain_fdr_p90": float(np.percentile(df["fdr"], 90)),
        "fp_bg_top20_share": float(df.nlargest(20, "fp_bg")["fp_bg"].sum() / total_fbg),
        "worst_recall_groups": df.head(15)[["group_id", "gt", "recall", "fdr", "fp_bg", "fp_cls"]].to_dict("records"),
        "top_fpbg_groups": top_fbg[["group_id", "gt", "fp_bg", "recall", "fdr"]].to_dict("records"),
        "cluster_summary": [
            {"cluster": int(k), "n": int(len(df[df["cluster"] == k])),
             "recall_mean": float(df[df["cluster"] == k]["recall"].mean()),
             "fdr_mean": float(df[df["cluster"] == k]["fdr"].mean()),
             "fp_bg_total": int(df[df["cluster"] == k]["fp_bg"].sum())}
            for k in range(4)],
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {args.output}")


if __name__ == "__main__":
    main()
