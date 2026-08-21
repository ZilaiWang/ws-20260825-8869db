"""D3: group-balanced 采样(方案5 §十二 批次4, 基于 D1 域聚类)。

D1 发现: 12 高误检域贡献 30% FP_BG, 20 低召回域 Recall 仅 0.382。
本脚本复用 D1 的 OER 工作点 + 每 group 错误统计, 选出 worst-group
(FP_BG 最多的 top-K + Recall 最低的 top-K, 去重), 映射到 image_id,
生成 E7 风格的 hard-curriculum JSON(这些图训练重复 1 次 = 2x 采样),
供 train_cv3_oof.py --hard-curriculum 使用。

纯本地(基于 nodes + d4 + oto + formal manifest)。
"""
import argparse
import json
import sys
from collections import defaultdict
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
    ap.add_argument("--top-k", type=int, default=20, help="每类 worst-group 选多少个")
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
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
    work_preds = [p for p in kept if p["oer"] >= work_thr]

    # 每 group 统计
    grp_gt = defaultdict(int)
    for img, gts in formal.boxes.items():
        grp_gt[grp_of.get(img, "?")] += len(gts)
    idx2p = {p["_idx"]: p for p in kept}
    grp_tp = defaultdict(int)
    grp_fpbg = defaultdict(int)
    grp_fpcls = defaultdict(int)
    for p in work_preds:
        if tp_floor.get(p["_idx"]):
            grp_tp[p["group_id"]] += 1
            continue
        gf = gt_fine.get(p["_idx"], set())
        if p["category_id"] in gf:
            grp_fpcls[p["group_id"]] += 1
        else:
            grp_fpbg[p["group_id"]] += 1

    rows = []
    for grp, ng in grp_gt.items():
        tp_g = grp_tp.get(grp, 0)
        fbg = grp_fpbg.get(grp, 0)
        rec = tp_g / ng if ng else 0
        rows.append({"group_id": grp, "gt": ng, "tp": tp_g, "fp_bg": fbg,
                     "fp_cls": grp_fpcls.get(grp, 0), "recall": rec})
    df = pd.DataFrame(rows)

    # worst-group 选择: FP_BG 最多 top-K + Recall 最低 top-K(gt>=5 过滤小样本噪声)
    df_valid = df[df["gt"] >= 5]
    top_fbg = set(df_valid.nlargest(args.top_k, "fp_bg")["group_id"])
    low_rec = set(df_valid.nsmallest(args.top_k, "recall")["group_id"])
    worst_groups = sorted(top_fbg | low_rec)

    # group → image 映射
    grp_images = defaultdict(list)
    for (img, _), o in formal.objects.items():
        grp_images[o.group_id].append(img)

    hard_images = []
    for g in worst_groups:
        hard_images.extend(grp_images.get(g, []))
    hard_images = sorted(set(hard_images))

    print(f"worst-groups: {len(worst_groups)} 个(FP_BG top{args.top_k} ∪ Recall bottom{args.top_k})")
    print(f"  对应 hard images: {len(hard_images)} 张")
    print(f"  worst-groups: {worst_groups}")

    # 主输出: 纯 hard-curriculum 数组(供 --hard-curriculum 直接读取)
    hard_curriculum = [{"image_id": int(i)} for i in hard_images]
    Path(args.output).write_text(json.dumps(hard_curriculum, ensure_ascii=False) + "\n", encoding="utf-8")
    # 附加信息(诊断用)
    meta = {
        "worst_groups": worst_groups,
        "n_worst_groups": len(worst_groups),
        "n_hard_images": len(hard_images),
        "top_k": args.top_k,
    }
    Path(str(args.output).replace(".json", ".meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写: {args.output} (+ .meta.json)")


if __name__ == "__main__":
    main()
