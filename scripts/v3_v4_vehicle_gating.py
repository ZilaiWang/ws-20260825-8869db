"""V3+V4: 车辆受控恢复(方案5 §九 批次3)——支持面 gating + density top-K。

V1 失败根因: vehicle-only 密集监督让候选 +79% 爆炸, 新增 FP 拖累前沿。
V3(支持面 gating)拒绝非"道路/停车区"环境的结构化背景车辆候选;
V4(density top-K)每图限制车辆候选预算, 防候选爆炸。

本脚本做 OER 后处理级验证(不重训 detector): 在现有 Y5 候选上, 对 vehicle 候选
用 region 特征聚类环境原型, 非支持面环境的 vehicle 候选降权/剔除, 再加每图 top-K,
看 Recall@FDR 是否改善(尤其 vehicle 粗类)。

纯本地(基于 nodes + fpn region 特征 + 14特征 OER)。
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

C3, C4, C5 = 128, 256, 512
LAYOUT = {
    "core_p3": (0, C3), "ring_p3": (C3, 2 * C3), "region_p3": (2 * C3, 3 * C3),
    "core_p4": (3 * C3, 3 * C3 + C4), "ring_p4": (3 * C3 + C4, 3 * C3 + 2 * C4),
    "region_p4": (3 * C3 + 2 * C4, 3 * C3 + 3 * C4),
    "core_p5": (3 * C3 + 3 * C4, 3 * C3 + 3 * C4 + C5),
    "ring_p5": (3 * C3 + 3 * C4 + C5, 3 * C3 + 3 * C4 + 2 * C5),
    "region_p5": (3 * C3 + 3 * C4 + 2 * C5, 3 * C3 + 3 * C4 + 3 * C5),
    "scene": (3 * C3 + 3 * C4 + 3 * C5, 3 * C3 + 3 * C4 + 3 * C5 + C5),
}
VEHICLE_CLS = 24

FEAT_BASE = [
    "y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
    "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density",
    "d4_support", "has_oto",
]


def slice_region(feat):
    parts = [feat[:, LAYOUT[b][0]:LAYOUT[b][1]]
             for b in ("region_p3", "region_p4", "region_p5")]
    return np.concatenate(parts, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--fpn-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--k-protos", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=12, help="每图 vehicle 候选预算")
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
    nodes["fold"] = nodes["image_id"].map(fold_of)

    uid2feat = {}
    for f in (0, 1, 2):
        z = np.load(Path(args.fpn_dir) / f"y5_fpn_feat_fold{f}.npz", allow_pickle=True)
        for uid, feat in zip(z["uids"], z["feats"]):
            uid2feat[str(uid)] = feat
    feat_mat = np.stack([uid2feat[str(u)] for u in nodes["proposal_uid"]],
                        axis=0).astype(np.float32)
    region = slice_region(feat_mat)

    folds = nodes["fold"].to_numpy()
    is_tp = nodes["is_valid"].to_numpy() == 1
    is_vehicle = (nodes["crop_top1_class"].to_numpy() == VEHICLE_CLS) | \
                 (nodes["category_id"].to_numpy() == VEHICLE_CLS)

    # V3: 支持面 gating —— 环境原型聚类, vehicle TP 密度高 = 车辆支持面
    proto_id = np.zeros(len(nodes), dtype=int)
    support_score = np.zeros(len(nodes))
    for held in (0, 1, 2):
        tr = folds != held
        va = folds == held
        sc = StandardScaler().fit(region[tr])
        pca = PCA(n_components=32, random_state=0).fit(sc.transform(region[tr]))
        km = KMeans(n_clusters=args.k_protos, random_state=42, n_init=3)
        km.fit(pca.transform(sc.transform(region[tr])))
        proto_id[va] = km.predict(pca.transform(sc.transform(region[va])))
        # 训练折每原型的 vehicle TP 密度(车辆支持面)
        for k in range(args.k_protos):
            mask = (proto_id[tr] == k)
            n_proto = int(mask.sum())
            if n_proto < 5:
                continue
            veh_tp = int((mask & is_tp[tr] & (nodes["category_id"].to_numpy()[tr] == VEHICLE_CLS)).sum())
            veh_all = int((mask & (nodes["category_id"].to_numpy()[tr] == VEHICLE_CLS)).sum())
            rate = veh_tp / veh_all if veh_all else 0.0
            va_mask = va & (proto_id == k)
            support_score[va_mask] = rate
    nodes["support_score"] = support_score

    print(f"=== V3 支持面 gating 诊断 ===")
    veh_tp_mask = is_tp & (nodes["category_id"].to_numpy() == VEHICLE_CLS)
    veh_fpbg = (nodes["is_valid"].to_numpy() == 0) & (nodes["fine_correct"].to_numpy() == 0) & \
               (nodes["crop_top1_class"].to_numpy() == VEHICLE_CLS)
    print(f"  vehicle TP 支持面分: {support_score[veh_tp_mask].mean():.3f}")
    print(f"  vehicle FP_BG 支持面分: {support_score[veh_fpbg].mean():.3f}")

    # OER 训练
    def train_oer(feats):
        X = nodes[feats].to_numpy(dtype=float)
        y = nodes["is_valid"].to_numpy(dtype=float)
        probs = np.zeros(len(nodes))
        for held in (0, 1, 2):
            tr = np.where(folds != held)[0]
            va = np.where(folds == held)[0]
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                                 l2_regularization=1.0, random_state=2026 + held)
            clf.fit(X[tr], y[tr])
            probs[va] = clf.predict_proba(X[va])[:, 1]
        return probs

    probs_base = train_oer(FEAT_BASE)
    probs_gate = train_oer(FEAT_BASE + ["support_score"])

    def evaluate(probs, apply_gate=False, apply_topk=False):
        cand = []
        for i, r in enumerate(nodes.itertuples()):
            oer = float(probs[i])
            if apply_gate and is_vehicle[i] and support_score[i] < 0.05:
                oer *= 0.5  # 非支持面车辆候选降权
            cand.append({"_idx": int(r.idx), "image_id": int(r.image_id),
                         "category_id": int(r.category_id), "crop_top1_class": int(r.crop_top1_class),
                         "is_veh": bool(is_vehicle[i]),
                         "bbox_xyxy": [r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2],
                         "oer": oer})
        # V4: 每图 vehicle top-K(先 NMS 前, 简单按 oer 截断)
        if apply_topk:
            pb_v = defaultdict(list)
            for p in cand:
                if p["is_veh"]:
                    pb_v[p["image_id"]].append(p)
            for img, pl in pb_v.items():
                od = sorted(pl, key=lambda p: -p["oer"])
                drop = {p["_idx"] for p in od[args.top_k:]}
                for p in cand:
                    if p["_idx"] in drop:
                        p["oer"] *= 0.01
        # oracle 改类 + NMS
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
        tp = {}
        used = defaultdict(set)
        pb3 = defaultdict(list)
        for p in kept:
            pb3[p["image_id"]].append(p)
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
        od2 = sorted(kept, key=lambda p: -p["oer"])
        tp_ = fp = 0
        br = {v: 0.0 for v in (0.12, 0.11, 0.10)}
        for p in od2:
            if tp.get(p["_idx"]):
                tp_ += 1
            else:
                fp += 1
            rec = tp_ / n_gt
            fdr = fp / (tp_ + fp) if tp_ + fp else 0
            for v in br:
                if fdr <= v:
                    br[v] = max(br[v], rec)
        return br

    br_base = evaluate(probs_base)
    br_gate = evaluate(probs_gate)
    br_gate_topk = evaluate(probs_gate, apply_gate=True, apply_topk=True)
    print(f"\n[OER 14特征]              R@FDR=.12={br_base[0.12]:.4f}")
    print(f"[OER 14+支持面分]         R@FDR=.12={br_gate[0.12]:.4f} (Δ{br_gate[0.12]-br_base[0.12]:+.4f})")
    print(f"[OER 14+支持面+gating+topK] R@FDR=.12={br_gate_topk[0.12]:.4f} (Δ{br_gate_topk[0.12]-br_base[0.12]:+.4f})")

    Path(args.output).write_text(json.dumps({
        "base": {"r_0.12": br_base[0.12], "r_0.11": br_base[0.11]},
        "gate": {"r_0.12": br_gate[0.12], "r_0.11": br_gate[0.11]},
        "gate_topk": {"r_0.12": br_gate_topk[0.12], "r_0.11": br_gate_topk[0.11]},
        "support_veh_tp_mean": float(support_score[veh_tp_mask].mean()),
        "support_veh_fpbg_mean": float(support_score[veh_fpbg].mean()),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {args.output}")


if __name__ == "__main__":
    main()
