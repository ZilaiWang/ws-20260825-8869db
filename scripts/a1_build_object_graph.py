#!/usr/bin/env python3
"""A1: 构建 OOF 对象图数据(节点特征 + 边监督)。

把 Y5 低阈值候选归并为对象级监督, 供 OER 对象证据解析器训练:
- 节点 = 每个候选(Y5 score + crop logits + 几何 + 密度);
- 边 = 空间相邻候选对(IoU>0 或中心距离<阈值), 标签三分类:
    same_object(匹配同一 annotation_uid) / different_object(匹配不同 uid)
    / background_related(至少一个背景)。

用法:
  python scripts/a1_build_object_graph.py \
    --predictions /tmp/Y5-full.json \
    --logits-dir /tmp/E1-Y5-LOGITS \
    --formal-crop-manifest outputs/.../formal_crop_manifest.csv \
    --project-config configs/project.yaml \
    --output-dir /tmp/a1-object-graph
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oer_labels import build_official_proposal_labels
from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def load_logits(logits_dir: str) -> dict[str, np.ndarray]:
    result = {}
    for fold in (0, 1, 2):
        d = np.load(Path(logits_dir) / f"fold_{fold}_logits.npz", allow_pickle=False)
        for uid, lg in zip(d["proposal_uids"].tolist(), d["logits"]):
            result[str(uid)] = np.asarray(lg, dtype=np.float32)
    return result


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--logits-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--edge-center-dist", type=float, default=50.0,
                    help="连边条件: 中心距离 < 该阈值(像素), 或 IoU>0")
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    logits = load_logits(args.logits_dir)

    # 加载候选(生成 proposal_uid, 与 E1 manifest 规则一致: y5-f{fold}-i{img}-p{idx})
    # fold 映射来自 oof_images.csv
    fold_of = {}
    oof_images = Path("outputs/M1-CV3-OOF-return-no-checkpoints-extracted-20260725"
                      "/M1-CV3-OOF-aggregate/oof_images.csv")
    with open(oof_images, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            fold_of[int(r["image_id"])] = int(r["fold"])

    preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    for i, p in enumerate(preds):
        p["score"] = float(p["score"])
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        p["_idx"] = i
        img = int(p["image_id"])
        p["proposal_uid"] = p.get("proposal_uid") or f"y5-f{fold_of.get(img, 0)}-i{img}-p{i:06d}"

    # 建立 GT 对象索引: (img_id, gt_index) -> annotation_uid
    uid_of_gt: dict[int, list[str]] = defaultdict(list)
    for (img_id, gi), obj in formal.objects.items():
        while len(uid_of_gt[img_id]) <= gi:
            uid_of_gt[img_id].append("")
        uid_of_gt[img_id][gi] = obj.annotation_uid

    # 官方 prediction-first 一对一匹配。is_valid/matched_uid 只能来自这条
    # evaluator trace，禁止在数据构建脚本中另写 GT-first 近似。
    pb: dict[int, list[dict]] = defaultdict(list)
    for p in preds:
        pb[p["image_id"]].append(p)
    label_result = build_official_proposal_labels(
        gt_boxes=formal.boxes,
        predictions=(
            {
                "image_id": p["image_id"],
                "category_id": p["category_id"],
                "score": p["score"],
                "bbox_xyxy": p["bbox_xyxy"],
                "source_prediction_index": p["_idx"],
            }
            for p in preds
        ),
        category_mapping=proto.category_mapping,
        iou_thresholds=proto.iou_thresholds,
    )
    match_of: dict[int, tuple[int, str, float]] = {}
    for candidate_id, label in label_result.labels.items():
        if not label.is_valid or label.matched_gt_index is None:
            continue
        gt_index = label.matched_gt_index
        match_of[candidate_id] = (
            gt_index,
            uid_of_gt[label.image_id][gt_index],
            label.matched_iou,
        )

    # 位置关联(类无关, 多对多, 用于定义 same/different 边 + fine_correct):
    # 每个候选 -> 位置覆盖的 GT 集合(不看候选细类)
    assoc: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for img_id, gts in formal.boxes.items():
        plist = pb.get(img_id, [])
        for p in plist:
            for gi, g in enumerate(gts):
                thr = proto.iou_thresholds[proto.category_mapping[int(g["category_id"])]]
                if compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr:
                    assoc[p["_idx"]].add((img_id, gi))

    n_tp = sum(1 for v in match_of.values() if v is not None)
    n_assoc = sum(1 for v in assoc.values() if v)
    print(f"候选 {len(preds)}, 官方 TP {n_tp}, 位置关联 GT 的候选 {n_assoc}")

    # 节点特征
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    node_rows = []
    for p in preds:
        lg = logits.get(p.get("proposal_uid", ""))
        x0, y0, x1, y1 = p["bbox_xyxy"]
        w, h = x1 - x0, y1 - y0
        area = max(w * h, 1e-6)
        se = min(w, h)
        aspect = max(w / h, h / w) if h > 0 else 1.0
        coarse = proto.category_mapping[p["category_id"]]
        coarse_id = {"aircraft": 0, "ship": 1, "vehicle": 2}[coarse]
        # crop 特征
        if lg is not None:
            prob = softmax(lg)
            order = np.argsort(-prob)
            top1_c, top2_c = int(order[0]), int(order[1])
            crop_top1 = float(prob[top1_c])
            crop_margin = float(prob[top1_c] - prob[top2_c])
            crop_ent = float(-(prob * np.log(prob + 1e-9)).sum())
            agree = 1 if top1_c == p["category_id"] else 0
        else:
            crop_top1 = crop_margin = crop_ent = 0.0
            top1_c = -1
            agree = 0
        m = match_of.get(p["_idx"])
        is_valid = 1 if m is not None else 0
        matched_uid = m[1] if m is not None else ""
        official_match_iou = m[2] if m is not None else 0.0
        # fine_correct: 位置覆盖的 GT 中, 是否有细类与候选一致的 GT
        fine_correct = 0
        for (img_id, gi) in assoc.get(p["_idx"], set()):
            if formal.boxes[img_id][gi]["category_id"] == p["category_id"]:
                fine_correct = 1
                break
        # 局部密度(同图同粗类候选数)
        density = sum(1 for q in pb[p["image_id"]]
                      if proto.category_mapping[q["category_id"]] == coarse)
        node_rows.append([
            p["proposal_uid"], p["_idx"], p["image_id"], p["category_id"], coarse_id,
            p["score"], crop_top1, crop_margin, crop_ent, top1_c, agree,
            w, h, area, se, aspect, density,
            (x1 + x0) / 2, (y1 + y0) / 2,  # 中心
            is_valid, fine_correct, matched_uid, official_match_iou,
        ])

    node_header = ["proposal_uid", "idx", "image_id", "category_id", "coarse_id",
                   "y5_score", "crop_top1", "crop_margin", "crop_entropy",
                   "crop_top1_class", "detector_crop_agree",
                   "w", "h", "area", "short_edge", "aspect", "local_density",
                   "cx", "cy", "is_valid", "fine_correct", "matched_uid",
                   "official_match_iou"]
    nodes_path = out / "nodes.csv"
    with open(nodes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(node_header)
        w.writerows(node_rows)
    print(f"节点表: {nodes_path} ({len(node_rows)} 行)")

    # 边构建(每图空间相邻候选对)
    edge_rows = []
    for img_id, plist in pb.items():
        n = len(plist)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = plist[i], plist[j]
                iou = compute_iou(a["bbox_xyxy"], b["bbox_xyxy"])
                ax, ay = (a["bbox_xyxy"][0] + a["bbox_xyxy"][2]) / 2, (a["bbox_xyxy"][1] + a["bbox_xyxy"][3]) / 2
                bx, by = (b["bbox_xyxy"][0] + b["bbox_xyxy"][2]) / 2, (b["bbox_xyxy"][1] + b["bbox_xyxy"][3]) / 2
                cdist = math.hypot(ax - bx, ay - by)
                if iou <= 0 and cdist >= args.edge_center_dist:
                    continue
                # 尺寸比(小/大)
                sa = max(a["bbox_xyxy"][2] - a["bbox_xyxy"][0], 1e-6) * max(a["bbox_xyxy"][3] - a["bbox_xyxy"][1], 1e-6)
                sb = max(b["bbox_xyxy"][2] - b["bbox_xyxy"][0], 1e-6) * max(b["bbox_xyxy"][3] - b["bbox_xyxy"][1], 1e-6)
                size_ratio = min(sa, sb) / max(sa, sb)
                fine_same = 1 if a["category_id"] == b["category_id"] else 0
                coarse_same = 1 if proto.category_mapping[a["category_id"]] == proto.category_mapping[b["category_id"]] else 0
                # 边监督(用位置关联 assoc 定义 same/different)
                sa = assoc.get(a["_idx"], set())
                sb = assoc.get(b["_idx"], set())
                if sa and sb:
                    label = 0 if (sa & sb) else 1  # same / different
                else:
                    label = 2  # background_related
                edge_rows.append([
                    a["_idx"], b["_idx"], img_id,
                    round(iou, 4), round(cdist, 2), round(size_ratio, 4),
                    fine_same, coarse_same, label,
                ])
    edge_header = ["idx_a", "idx_b", "image_id", "iou", "center_dist",
                   "size_ratio", "fine_same", "coarse_same", "label"]
    with open(out / "edges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(edge_header)
        w.writerows(edge_rows)
    from collections import Counter
    labc = Counter(r[-1] for r in edge_rows)
    print(f"边表: {out/'edges.csv'} ({len(edge_rows)} 行)")
    print(f"  标签分布: same={labc.get(0,0)} different={labc.get(1,0)} background={labc.get(2,0)}")
    audit = {
        "status": "complete",
        "label_contract_version": label_result.contract_version,
        "matcher": "rsdet.evaluation.official_metric.evaluate_predictions_with_trace",
        "matching_policy": label_result.metrics.details["matching_policy"],
        "n_predictions": len(preds),
        "n_gt": int(label_result.metrics.details["total_gt"]),
        "tp": int(label_result.metrics.details["tp"]),
        "fp": int(label_result.metrics.details["fp"]),
        "fn": int(label_result.metrics.details["fn"]),
        "nodes_sha256": hashlib.sha256(nodes_path.read_bytes()).hexdigest(),
    }
    (out / "label_contract.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"标签合同: {out/'label_contract.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
