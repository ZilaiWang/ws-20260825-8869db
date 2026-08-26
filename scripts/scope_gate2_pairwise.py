"""Gate 2 前提验证：pairwise action interaction Δ_ij 是否显著非零。

背景：
- 单候选 delta_utility 极稀疏（oracle 改类候选 97.1% 的 delta_tp=0），
  因为"匹配抢占"——单改一个候选时，另一个候选仍覆盖同一 GT，TP 不变。
- 组合改类的收益来自匹配重新分配，是二阶交互项。

本脚本验证：对 top 模糊候选，成对动作的组合收益
    Δ_ij = uij - ui - uj + u0
是否显著非零。若显著，则方案6 Gate 2 的 U4(pairwise action utility)
关系集合网络方向成立；否则需要先补可观测证据。

只读本地资产，纯 CPU。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config
from rsdet.scope.official_scorer import CandidateView
from rsdet.scope.incremental_scorer import IncrementalFrontierScorer
from sklearn.ensemble import HistGradientBoostingClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--top-images", type=int, default=20, help="oracle改类候选最多的 image 数")
    ap.add_argument("--top-cands-per-img", type=int, default=10, help="每 image 取 top 模糊候选数")
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}

    nodes = pd.read_csv(args.nodes)
    nodes["fold"] = nodes["image_id"].map(fold_of)
    node_map = {int(r.idx): r for r in nodes.itertuples()}

    preds = json.load(open(args.preds_d4))
    for i, p in enumerate(preds):
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        p["_idx"] = i

    # OER 分数（复用 gate0 逻辑）
    oto_by_img = defaultdict(list)
    for f in (0, 1, 2):
        fp = Path(args.oto_dir) / f"a5_oto_fold{f}.json"
        if fp.exists():
            for p in json.load(open(fp)):
                oto_by_img[int(p["image_id"])].append(p)
    has_oto = np.zeros(len(preds), dtype=np.int8)
    for i, p in enumerate(preds):
        for q in oto_by_img.get(p["image_id"], []):
            if int(q["category_id"]) == p["category_id"] and q["score"] > 0.5 and \
                    compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > 0.5:
                has_oto[i] = 1
                break
    nodes["has_oto"] = has_oto
    FEAT_BASE = ["y5_score", "crop_top1", "crop_margin", "crop_entropy",
                 "crop_top1_class", "detector_crop_agree", "w", "h", "area",
                 "short_edge", "aspect", "local_density"]
    X = nodes[FEAT_BASE + ["has_oto"]].to_numpy(float)
    y = nodes["is_valid"].to_numpy(float)
    folds = nodes["fold"].to_numpy()
    oer = np.zeros(len(nodes))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=6, l2_regularization=1.0,
                                             class_weight="balanced", random_state=2026 + held)
        clf.fit(X[tr], y[tr])
        oer[va] = clf.predict_proba(X[va])[:, 1]

    image_ids = {img for (img, _), o in formal.objects.items() if o.fold == args.fold}

    # gt_fine
    pb = defaultdict(list)
    for p in preds:
        if p["image_id"] in image_ids:
            pb[p["image_id"]].append(p)
    gt_fine = defaultdict(set)
    for img, gts in formal.boxes.items():
        if img not in image_ids:
            continue
        for p in pb.get(img, []):
            for g in gts:
                thr = proto.iou_thresholds[proto.category_mapping[int(g["category_id"])]]
                if compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= thr:
                    gt_fine[p["_idx"]].add(int(g["category_id"]))

    # oracle 改类候选（位置匹配 GT 但类别错，且 crop_top1 命中 GT）
    relabel_cand_by_img = defaultdict(list)
    for p in preds:
        if p["image_id"] not in image_ids:
            continue
        i = p["_idx"]
        gf = gt_fine.get(i, set())
        if not gf or p["category_id"] in gf:
            continue
        r = node_map.get(i)
        if r is not None and int(r.crop_top1_class) in gf:
            relabel_cand_by_img[p["image_id"]].append(i)

    # 选 oracle 改类候选最多的 image
    top_imgs = sorted(relabel_cand_by_img, key=lambda i: -len(relabel_cand_by_img[i]))[: args.top_images]
    print(f"fold{args.fold}: oracle改类候选总数={sum(len(v) for v in relabel_cand_by_img.values())}, "
          f"分布 {len(relabel_cand_by_img)} image")
    print(f"top-{args.top_images} image 的改类候选数: "
          f"{[len(relabel_cand_by_img[i]) for i in top_imgs]}")

    # 构造全部候选（fold0）+ 增量 scorer
    all_cands = [
        CandidateView(_idx=i, image_id=p["image_id"], category_id=p["category_id"],
                      score=float(oer[i]), bbox_xyxy=p["bbox_xyxy"])
        for i, p in enumerate(preds) if p["image_id"] in image_ids
    ]
    scorer = IncrementalFrontierScorer(proto, formal.boxes, image_ids, all_cands)

    def frontier_with_relabel(relabel_map: dict[int, int]) -> float:
        edits = {i: ("relabel", c) for i, c in relabel_map.items()}
        u, _, _ = scorer.score_after_multi(edits)
        return u

    u0 = scorer.score()
    print(f"fold{args.fold} base frontier = {u0:.4f}")

    # 对每个 top image，取 top 模糊候选，两两配对算 Δ_ij
    deltas = []
    n_pairs = 0
    for img in top_imgs:
        cand_ids = relabel_cand_by_img[img][: args.top_cands_per_img]
        # 每个候选的目标类 = crop_top1_class
        alt = {i: int(node_map[i].crop_top1_class) for i in cand_ids}
        # 单候选 utility
        ui = {i: frontier_with_relabel({i: alt[i]}) for i in cand_ids}
        # pairwise
        for a in range(len(cand_ids)):
            for b in range(a + 1, len(cand_ids)):
                ia, ib = cand_ids[a], cand_ids[b]
                uij = frontier_with_relabel({ia: alt[ia], ib: alt[ib]})
                dij = uij - ui[ia] - ui[ib] + u0
                deltas.append(dij)
                n_pairs += 1

    deltas = np.array(deltas)
    print(f"\n=== pairwise interaction Δ_ij 统计（{n_pairs} 对）===")
    print(f"非零占比: {(deltas != 0).mean():.3f}")
    print(f"正(协同增益)占比: {(deltas > 0).mean():.3f}")
    print(f"负(互斥)占比: {(deltas < 0).mean():.3f}")
    print(f"|Δ_ij| 分布: min={np.abs(deltas).min():.2e} p50={np.percentile(np.abs(deltas), 50):.2e} "
          f"p90={np.percentile(np.abs(deltas), 90):.2e} max={np.abs(deltas).max():.2e}")
    print(f"Δ_ij 分布: min={deltas.min():.2e} p50={np.percentile(deltas, 50):.2e} max={deltas.max():.2e}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out = {
        "fold": args.fold,
        "n_pairs": n_pairs,
        "nonzero_ratio": float((deltas != 0).mean()),
        "positive_ratio": float((deltas > 0).mean()),
        "negative_ratio": float((deltas < 0).mean()),
        "abs_p50": float(np.percentile(np.abs(deltas), 50)),
        "abs_p90": float(np.percentile(np.abs(deltas), 90)),
        "abs_max": float(np.abs(deltas).max()),
    }
    json.dump(out, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已写: {args.output}")


if __name__ == "__main__":
    main()
