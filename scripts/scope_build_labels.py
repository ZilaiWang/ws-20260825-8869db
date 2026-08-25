#!/usr/bin/env python3
"""Gate 1a: 反事实动作价值标签生成（SCOPE Counterfactual Utility Engine）。

对「固定完整候选集合」，为「模糊候选」生成 KEEP/DROP/RELABEL 动作的 delta_utility：
    delta = S(完整集合执行动作后的预测) - S(完整集合基线预测)

其中 S 是官方 Recall@FDR=0.12 前沿分数（FrontierScorer）。

关键：frontier 的 base 状态用**完整候选集合**（所有作用域候选），动作只对
**模糊候选**（score 落在 ambiguous_range）生成——否则 base frontier 会因候选
被截断而失真（这正是上一版的 bug）。

GT 仅在本离线标签阶段使用；deploy 包严禁 import 本脚本。
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config
from rsdet.scope.official_scorer import FrontierScorer, CandidateView

from scope_router.actions import Action, ActionKind, Candidate
from scope_router.counterfactual import CounterfactualLabelBuilder, write_jsonl

FEAT_BASE = ["y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
             "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density"]


class RepositoryFrontierScorer:
    """把 scope_router.Candidate 集合映射为官方 frontier 分数（ExactScorer 协议）。"""

    def __init__(self, proto, gt_boxes, image_ids):
        self.scorer = FrontierScorer(proto, gt_boxes, image_ids)

    def __call__(self, predictions, ground_truth) -> float:
        views = [
            CandidateView(
                _idx=int(c.payload["_idx"]),
                image_id=int(c.payload["image_id"]),
                category_id=int(c.cls_id),
                score=float(c.score),
                bbox_xyxy=[float(v) for v in c.box_xyxy],
            )
            for c in predictions
        ]
        return self.scorer.score(views, fdr_level=0.12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--oto-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--fold", type=int, default=None, help="限制单折（快速验证）")
    ap.add_argument("--max-candidates", type=int, default=None, help="限制模糊候选数（快速验证）")
    ap.add_argument("--crop-top1-min", type=float, default=0.5, help="RELABEL 候选的 crop_top1 置信下界")
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}

    nodes = pd.read_csv(args.nodes)
    nodes["fold"] = nodes["image_id"].map(fold_of)
    node_map = {int(r.idx): r for r in nodes.itertuples()}

    preds = json.loads(Path(args.preds_d4).read_text(encoding="utf-8"))
    for i, p in enumerate(preds):
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        p["_idx"] = i

    oto_by_img = defaultdict(list)
    for f in (0, 1, 2):
        for p in json.loads(Path(args.oto_dir, f"a5_oto_fold{f}.json").read_text(encoding="utf-8")):
            p["category_id"] = int(p["category_id"])
            p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
            p["image_id"] = int(p["image_id"])
            oto_by_img[p["image_id"]].append(p)

    has_oto = np.zeros(len(preds), dtype=np.int8)
    for i, p in enumerate(preds):
        for q in oto_by_img.get(p["image_id"], []):
            if q["category_id"] != p["category_id"] or q["score"] < 0.5:
                continue
            if compute_iou(p["bbox_xyxy"], q["bbox_xyxy"]) > 0.5:
                has_oto[i] = 1
                break
    nodes["has_oto"] = has_oto

    from sklearn.ensemble import HistGradientBoostingClassifier
    X = nodes[FEAT_BASE + ["has_oto"]].to_numpy(dtype=float)
    y = nodes["is_valid"].to_numpy(dtype=float)
    folds = nodes["fold"].to_numpy()
    oer = np.zeros(len(nodes))
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6,
            l2_regularization=1.0, class_weight="balanced", random_state=2026 + held,
        )
        clf.fit(X[tr], y[tr])
        oer[va] = clf.predict_proba(X[va])[:, 1]

    if args.fold is not None:
        image_ids = {img for (img, _), o in formal.objects.items() if o.fold == args.fold}
    else:
        image_ids = set(formal.boxes.keys())

    # 完整候选集合（frontier base）+ 动作候选（可能改类/删除）
    all_candidates: list[Candidate] = []
    amb_candidates: list[Candidate] = []
    for i, p in enumerate(preds):
        if p["image_id"] not in image_ids:
            continue
        s = float(oer[i])
        r = node_map.get(i)
        c = Candidate(
            candidate_id=str(i),
            box_xyxy=tuple(p["bbox_xyxy"]),
            cls_id=int(p["category_id"]),
            score=s,
            payload={
                "_idx": i,
                "image_id": int(p["image_id"]),
                "crop_top1": float(r.crop_top1) if r is not None else 0.0,
                "crop_top1_class": int(r.crop_top1_class) if r is not None else int(p["category_id"]),
                "y5_score": float(p.get("score", 0.0)),
            },
        )
        all_candidates.append(c)
        # 动作候选 = crop 认为类别不同且 crop 足够自信（可能改类）
        alt = int(c.payload["crop_top1_class"])
        if alt != c.cls_id and float(c.payload["crop_top1"]) >= args.crop_top1_min:
            amb_candidates.append(c)

    if args.max_candidates is not None:
        amb_candidates = amb_candidates[: args.max_candidates]

    scorer = RepositoryFrontierScorer(proto, formal.boxes, image_ids)
    builder = CounterfactualLabelBuilder(
        scorer, scorer_version="official-frontier-v1", cache_dir=args.cache_dir,
    )

    # 动作只对模糊候选：KEEP + DROP + RELABEL(crop_top1_class)
    actions_by_candidate: dict[str, list[Action]] = {}
    for c in amb_candidates:
        acts = [Action(ActionKind.KEEP), Action(ActionKind.DROP)]
        alt = int(c.payload["crop_top1_class"])
        if alt != c.cls_id:
            acts.append(Action(ActionKind.RELABEL, cls_id=alt))
        actions_by_candidate[c.candidate_id] = acts

    base = scorer(all_candidates, None)
    print(f"完整候选数: {len(all_candidates)}, 模糊候选数: {len(amb_candidates)}")
    print(f"base frontier(fold{'%d' % args.fold if args.fold is not None else 'all'}) = {base:.4f}")

    rows = builder.build_group(
        group_id="fold" + ("%d" % args.fold if args.fold is not None else "all"),
        predictions=all_candidates,
        ground_truth=None,
        actions_by_candidate=actions_by_candidate,
        detector_model_id="y5-oer",
        detector_train_groups=[],
    )

    write_jsonl(rows, args.output)
    deltas = np.array([r.delta_utility for r in rows])
    print(f"动作记录: {len(rows)}")
    print(f"delta_utility: mean={deltas.mean():.2e} min={deltas.min():.2e} max={deltas.max():.2e} "
          f"正收益占比={(deltas > 0).mean():.3f} 非零占比={(deltas != 0).mean():.3f}")
    print(f"已写: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
