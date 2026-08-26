#!/usr/bin/env python3
"""Gate 1a: 反事实动作价值标签生成（含 ΔU + ΔTP/FP/FN 边际贡献）。

对「固定完整候选集合」，为「动作候选」生成动作的监督信号：
  delta_utility  = S(动作后) - S(基线)          —— 官方 frontier 分数变化（最终目标）
  delta_tp/fp/fn = 动作后的全局 TP/FP/FN - 基线  —— 匹配状态边际变化（平滑可学习信号）

关键：delta_utility 非平滑（frontier 阶梯），单候选动作大多为 0；但 delta_tp/fp/fn
是确定的边际贡献（单个候选从 FP→TP 即 +1/-1），是方案6 §5.3 的 λ3 辅助监督。

GT 仅在本离线标签阶段使用；deploy 包严禁 import 本脚本。
"""
from __future__ import annotations

import argparse
import hashlib
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
from rsdet.scope.incremental_scorer import IncrementalFrontierScorer

from scope_router.actions import Action, ActionKind, Candidate, apply_action

FEAT_BASE = ["y5_score", "crop_top1", "crop_margin", "crop_entropy", "crop_top1_class",
             "detector_crop_agree", "w", "h", "area", "short_edge", "aspect", "local_density"]


def to_views(candidates: list[Candidate]) -> list[CandidateView]:
    return [
        CandidateView(
            _idx=int(c.payload["_idx"]),
            image_id=int(c.payload["image_id"]),
            category_id=int(c.cls_id),
            score=float(c.score),
            bbox_xyxy=[float(v) for v in c.box_xyxy],
        )
        for c in candidates
    ]


def _cache_path(cache_dir, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.json"


def _read_cache(cache_dir, key: str) -> dict | None:
    p = _cache_path(cache_dir, key)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _write_cache(cache_dir, key: str, payload: dict) -> None:
    p = _cache_path(cache_dir, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


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
    ap.add_argument("--max-candidates", type=int, default=None, help="限制动作候选数")
    ap.add_argument("--crop-top1-min", type=float, default=0.5, help="RELABEL 候选 crop_top1 置信下界")
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    proto = parse_evaluation_protocol(load_config(args.project_config))
    fold_of = {i: o.fold for (i, _), o in formal.objects.items()}

    nodes = pd.read_csv(args.nodes)
    nodes["fold"] = nodes["image_id"].map(fold_of)
    node_map = {int(r.idx): r for r in nodes.itertuples()}

    preds = json.loads(Path(args.preds_d4).read_text(encoding="utf-8"))
    d4_map = {}
    for p in preds:
        p["category_id"] = int(p["category_id"])
        p["bbox_xyxy"] = [float(v) for v in p["bbox_xyxy"]]
        p["image_id"] = int(p["image_id"])
        if "d4_support" in p:
            d4_map[p.get("proposal_uid")] = int(p.get("d4_support", 0))
    for i, p in enumerate(preds):
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
            },
        )
        all_candidates.append(c)
        alt = int(c.payload["crop_top1_class"])
        if alt != c.cls_id and float(c.payload["crop_top1"]) >= args.crop_top1_min:
            amb_candidates.append(c)

    if args.max_candidates is not None:
        amb_candidates = amb_candidates[: args.max_candidates]

    scorer = IncrementalFrontierScorer(proto, formal.boxes, image_ids, to_views(all_candidates))

    # base 状态
    base_u = scorer.score()
    base_tp, base_fp = scorer.n_tp, scorer.n_fp
    print(f"完整候选: {len(all_candidates)}, 动作候选: {len(amb_candidates)}")
    print(f"base frontier = {base_u:.4f} (TP={base_tp}, FP={base_fp})")

    # 动作：DROP + RELABEL(crop_top1)（增量计算，streaming 写避免内存累积）
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    du_list = []
    dtp_list = []
    import gc
    with open(args.output, "w", encoding="utf-8") as f:
        for ci, c in enumerate(amb_candidates):
            alt = int(c.payload["crop_top1_class"])
            for act_kind, new_cls in (("drop", None), ("relabel", alt)):
                utility, tp, fp = scorer.score_after(int(c.candidate_id), act_kind, new_cls)
                du = utility - base_u
                dtp = tp - base_tp
                dfp = fp - base_fp
                r = {
                    "candidate_id": c.candidate_id,
                    "action": {"kind": act_kind, "cls_id": new_cls},
                    "base_utility": base_u,
                    "action_utility": utility,
                    "delta_utility": du,
                    "delta_tp": dtp,
                    "delta_fp": dfp,
                    "delta_fn": -(dtp),
                }
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
                du_list.append(du)
                dtp_list.append(dtp)
            if (ci + 1) % 2000 == 0:
                gc.collect()
                print(f"进度 {ci+1}/{len(amb_candidates)}", flush=True)

    du = np.array(du_list)
    dtp = np.array(dtp_list)
    print(f"动作记录: {len(du_list)}")
    print(f"delta_utility: 正收益占比={(du > 0).mean():.3f} 非零占比={(du != 0).mean():.3f}")
    print(f"delta_tp: 正={(dtp > 0).mean():.3f} 负={(dtp < 0).mean():.3f} 零={(dtp == 0).mean():.3f}")
    print(f"已写: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
