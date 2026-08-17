#!/usr/bin/env python3
"""Y4 AFSS 相关性正式分析（材料 19 Y4 第二阶段证据）。

用 R1-6 低阈值 OOF 预测，逐图计算：
  - 学习充分度 suff = min(precision, recall)（复用 afss_diagnose 口径）；
  - 错误构成（FP_BG / FP_CLS / FP_DUP / FP_LOC / FN_CLS / FN_LOC / FN_MISS / TP）。

然后量化"充分度 ↔ 错误类型"的相关性（AFSS 采样器成立的正式依据）：

  - point-biserial：suff 与"该图是否存在某类错误"（二值）的相关 + p 值；
  - Spearman：suff 与错误计数等级相关；
  - 分组对比：有/无某类错误的图，suff 均值差 + Welch t 检验。

纯 CPU。输出 JSON（含每图错误表 + 相关矩阵 + 结论）。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

from rsdet.analysis.oof_detection import (
    decompose_official_errors,
    load_formal_ground_truth,
)
from rsdet.evaluation.official_metric import evaluate_predictions_with_trace
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

PRED_REASONS = ("FP_BG", "FP_CLS", "FP_DUP", "FP_LOC")
GT_REASONS = ("FN_CLS", "FN_LOC", "FN_MISS")


def _per_image_error_table(
    formal,
    pred_boxes,
    protocol,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    """逐图错误计数 + TP 计数（官方 trace 口径）。"""
    _, trace = evaluate_predictions_with_trace(
        formal.boxes,
        pred_boxes,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    tp_per_image: dict[int, int] = defaultdict(int)
    for match in trace.matches:
        tp_per_image[int(match.image_id)] += 1

    _, cases, _ = decompose_official_errors(
        formal,
        pred_boxes,
        threshold=0.0,
        protocol=protocol,
        model_key="R1_6_FINAL_CHAIN",
        include_cases=True,
    )
    table: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in cases:
        image_id = int(case["image_id"])
        reason = str(case["reason"])
        table[image_id][reason] += 1
    return {i: dict(c) for i, c in table.items()}, dict(tp_per_image)


def _point_biserial(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """x 连续（充分度）、y 二值（有无错误）→ point-biserial + p 值。"""
    if len(set(y)) < 2:
        return {"r": None, "p": None, "note": "该错误类型无变化（全有或全无），无法计算"}
    r = stats.pointbiserialr(x, y)
    return {"r": round(float(r.correlation), 4), "p": round(float(r.pvalue), 4)}


def _spearman(x: Sequence[float], y: Sequence[int]) -> dict[str, Any]:
    if len(set(y)) < 2:
        return {"rho": None, "p": None, "note": "计数无变化，无法计算"}
    rho = stats.spearmanr(x, y)
    return {"rho": round(float(rho.correlation), 4), "p": round(float(rho.pvalue), 4)}


def _group_compare(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """有/无该错误的两组充分度：均值、差异、Welch t p 值。"""
    has = [float(v) for v, flag in zip(x, y) if flag]
    hasnt = [float(v) for v, flag in zip(x, y) if not flag]
    if len(has) < 2 or len(hasnt) < 2:
        return {"mean_with": None, "mean_without": None, "diff": None, "p": None}
    mean_with = float(np.mean(has))
    mean_without = float(np.mean(hasnt))
    t = stats.ttest_ind(has, hasnt, equal_var=False)
    return {
        "mean_with": round(mean_with, 4),
        "mean_without": round(mean_without, 4),
        "diff": round(mean_with - mean_without, 4),
        "p": round(float(t.pvalue), 4),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--formal-crop-manifest",
        type=Path,
        default=Path("outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/splits/cv3_airport_proxy_k60_v2.json"),
    )
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import sys

    # 仓库根目录加入路径，保证 `from scripts.afss_diagnose import ...` 可导入
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from scripts.afss_diagnose import _load_predictions, _per_image_sufficiency

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    iou_thresholds = {
        str(cid): protocol.iou_thresholds[coarse]
        for cid, coarse in protocol.category_mapping.items()
    }
    pred_boxes = _load_predictions(args.predictions)
    formal = load_formal_ground_truth(
        args.formal_crop_manifest, expected_images=4481, expected_annotations=20933
    )
    doc = json.loads(args.split.read_text(encoding="utf-8"))
    split = {
        int(s["image_id"]): (int(s["fold"]), str(s["group_id"])) for s in doc["samples"]
    }

    suff = _per_image_sufficiency(formal.boxes, pred_boxes, iou_thresholds)
    error_table, tp_per_image = _per_image_error_table(formal, pred_boxes, protocol)

    # 逐图特征向量（对齐 suff / error / tp / fold / source_group）
    all_image_ids = sorted(set(suff) | set(error_table) | set(tp_per_image))
    rows: list[dict[str, Any]] = []
    for image_id in all_image_ids:
        counts = error_table.get(image_id, {})
        fold, group = split.get(image_id, (-1, "unknown"))
        rows.append(
            {
                "image_id": image_id,
                "fold": fold,
                "source_group": group,
                "suff": round(float(suff.get(image_id, 0.0)), 4),
                "tp": int(tp_per_image.get(image_id, 0)),
                **{r: int(counts.get(r, 0)) for r in PRED_REASONS},
                **{r: int(counts.get(r, 0)) for r in GT_REASONS},
                "n_errors": int(
                    sum(counts.get(r, 0) for r in PRED_REASONS)
                    + sum(counts.get(r, 0) for r in GT_REASONS)
                ),
            }
        )

    x = np.asarray([r["suff"] for r in rows], dtype=float)
    correlations: dict[str, Any] = {}
    error_keys = ("tp", *PRED_REASONS, *GT_REASONS, "n_errors")
    for key in error_keys:
        y = np.asarray([r[key] for r in rows], dtype=float)
        has_error = np.asarray([r[key] > 0 for r in rows], dtype=float)
        correlations[key] = {
            "point_biserial_has": _point_biserial(x.tolist(), has_error.tolist()),
            "spearman_count": _spearman(x.tolist(), y.astype(int).tolist()),
            "group_compare": _group_compare(x.tolist(), has_error.tolist()),
            "n_images_with": int(has_error.sum()),
        }

    # 困难图（最低 10%）错误构成对比：全图 vs 困难图
    n_hard = max(1, len(rows) // 10)
    hard_ids = {r["image_id"] for r in sorted(rows, key=lambda r: r["suff"])[:n_hard]}
    hard_rows = [r for r in rows if r["image_id"] in hard_ids]
    hard_profile: dict[str, Any] = {}
    for key in error_keys:
        hard_sum = sum(r[key] for r in hard_rows)
        all_sum = sum(r[key] for r in rows)
        hard_profile[key] = {
            "hard_count": hard_sum,
            "all_count": all_sum,
            "hard_share": round(hard_sum / all_sum, 4) if all_sum else None,
            "hard_share_of_images": round(len(hard_rows) / len(rows), 4),
        }

    payload = {
        "model": "R1-6（低阈值 OOF）",
        "n_images": len(rows),
        "suff_mean": round(float(x.mean()), 4),
        "correlations": correlations,
        "hard_profile_lowest10pct": hard_profile,
        "per_image": rows,
        "conclusion": (
            "point_biserial_has.r < 0 且 p<0.05：充分度越低，越容易含该错误"
            "（采样器应持续参与）；spearman_count.rho < 0：错误越多充分度越低。"
            "hard_profile 显示困难图错误占比若显著高于其图占比(10%)，则该错误"
            "是 AFSS 的主要目标。"
        ),
        "note": (
            "suff = 每图 min(precision, recall)；错误按官方 V1.6 trace/decompose 口径。"
            "相关性成立即 Y4 采样器（困难图持续参与/中等图短期覆盖/容易图低频回看）"
            "有正式依据。"
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 简洁 stdout（sealed 摘要）
    print(f"n_images={len(rows)} suff_mean={payload['suff_mean']}")
    for key in ("FP_BG", "FP_CLS", "FP_DUP", "FP_LOC", "FN_CLS", "FN_LOC", "FN_MISS", "tp"):
        c = correlations[key]
        pb = c["point_biserial_has"]
        gc = c["group_compare"]
        print(
            f"  {key:8s} n_with={c['n_images_with']:5d} r_pb={pb.get('r')} p={pb.get('p')} "
            f"diff={gc.get('diff')} p={gc.get('p')} rho={c['spearman_count'].get('rho')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
