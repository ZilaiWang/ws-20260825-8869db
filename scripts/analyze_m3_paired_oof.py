#!/usr/bin/env python3
"""M3 与 M1 的 paired OOF 分析（改进方案1 4.2 节分析顺序）。

对每个 GT 目标，判断 M1 / M3 在低阈值（score>=0.001）下是否各自有匹配候选
（同细类 + 粗类 IoU 阈值），统计：

- M1-only TP：M1 检出、M3 未检出；
- M3-only TP：M3 检出、M1 未检出（= M3 补回的漏检）；
- both：共同检出；
- neither：双双未检出（残余 FN）。

同时按粗类（ship/aircraft/vehicle）、尺寸分层，输出 M3 的互补价值判断。

纯 CPU。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

MIN_SCORE = 0.001  # 低阈值 OOF 评估口径


def _load_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            x, y, w, h = (
                float(row["x"]),
                float(row["y"]),
                float(row["width"]),
                float(row["height"]),
            )
            grouped[int(row["image_id"])].append(
                {
                    "category_id": int(row["category_id"]),
                    "score": float(row["score"]),
                    "bbox_xyxy": [x, y, x + w, y + h],
                }
            )
    return dict(grouped)


def _matched(
    gt: dict[str, Any],
    preds: list[dict[str, Any]],
    threshold: float,
    used: set[int],
) -> bool:
    """GT 是否被任一未使用预测匹配（同细类 + IoU >= 阈值）。"""
    best_iou, best_idx = 0.0, None
    for idx, p in enumerate(preds):
        if idx in used:
            continue
        if p["category_id"] != gt["category_id"]:
            continue
        iou = compute_iou(list(p["bbox_xyxy"]), list(gt["bbox_xyxy"]))
        if iou > best_iou:
            best_iou, best_idx = iou, idx
    if best_idx is not None and best_iou >= threshold:
        used.add(best_idx)
        return True
    return False


def _short_edge(box: list[float]) -> float:
    return min(box[2] - box[0], box[3] - box[1])


def _size_bin(se: float) -> str:
    if se < 32:
        return "tiny"
    if se < 96:
        return "small"
    if se < 256:
        return "medium"
    return "large"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-proposals", type=Path, required=True)
    parser.add_argument("--m3-proposals", type=Path, required=True)
    parser.add_argument(
        "--formal-crop-manifest",
        type=Path,
        default=Path("outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv"),
    )
    parser.add_argument("--split", type=Path, default=Path("data/splits/cv3_airport_proxy_k60_v2.json"))
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    iou_thresholds = {
        cid: protocol.iou_thresholds[protocol.category_mapping[cid]]
        for cid in protocol.category_mapping
    }

    # image_id -> (fold, source_group)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    id2meta = {
        int(s["image_id"]): (int(s["fold"]), str(s["group_id"]))
        for s in split["samples"]
    }

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    m1 = _load_predictions(args.m1_proposals)
    m3 = _load_predictions(args.m3_proposals)

    total = {"m1_only": 0, "m3_only": 0, "both": 0, "neither": 0}
    per_coarse: dict[str, dict[str, int]] = defaultdict(
        lambda: {"m1_only": 0, "m3_only": 0, "both": 0, "neither": 0}
    )
    per_size: dict[str, dict[str, int]] = defaultdict(
        lambda: {"m1_only": 0, "m3_only": 0, "both": 0, "neither": 0}
    )
    per_fold: dict[int, dict[str, int]] = defaultdict(
        lambda: {"m1_only": 0, "m3_only": 0, "both": 0, "neither": 0}
    )
    m3_only_groups: dict[str, int] = defaultdict(int)  # M3-only 补回 GT 的 source group 分布

    for image_id, gts in formal.boxes.items():
        m1_preds = m1.get(image_id, [])
        m3_preds = m3.get(image_id, [])
        fold, group = id2meta.get(image_id, (-1, "unknown"))
        m1_used: set[int] = set()
        m3_used: set[int] = set()
        for gt in gts:
            coarse = protocol.category_mapping[int(gt["category_id"])]
            thr = iou_thresholds[int(gt["category_id"])]
            hit1 = _matched(gt, m1_preds, thr, m1_used)
            hit3 = _matched(gt, m3_preds, thr, m3_used)
            key = ("both" if hit1 and hit3 else "m1_only" if hit1 else "m3_only" if hit3 else "neither")
            total[key] += 1
            per_coarse[coarse][key] += 1
            per_size[_size_bin(_short_edge(list(gt["bbox_xyxy"])))][key] += 1
            per_fold[fold][key] += 1
            if key == "m3_only":
                m3_only_groups[group] += 1

    def summarize(table: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
        rows = []
        for group, counts in sorted(table.items(), key=lambda kv: str(kv[0])):
            n = sum(counts.values())
            m3_only = counts["m3_only"]
            m1_only = counts["m1_only"]
            rows.append(
                {
                    "group": group,
                    "n_gt": n,
                    "both": counts["both"],
                    "m1_only": m1_only,
                    "m3_only": m3_only,
                    "neither": counts["neither"],
                    "m3_recall_boost": round(m3_only / n, 4) if n else None,
                    "m1_recall_boost": round(m1_only / n, 4) if n else None,
                }
            )
        return rows

    # 门禁判定（改进方案1 4.2 节）
    folds_with_m3_only = sorted(
        f for f, c in per_fold.items() if c.get("m3_only", 0) > 0 and f >= 0
    )
    n_source_groups = len(m3_only_groups)
    n_m3_only_total = total["m3_only"]
    vehicle_m3_only = per_coarse.get("vehicle", {}).get("m3_only", 0)
    gate = {
        "gate_1_folds_ge_2_of_3": {
            "pass": len(folds_with_m3_only) >= 2,
            "folds_with_m3_only": folds_with_m3_only,
            "n_folds_with_m3_only": len(folds_with_m3_only),
        },
        "gate_2_source_groups_ge_3": {
            "pass": n_source_groups >= 3,
            "n_source_groups": n_source_groups,
            "top5_source_groups": dict(
                sorted(m3_only_groups.items(), key=lambda kv: -kv[1])[:5]
            ),
        },
        "gate_3_net_gain": {
            "pass": n_m3_only_total >= 30 or vehicle_m3_only >= 21,
            "n_m3_only_total": n_m3_only_total,
            "vehicle_m3_only": vehicle_m3_only,
        },
        "note": "gate_4(经 N2-CFG 后 FDR≤0.17) 与 gate_5(10K p95≤18s) 需 N2-CFG/E 正式测速后判定",
    }

    payload = {
        "model": "M3 (RT-DETR-L) vs M1 (YOLO26s) paired OOF",
        "min_score": MIN_SCORE,
        "total": {k: v for k, v in total.items()},
        "n_gt": sum(total.values()),
        "per_coarse": summarize(per_coarse),
        "per_size": summarize(per_size),
        "per_fold": {str(f): {k: v for k, v in c.items()} for f, c in sorted(per_fold.items()) if f >= 0},
        "m3_only_source_groups_top": dict(
            sorted(m3_only_groups.items(), key=lambda kv: -kv[1])[:10]
        ),
        "admission_gate": gate,
        "note": "m3_only = M3 补回的 M1 漏检 GT（低阈值下）；neither = 双模型均无候选的残余 FN。"
                "paired 匹配为同细类 + 粗类 IoU 阈值贪心（与官方口径一致）。",
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
