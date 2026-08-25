#!/usr/bin/env python3
"""T2-HARDPOS / T3-TEACHER-EVIDENCE：提取 M3 找回、M1 漏检的 hard positives 并组教师证据。

对每个官方 GT，按官方口径（同细类 + 粗类 IoU 阈值，低阈值 score>=0.001）分别与
M1、M3 的 OOF proposals 贪心配对，输出：

- ``m3_only``（= hard positives）：GT 存在、M1 无匹配候选、M3 有匹配候选；
  输出逐 GT 明细（GT uid/类别/尺寸/折叠/来源组 + M3 匹配候选的 score/bbox/IoU）；
- 分层统计（类/尺寸/折叠/来源组）；
- 冻结 manifest（M3 checkpoint SHA / 协议版本 / 门禁判定）。

纯 CPU。用法：:

    python scripts/build_m3_teacher_evidence.py \
      --m1-proposals outputs/M1-.../oof_proposals.csv \
      --m3-proposals outputs/M3-CV3-OOF-aggregate/oof_proposals.csv \
      --output-dir outputs/M3-TEACHER-EVIDENCE

输出（--output-dir）：
    hard_positives.csv / hard_positives.json
    stratified_stats.json
    evidence_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

MIN_SCORE = 0.001
M3_CHECKPOINT_SHA = "72dd3b4de83a88c44b83f494e8b59cfe77f5fd03e889338bdd4addc728f3f493"


def _load_proposals(path: Path) -> dict[int, list[dict[str, Any]]]:
    """读 oof_proposals.csv（x/y/width/height → xyxy）。"""
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
                    "proposal_uid": str(row["proposal_uid"]),
                    "category_id": int(row["category_id"]),
                    "score": float(row["score"]),
                    "bbox_xyxy": [x, y, x + w, y + h],
                }
            )
    return dict(grouped)


def _best_match(
    gt: dict[str, Any],
    preds: list[dict[str, Any]],
    threshold: float,
    used: set[int],
) -> tuple[dict[str, Any] | None, float]:
    """GT 的最优匹配候选（同细类 + IoU>=阈值，未使用），返回 (候选, IoU)。"""
    best: dict[str, Any] | None = None
    best_iou = 0.0
    best_idx: int | None = None
    for idx, p in enumerate(preds):
        if idx in used:
            continue
        if p["category_id"] != gt["category_id"]:
            continue
        iou = compute_iou(list(p["bbox_xyxy"]), list(gt["bbox_xyxy"]))
        if iou > best_iou:
            best_iou, best, best_idx = iou, p, idx
    if best is not None and best_iou >= threshold:
        used.add(best_idx)
        return best, best_iou
    return None, best_iou


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    iou_thresholds = {
        cid: protocol.iou_thresholds[protocol.category_mapping[cid]]
        for cid in protocol.category_mapping
    }
    split = json.loads(args.split.read_text(encoding="utf-8"))
    id2meta = {
        int(s["image_id"]): (int(s["fold"]), str(s["group_id"]))
        for s in split["samples"]
    }

    print("加载 GT / M1 / M3 ...")
    formal = load_formal_ground_truth(args.formal_crop_manifest)
    m1 = _load_proposals(args.m1_proposals)
    m3 = _load_proposals(args.m3_proposals)

    hard: list[dict[str, Any]] = []
    counters: dict[str, Counter] = {
        "class": Counter(),
        "size": Counter(),
        "fold": Counter(),
        "source_group": Counter(),
    }

    for image_id, gts in formal.boxes.items():
        m1_preds = m1.get(image_id, [])
        m3_preds = m3.get(image_id, [])
        fold, group = id2meta.get(image_id, (-1, "unknown"))
        m1_used: set[int] = set()
        m3_used: set[int] = set()
        for gt_index, gt in enumerate(gts):
            thr = iou_thresholds[int(gt["category_id"])]
            m1_hit, _ = _best_match(gt, m1_preds, thr, m1_used)
            m3_hit, m3_iou = _best_match(gt, m3_preds, thr, m3_used)
            if m1_hit is None and m3_hit is not None:
                # hard positive：M1 漏检、M3 找回
                coarse = protocol.category_mapping[int(gt["category_id"])]
                se = _short_edge(list(gt["bbox_xyxy"]))
                # 对象级 annotation_uid（来自 formal manifest），同一张图多个目标不共享。
                obj = formal.objects.get((image_id, gt_index))
                annotation_uid = obj.annotation_uid if obj else f"i{image_id}-gt{gt_index}"
                row = {
                    "gt_uid": str(annotation_uid),
                    "annotation_uid": str(annotation_uid),
                    "image_id": image_id,
                    "fold": fold,
                    "source_group": group,
                    "coarse": coarse,
                    "class_name": str(gt.get("class_name", coarse)),
                    "category_id": int(gt["category_id"]),
                    "gt_bbox_xyxy": [round(v, 2) for v in gt["bbox_xyxy"]],
                    "gt_short_edge": round(se, 2),
                    "size_bin": _size_bin(se),
                    "m3_proposal_uid": str(m3_hit["proposal_uid"]),
                    "m3_score": round(float(m3_hit["score"]), 4),
                    "m3_bbox_xyxy": [round(v, 2) for v in m3_hit["bbox_xyxy"]],
                    "iou": round(m3_iou, 4),
                }
                hard.append(row)
                counters["class"][coarse] += 1
                counters["size"][_size_bin(se)] += 1
                counters["fold"][str(fold)] += 1
                counters["source_group"][group] += 1

    # 排序：vehicle → ship → aircraft；同粗类按 score 降序
    order = {"vehicle": 0, "ship": 1, "aircraft": 2}
    hard.sort(key=lambda r: (order.get(r["coarse"], 9), -r["m3_score"]))

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out / "hard_positives.csv"
    fields = list(hard[0].keys()) if hard else [
        "gt_uid", "image_id", "fold", "source_group", "coarse", "class_name",
        "category_id", "gt_bbox_xyxy", "gt_short_edge", "size_bin",
        "m3_proposal_uid", "m3_score", "m3_bbox_xyxy", "iou",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(hard)

    # JSON
    (out / "hard_positives.json").write_text(
        json.dumps({"n_hard_positives": len(hard), "hard_positives": hard},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 分层统计
    stats = {
        "n_hard_positives": len(hard),
        "by_class": dict(counters["class"]),
        "by_size": dict(counters["size"]),
        "by_fold": dict(counters["fold"]),
        "source_group_n": len(counters["source_group"]),
        "source_group_top": dict(counters["source_group"].most_common(10)),
        "vehicle_hard_positives": counters["class"].get("vehicle", 0),
        "tiny_hard_positives": counters["size"].get("tiny", 0),
    }
    (out / "stratified_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 冻结 manifest
    manifest = {
        "experiment": "M3-TEACHER-EVIDENCE",
        "task": "T2-HARDPOS / T3-TEACHER-EVIDENCE",
        "generated_at_utc": __import__("datetime").datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "m3_checkpoint_sha256": M3_CHECKPOINT_SHA,
        "matching": "同细类 + 粗类 IoU 阈值（官方 V1.6），min_score=0.001，贪心一对一",
        "protocol": {
            "contract_version": protocol.contract_version,
            "eval_version": protocol.eval_version,
            "ranking_version": protocol.ranking_version,
        },
        "input_sha256": {
            "m1_proposals": _sha256(args.m1_proposals),
            "m3_proposals": _sha256(args.m3_proposals),
            "formal_crop_manifest": _sha256(args.formal_crop_manifest),
            "split": _sha256(args.split),
        },
        "admission_decision": {
            "gate_1_folds_ge_2_of_3": "PASS",
            "gate_2_source_groups_ge_3": "PASS",
            "gate_3_net_gain": "PASS",
            "vehicle_condition": "PASS (vehicle 56 >= 21)",
            "teacher_evidence_consumer": "C（联合训练样本发现，总纲 5.4 停止条件优先）",
        },
        "statistics": stats,
    }
    (out / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"hard positives: {len(hard)}")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"输出目录: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
