#!/usr/bin/env python3
"""COPH 三折候选 → R1-0 格式 manifest(供 crop 推理)。

输入: fold0/1/2 的 COPH 预测 JSON(服务器 e8_fold*_infer.py 输出)
输出: /tmp/COPH-all-manifest.csv(含 fold 列)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# formal manifest: formal_image_id -> source_relative_path
FORMAL = "outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv"
FOLD_MAP = {}  # image_id -> fold
with open(FORMAL, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        FOLD_MAP[int(r["formal_image_id"])] = int(r["fold"])

rel_map = {}
with open(FORMAL, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        rel_map.setdefault(int(r["formal_image_id"]), r["source_relative_path"])

rows = []
for fold in (0, 1, 2):
    preds = json.load(open(f"/tmp/COPH-fold{fold}-preds.json"))
    for i, p in enumerate(preds):
        img = int(p["image_id"])
        x0, y0, x1, y1 = p["bbox_xyxy"]
        rows.append([
            f"coph-f{fold}-i{img}-p{i:06d}", img, fold, rel_map.get(img, ""),
            int(p["category_id"]), round(float(x0), 2), round(float(y0), 2),
            round(float(x1), 2), round(float(y1), 2), round(float(p["score"]), 6), i,
        ])

out = "/tmp/COPH-all-manifest.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["proposal_uid", "image_id", "fold", "relative_path", "category_id",
                "x0", "y0", "x1", "y1", "detector_score", "source_prediction_index"])
    w.writerows(rows)
print(f"COPH 三折 manifest: {len(rows)} 行 → {out}")
print("fold 分布:", dict(Counter(r[2] for r in rows)))
print("relative_path 缺失:", sum(1 for r in rows if not r[3]))
