#!/usr/bin/env python3
"""DFD 三折候选 → R1-0 格式 manifest(供 crop 推理)。

输入: outputs/E9-DFD-FOLD*/DFD-fold{fold}-preds.json
输出: outputs/E9-DFD-FOLD0/dfd-all-manifest.csv(含 fold 列, 过滤退化框)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FORMAL = "outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv"
rel_map = {}
with open(FORMAL, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        rel_map.setdefault(int(r["formal_image_id"]), r["source_relative_path"])

rows = []
degen = 0
for fold in (0, 1, 2):
    src = Path(f"outputs/E9-DFD-FOLD0/DFD-fold{fold}-preds.json")
    if not src.exists():
        src = Path(f"/tmp/DFD-fold{fold}-preds.json")
    preds = json.load(open(src))
    for i, p in enumerate(preds):
        img = int(p["image_id"])
        x0, y0, x1, y1 = [float(v) for v in p["bbox_xyxy"]]
        if x1 - x0 <= 1.0 or y1 - y0 <= 1.0:
            degen += 1
            continue
        rows.append([
            f"dfd-f{fold}-i{img}-p{i:06d}", img, fold, rel_map.get(img, ""),
            int(p["category_id"]), round(x0, 2), round(y0, 2),
            round(x1, 2), round(y1, 2), round(float(p["score"]), 6), i,
        ])

out = "outputs/E9-DFD-FOLD0/dfd-all-manifest.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["proposal_uid", "image_id", "fold", "relative_path", "category_id",
                "x0", "y0", "x1", "y1", "detector_score", "source_prediction_index"])
    w.writerows(rows)
print(f"DFD 三折 manifest: {len(rows)} 行 → {out}")
print("fold 分布:", dict(Counter(r[2] for r in rows)))
print("退化框过滤:", degen, "| relative_path 缺失:", sum(1 for r in rows if not r[3]))
