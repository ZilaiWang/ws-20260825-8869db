"""F1: foreground rejector 训练 manifest 生成(方案5 §七.2)。

bg_gate 的根因: 负样本只用人工白名单 clear_background, ship/vehicle 门控没学会
(0.14%/2.6%)。F1 改为用 Y5 proposal 域的真实 FP_BG 作负样本(近分布结构化背景),
并三分类: 真实目标 / 结构化背景(像 MS/QHS/FSC/SU-24 的) / 普通背景。

manifest 字段对齐 train_bg_gate.py 的 GATE_MANIFEST_COLUMNS。
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# 结构化背景的"幻觉类"(crop 教师把背景判成这些简单类)
STRUCTURED_CLASSES = {2: "QHS", 3: "MS", 23: "SU-24", 24: "FSC"}
CONTEXT_EXPANSION = 1.25


def coarse_name(category_id: int) -> str:
    if category_id < 4:
        return "ship"
    if category_id < 24:
        return "aircraft"
    return "vehicle"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-fpbg", type=int, default=30000, help="FP_BG 负样本上限")
    ap.add_argument("--structured-ratio", type=float, default=0.5, help="结构化背景在负样本中的比例")
    args = ap.parse_args()

    nodes = pd.read_csv(args.nodes)
    preds4 = json.load(open(args.preds_d4))
    bbox_of = {p["proposal_uid"]: p["bbox_xyxy"] for p in preds4}

    # image_id -> (source_relative_path, source_width, source_height, fold)
    img_map = {}
    with open(args.formal_crop_manifest, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            img_map.setdefault(int(r["formal_image_id"]), (
                r["source_relative_path"],
                float(r["source_width"]), float(r["source_height"]),
                int(r["fold"]),
            ))

    # 标签
    is_valid = nodes["is_valid"].to_numpy() == 1
    fine_correct = nodes["fine_correct"].to_numpy() == 1
    is_fpbg = (~is_valid) & (~fine_correct)
    crop_cls = nodes["crop_top1_class"].to_numpy()

    rows = []
    n_pos = 0
    n_struct = 0
    n_plain = 0
    for r in nodes.itertuples():
        uid = r.proposal_uid
        if uid not in bbox_of:
            continue
        bbox = bbox_of[uid]
        x0, y0, x1, y1 = [float(v) for v in bbox]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = x1 - x0, y1 - y0
        cw, ch = w * CONTEXT_EXPANSION, h * CONTEXT_EXPANSION
        cx0, cy0, cx1, cy1 = cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2
        if r.image_id not in img_map:
            continue
        rel_path, sw, sh, fold = img_map[r.image_id]

        if is_valid[r.idx]:
            # 正样本(官方 TP)
            cid = int(r.category_id)
            rows.append([uid, int(r.image_id), fold, coarse_name(cid),
                         cid, 1, round(float(r.y5_score), 6), int(r.idx),
                         round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2),
                         round(cx0, 2), round(cy0, 2), round(cx1, 2), round(cy1, 2),
                         rel_path, sw, sh])
            n_pos += 1
        elif is_fpbg[r.idx]:
            # 负样本(FP_BG), 按 crop 幻觉类分结构化/普通
            cid = int(r.category_id)
            if int(r.crop_top1_class) in STRUCTURED_CLASSES:
                is_struct = 1
                n_struct += 1
            else:
                is_struct = 0
                n_plain += 1
            # coarse 用候选被判成的粗类(门控学到"像 MS 的背景不是 MS")
            coarse = coarse_name(cid)
            rows.append([uid, int(r.image_id), fold, coarse,
                         25, 0, round(float(r.y5_score), 6), int(r.idx),
                         round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2),
                         round(cx0, 2), round(cy0, 2), round(cx1, 2), round(cy1, 2),
                         rel_path, sw, sh, is_struct])

    # 负样本下采样(结构化:普通 = structured_ratio)
    import random
    rng = random.Random(42)
    pos_rows = [r for r in rows if r[5] == 1]
    bg_rows = [r for r in rows if r[5] == 0]
    struct_rows = [r for r in bg_rows if r[-1] == 1]
    plain_rows = [r for r in bg_rows if r[-1] == 0]
    n_target_bg = min(args.max_fpbg, len(bg_rows))
    n_target_struct = int(n_target_bg * args.structured_ratio)
    n_target_plain = n_target_bg - n_target_struct
    sel_struct = rng.sample(struct_rows, min(n_target_struct, len(struct_rows)))
    sel_plain = rng.sample(plain_rows, min(n_target_plain, len(plain_rows)))
    sel_bg = sel_struct + sel_plain

    header = ["proposal_uid", "image_id", "fold", "coarse", "class_id",
              "is_foreground", "score", "source_prediction_index",
              "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
              "context_x0", "context_y0", "context_x1", "context_y1",
              "source_relative_path", "source_width", "source_height"]
    out_rows = pos_rows + [r[:-1] for r in sel_bg]  # 去掉 is_struct 标记
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out_rows)

    print(f"manifest: 正样本 {n_pos}, 结构化背景 {n_struct}(采 {len(sel_struct)}), "
          f"普通背景 {n_plain}(采 {len(sel_plain)})")
    print(f"总行数: {len(out_rows)} → {args.output}")


if __name__ == "__main__":
    main()
