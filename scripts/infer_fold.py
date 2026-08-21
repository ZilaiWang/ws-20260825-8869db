#!/usr/bin/env python3
"""通用低阈值整图推理: 用指定权重对某折 val 图推理, 输出候选 preds.json。

供各创新实验(F2/D3/D4/V2/F3/F4/F6)的评估复用。输出:
image_id / category_id / score / bbox_xyxy, conf=0.001 低阈值。

用法(服务器):
  python scripts/infer_fold.py --weights <last.pt> --fold 0 \
    --split-view /workspace/results/Y5-ROT90-CV3-OOF/fold_0/split_view.json \
    --data-root /workspace/data --output /workspace/results/F2-FAMILY-FOLD0-40EP/F2-fold0-preds.json
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--split-view", required=True)
    ap.add_argument("--data-root", default="/workspace/data")
    ap.add_argument("--output", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=1024)
    args = ap.parse_args()

    view = json.loads(Path(args.split_view).read_text(encoding="utf-8"))
    val_rel = [s["relative_path"] for s in view["samples"] if s.get("split") == "val"]
    rel2id = {s["relative_path"]: int(s["image_id"]) for s in view["samples"]}
    print(f"fold{args.fold} val 图: {len(val_rel)}")

    model = YOLO(args.weights)
    out = []
    for i, rel in enumerate(val_rel):
        path = Path(args.data_root) / rel
        img_id = rel2id[rel]
        results = model.predict(str(path), conf=args.conf, imgsz=args.imgsz, verbose=False)
        for r in results:
            for box in r.boxes:
                x0, y0, x1, y1 = [float(v) for v in box.xyxy[0]]
                out.append({
                    "image_id": img_id,
                    "category_id": int(box.cls[0]),
                    "score": float(box.conf[0]),
                    "bbox_xyxy": [x0, y0, x1, y1],
                })
        if (i + 1) % 200 == 0:
            print(f"  已推理 {i + 1}/{len(val_rel)} (候选 {len(out)})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"fold{args.fold} 候选数: {len(out)}")
    print(f"INFER_DONE_fold{args.fold}")


if __name__ == "__main__":
    main()
