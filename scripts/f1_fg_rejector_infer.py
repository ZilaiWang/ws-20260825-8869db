"""F1 foreground rejector 推理: 对全部 Y5 候选渲染 context crop, 输出 foreground logit。

复用 train_bg_gate 的模型(ConvNeXt-T coarse foreground gate), 按 fold 用对应 checkpoint。
输出 {proposal_uid: foreground_logit}(shared + coarse residual)。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.background_gate import COARSE_CLASSES, expand_context_bbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--preds-d4", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--convnext-weights", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import csv
    import numpy as np
    import torch
    from PIL import Image

    from rsdet.models.background_gate_classifier import build_coarse_foreground_gate
    from rsdet.data.crop_classification import render_crop

    nodes = list(csv.DictReader(open(args.nodes, encoding="utf-8-sig")))
    preds4 = json.load(open(args.preds_d4))
    bbox_of = {p["proposal_uid"]: p["bbox_xyxy"] for p in preds4}

    # image_id -> (source_relative_path, fold)
    img_map = {}
    with open(args.formal_crop_manifest, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            img_map.setdefault(int(r["formal_image_id"]),
                               (r["source_relative_path"], int(r["fold"])))

    # 候选行
    rows = []
    for r in nodes:
        uid = r["proposal_uid"]
        img_id = int(r["image_id"])
        if uid not in bbox_of or img_id not in img_map:
            continue
        rel, fold = img_map[img_id]
        rows.append({"proposal_uid": uid, "image_id": img_id,
                     "fold": fold, "category_id": int(r["category_id"]),
                     "bbox_xyxy": bbox_of[uid], "rel": rel})
    print(f"候选: {len(rows)}")

    device = torch.device(args.device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    models = {}
    for fold in (0, 1, 2):
        ckpt = Path(args.checkpoint_dir) / f"bg_gate_fold{fold}_final.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"缺少 checkpoint: {ckpt}")
        model = build_coarse_foreground_gate(weight_path=args.convnext_weights,
                                             freeze="freeze_backbone",
                                             verify_weight_sha256=False, device=device)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        models[fold] = model

    data_root = Path(args.data_root)
    image_cache = {}
    result = {}
    # 按 fold 分组处理
    by_fold = defaultdict(list)
    for row in rows:
        by_fold[row["fold"]].append(row)

    for fold, fold_rows in by_fold.items():
        model = models[fold]
        for start in range(0, len(fold_rows), args.batch_size):
            chunk = fold_rows[start:start + args.batch_size]
            tensors = []
            for row in chunk:
                img_path = (data_root / row["rel"]).resolve()
                if img_path not in image_cache:
                    image_cache[img_path] = Image.open(img_path)
                context = expand_context_bbox(row["bbox_xyxy"], ratio=1.25)
                crop = render_crop(image_cache[img_path], context, args.resolution)
                arr = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)
                tensors.append(torch.from_numpy(arr))
            x = torch.stack(tensors).to(device)
            x = (x / 255.0 - mean) / std
            with torch.no_grad():
                out = model(x)
            shared = out.shared_logit.squeeze(1).cpu().tolist()
            # 粗类条件 logit = shared + coarse residual
            coarse_idx = [COARSE_CLASSES.index(
                "ship" if r["category_id"] < 4 else "aircraft" if r["category_id"] < 24 else "vehicle")
                for r in chunk]
            coarse_logits = out.coarse_logits.cpu().numpy()
            for j, row in enumerate(chunk):
                fg = float(shared[j] + coarse_logits[j, coarse_idx[j]])
                result[row["proposal_uid"]] = fg
    print(f"foreground logit 输出: {len(result)}")
    Path(args.output).write_text(json.dumps(result) + "\n")
    print(f"已写: {args.output}")


if __name__ == "__main__":
    main()
