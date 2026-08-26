"""Gate 5 探针：DINOv2 特征的"错误可观测性"验证。

方案6 §7 核心判断：deploy 改类天花板 = crop 分类器（P03 ConvNeXt-tiny）精度
（tp 上 5.5% 判错 → broken）。要突破，需更强的视觉证据。

本脚本验证：用 DINOv2 ViT-B/14 特征（比 P03 ConvNeXt-tiny 更强）在 formal GT crop
上做严格 OOF 的线性 probe，度量 top-1 准确率。若显著高于 P03，则 Gate 5 方向成立。

数据：/workspace/p04-cache/dinov2-vitb14-tight224-d4-v1（P04 阶段已提取，20933 GT crop）。

纯服务器（需 torch 读 DINOv2 特征 + sklearn）。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def load_dino_features(cache_dir: str, view: str = "r0"):
    """加载 DINOv2 特征，返回 (crop_id, feats, labels)。"""
    crop_ids, feats, labels = [], [], []
    for sp in sorted(glob.glob(str(Path(cache_dir) / "shard-*.npz"))):
        s = np.load(sp, allow_pickle=False)
        mask = s["row__view_id"] == view
        crop_ids.append(s["row__crop_id"][mask])
        feats.append(s["feature__dino_cls"][mask].astype(np.float32))
        labels.append(s["row__class_id_at_extraction"][mask])
    return (
        np.concatenate(crop_ids),
        np.concatenate(feats),
        np.concatenate(labels).astype(int),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--project-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--views", nargs="+", default=["r0"],
                    help="用哪些 view（r0 或全部 8 个做 mean）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # formal manifest: crop_id -> fold
    formal = load_formal_ground_truth(args.formal_crop_manifest)
    # objects 的 key 是 (image_id, gt_index)，无 crop_id；改从 csv 读 crop_id
    import pandas as pd
    df = pd.read_csv(args.formal_crop_manifest)
    crop_to_fold = dict(zip(df["crop_id"], df["fold"]))
    crop_to_cls = dict(zip(df["crop_id"], df["class_id"]))

    crop_ids, feats, labels = load_dino_features(args.cache_dir, args.views[0])
    folds = np.array([crop_to_fold.get(c, -1) for c in crop_ids], dtype=int)
    valid = folds >= 0
    crop_ids, feats, labels, folds = crop_ids[valid], feats[valid], labels[valid], folds[valid]
    print(f"DINOv2 特征: {len(feats)} 个 crop（view={args.views[0]}），feat dim={feats.shape[1]}")
    print(f"fold 分布: {dict(zip(*np.unique(folds, return_counts=True)))}")
    print(f"类别数: {len(np.unique(labels))}")

    # 严格 OOF 线性 probe
    scaler = StandardScaler()
    oof_pred = np.zeros(len(feats), dtype=int)
    oof_prob = np.zeros((len(feats), 25), dtype=float)
    for held in (0, 1, 2):
        tr = np.where(folds != held)[0]
        va = np.where(folds == held)[0]
        Xtr = scaler.fit_transform(feats[tr])
        Xva = scaler.transform(feats[va])
        clf = LogisticRegression(max_iter=2000, C=10.0, solver="lbfgs", multi_class="multinomial",
                                 random_state=args.seed)
        clf.fit(Xtr, labels[tr])
        oof_pred[va] = clf.predict(Xva)
        oof_prob[va] = clf.predict_proba(Xva)

    acc = accuracy_score(labels, oof_pred)
    print(f"\n=== DINOv2 线性 probe 严格 OOF top-1 准确率 ===")
    print(f"整体: {acc:.4f}")

    # 各类准确率
    per_cls = {}
    for c in range(25):
        m = labels == c
        if m.sum() > 0:
            per_cls[c] = float((oof_pred[m] == c).mean())
    print("各类准确率（低准确率类=混淆高风险）:")
    for c in sorted(per_cls, key=lambda x: per_cls[x])[:10]:
        print(f"  class {c}: {per_cls[c]:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"accuracy": acc, "per_class": {str(k): v for k, v in per_cls.items()}},
              open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已写: {args.output}")


if __name__ == "__main__":
    main()
