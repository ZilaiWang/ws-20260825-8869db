#!/usr/bin/env python3
"""构建 N2-CFG 正式 outer-fold-pure 三折的 inner 70/15/15 划分。

依据执行总纲 8.2 节「冻结的 outer-train 独立 group-calibration 协议」：

- 对每个 outer fold f，只在它的 outer-train source groups（fold != f）内，
  以 seed 202625、只依据图像/组统计（不看预测结果）确定性划分：
      inner_model_train   ≈ 70% images
      inner_error_mining  ≈ 15% images
      inner_score_calib   ≈ 15% images
- 三部分 group_id 互斥（同一 source group 不跨 split）；
- 粗类（ship/aircraft/vehicle）覆盖约束检查（从 group_rule 推导）。

细类（25 类）覆盖依赖 formal_crop_manifest 的 class_id → group 映射，本脚本
只做粗类覆盖检查并如实记录，细类覆盖待 GT 映射完善后补验。

纯 CPU，不依赖 GPU / 预测结果。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coarse_of_rule(group_rule: str) -> str:
    if group_rule == "ship_scene_id":
        return "ship"
    if group_rule == "mar20_airport_proxy_k60":
        return "aircraft"
    if group_rule == "vehicle_latlon":
        return "vehicle"
    raise ValueError(f"未知 group_rule: {group_rule}")


def _greedy_assign(groups: list[dict[str, Any]], seed: int) -> dict[str, list[str]]:
    """按 seed 打乱 group，贪心分配 image 比例 70/15/15。"""
    rng = random.Random(seed)
    shuffled = groups[:]
    rng.shuffle(shuffled)

    total = sum(g["n_images"] for g in groups)
    targets = {"inner_model_train": 0.70 * total, "inner_error_mining": 0.15 * total,
               "inner_score_calib": 0.15 * total}
    assigned: dict[str, list[str]] = {k: [] for k in targets}
    counts: dict[str, int] = {k: 0 for k in targets}

    for g in shuffled:
        # 分给"相对目标缺口最大"的 split（贪心平衡 image 比例）
        best_key = min(targets, key=lambda k: counts[k] / targets[k])
        assigned[best_key].append(g["group_id"])
        counts[best_key] += g["n_images"]

    return assigned, counts


def build_inner_splits(samples: list[dict[str, Any]], fold: int, seed: int) -> dict[str, Any]:
    outer_train = [s for s in samples if s["fold"] != fold]
    by_group: dict[str, dict[str, Any]] = {}
    for s in outer_train:
        gid = s["group_id"]
        if gid not in by_group:
            by_group[gid] = {"group_id": gid, "group_rule": s["group_rule"],
                             "n_images": 0, "image_ids": []}
        by_group[gid]["n_images"] += 1
        by_group[gid]["image_ids"].append(s["image_id"])

    groups = list(by_group.values())
    assigned, counts = _greedy_assign(groups, seed)

    # 粗类覆盖检查
    def coarse_coverage(gids: list[str]) -> set[str]:
        return {_coarse_of_rule(by_group[g]["group_rule"]) for g in gids}

    coverage = {k: sorted(coarse_coverage(v)) for k, v in assigned.items()}

    splits = {}
    for key in ("inner_model_train", "inner_error_mining", "inner_score_calib"):
        gids = sorted(assigned[key])
        image_ids = sorted(i for g in gids for i in by_group[g]["image_ids"])
        splits[key] = {
            "group_ids": gids,
            "n_groups": len(gids),
            "n_images": len(image_ids),
            "coarse_covered": coverage[key],
        }

    total = sum(g["n_images"] for g in groups)
    return {
        "outer_fold": fold,
        "seed": seed,
        "split_rule": "source_group_level_greedy_image_70_15_15",
        "n_outer_train_images": total,
        "n_outer_train_groups": len(groups),
        "image_ratio": {k: round(v / total, 4) for k, v in counts.items()},
        "coarse_coverage": coverage,
        "splits": splits,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path,
                        default=Path("data/splits/cv3_airport_proxy_k60_v2.json"))
    parser.add_argument("--seed", type=int, default=202625)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/N2-CFG-INNER-SPLITS"))
    args = parser.parse_args(argv)

    doc = json.loads(args.split.read_text(encoding="utf-8"))
    samples = doc["samples"]
    if not all(s["fold"] in (0, 1, 2) for s in samples):
        raise ValueError("samples 的 fold 必须是 0/1/2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "contract_version": "n2_inner_split_v1",
        "split_manifest": str(args.split.resolve()),
        "split_manifest_sha256": _sha256_text(args.split.read_text(encoding="utf-8")),
        "seed": args.seed,
        "folds": [],
    }
    for fold in (0, 1, 2):
        result = build_inner_splits(samples, fold, args.seed)
        # 粗类覆盖硬约束：model_train 必须三大类，mining/calib 必须三大类
        assert set(result["coarse_coverage"]["inner_model_train"]) == {"aircraft", "ship", "vehicle"}, \
            f"fold{fold} model_train 粗类覆盖不足: {result['coarse_coverage']}"
        for key in ("inner_error_mining", "inner_score_calib"):
            assert set(result["coarse_coverage"][key]) == {"aircraft", "ship", "vehicle"}, \
                f"fold{fold} {key} 粗类覆盖不足: {result['coarse_coverage']}"
        payload["folds"].append(result)
        print(f"fold{fold}: model={result['splits']['inner_model_train']['n_images']} "
              f"mining={result['splits']['inner_error_mining']['n_images']} "
              f"calib={result['splits']['inner_score_calib']['n_images']} "
              f"ratio={result['image_ratio']}")

    out = args.output_dir / "inner_splits.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".sha256").write_text(
        hashlib.sha256(text.encode("utf-8")).hexdigest() + "  inner_splits.json\n",
        encoding="utf-8",
    )
    print(f"\n写入: {out}")
    print(f"SHA256: {hashlib.sha256(text.encode('utf-8')).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
