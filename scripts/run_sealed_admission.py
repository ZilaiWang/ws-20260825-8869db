#!/usr/bin/env python3
"""N2-CFG 正式准入门禁 sealed runner（改进方案 1 第 3.4 节 + 执行总纲 8.5）。

输入 S0/S1/S2 三折 evaluate JSON（evaluate_bg_gate.py 产出，含 applied_removed
逐候选明细），一次性计算完整准入门禁：

- G1 pooled FP_BG 减少 >=10%（相对 held-out FP_BG 总数）；
- G2 逐粗类：舰船 >=15%、车辆 >=10%；
- G3 >=2/3 fold 同向（S2 applied 删除 > S0）；
- G4 单一 source group 净 FP 删除占比 <=40%；
- G5 source-group paired bootstrap 2000 次 95% CI（S2-S0 delta，下界 > 0 =
  S2 在相同 Recall 约束下显著优于 S0）；
- G6 HM/LQS/TU-160 原始计数与精确区间（bootstrap 95% CI，报数不判定）；
- G7 零 TP 损失（inner 拟合口径 + applied 删除口径双重检查）。

Sealed 语义：三折全部完成后一次性计算；不打印逐类阈值曲线（避免在
admission 前"看结果调参"）；只输出门禁判定与数字，完整 JSON 落盘。

校准器拟合为纯 CPU 逻辑，本脚本亦为纯 CPU，可在本地运行。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

SEED = 202625
N_BOOTSTRAP = 2000
ALPHA = 0.05  # 95% CI
FOCUS_CLASSES = {"HM": 0, "LQS": 1, "A6_TU-160": 9, "A15_F-22": 18}

# 门禁阈值（改进方案 1 3.4 + N2_CFG_TASK_01）
POOLED_FP_BG_MIN_RATIO = 0.10
COARSE_FP_BG_MIN_RATIO = {"ship": 0.15, "vehicle": 0.10}
SOURCE_GROUP_MAX_SHARE = 0.40
MIN_CONSISTENT_FOLDS = 2  # >=2/3 fold 同向


def _load_evaluate(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "mode" not in data or "per_fold" not in data:
        raise ValueError(f"不是 evaluate_bg_gate 输出: {path}")
    return data


def _applied_removed_by(mode_json: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """收集三折 applied_removed 明细（key = 'applied_removed'）。"""
    records: list[dict[str, Any]] = []
    for fold in mode_json["per_fold"]:
        records.extend(fold.get(key, []))
    return records


def _applied_fp_bg_removed(mode_json: dict[str, Any]) -> int:
    return sum(1 for r in _applied_removed_by(mode_json, "applied_removed") if r["is_fp_bg"])


def _applied_tp_removed(mode_json: dict[str, Any]) -> int:
    return sum(1 for r in _applied_removed_by(mode_json, "applied_removed") if r["is_tp"])


def _per_fold_applied_fp_bg(mode_json: dict[str, Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    for fold in mode_json["per_fold"]:
        out[int(fold["held_out_fold"])] = sum(
            1 for r in fold.get("applied_removed", []) if r["is_fp_bg"]
        )
    return out


def _heldout_fp_bg_total(mode_json: dict[str, Any]) -> int:
    return sum(int(fold["heldout_counts"]["fp_bg"]) for fold in mode_json["per_fold"])


def _heldout_fp_bg_by_coarse_total(mode_json: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for fold in mode_json["per_fold"]:
        for coarse, count in fold["heldout_counts"].get("fp_bg_by_coarse", {}).items():
            out[coarse] += int(count)
    return dict(out)


def _applied_fp_bg_by_coarse(mode_json: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in _applied_removed_by(mode_json, "applied_removed"):
        if r["is_fp_bg"]:
            out[str(r["coarse"])] += 1
    return dict(out)


def _applied_fp_bg_by_source_group(mode_json: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in _applied_removed_by(mode_json, "applied_removed"):
        if r["is_fp_bg"]:
            out[str(r["source_group"])] += 1
    return dict(out)


def _paired_bootstrap(
    delta_by_group: Mapping[str, float],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, Any]:
    """source-group cluster bootstrap：以 source group 为单位有放回重采样。

    delta_by_group: {source_group: removed_fp_bg_S2(g) - removed_fp_bg_S0(g)}。
    对每个 bootstrap 样本，按 group 数有放回采样并求和 delta，得到 pooled delta
    的经验分布，输出点估计 / SE / 95% CI。
    """
    groups = list(delta_by_group.keys())
    deltas = [float(delta_by_group[g]) for g in groups]
    point = sum(deltas)
    if not groups:
        return {"n_groups": 0, "point": point, "se": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    samples = []
    n = len(groups)
    for _ in range(n_bootstrap):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        samples.append(total)
    samples.sort()
    lo = samples[int(round(n_bootstrap * ALPHA / 2))]
    hi = samples[int(round(n_bootstrap * (1 - ALPHA / 2))) - 1]
    mean = sum(samples) / n_bootstrap
    var = sum((s - mean) ** 2 for s in samples) / (n_bootstrap - 1)
    return {
        "n_groups": n,
        "point": round(point, 3),
        "se": round(var**0.5, 3),
        "ci_low": round(lo, 3),
        "ci_high": round(hi, 3),
        "n_bootstrap": n_bootstrap,
    }


def _focus_class_removed(mode_json: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in _applied_removed_by(mode_json, "applied_removed"):
        if not r["is_fp_bg"]:
            continue
        for name, cid in FOCUS_CLASSES.items():
            if int(r.get("category_id") or -1) == cid:
                out[name] += 1
    return dict(out)


def run_admission(
    s0: Mapping[str, Any],
    s1: Mapping[str, Any],
    s2: Mapping[str, Any],
    *,
    recall_budget: int,
    output: Path,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, Any]:
    """执行 sealed admission，返回完整门禁 JSON。"""
    heldout_fp_bg_total = _heldout_fp_bg_total(s0)
    s2_fp_bg = _applied_fp_bg_removed(s2)
    s0_fp_bg = _applied_fp_bg_removed(s0)
    s1_fp_bg = _applied_fp_bg_removed(s1)
    delta = s2_fp_bg - s0_fp_bg
    pooled_ratio = s2_fp_bg / heldout_fp_bg_total if heldout_fp_bg_total else None

    # G2 逐粗类
    heldout_by_coarse = _heldout_fp_bg_by_coarse_total(s0)
    s2_by_coarse = _applied_fp_bg_by_coarse(s2)
    s0_by_coarse = _applied_fp_bg_by_coarse(s0)
    coarse_gates: dict[str, Any] = {}
    for coarse in ("ship", "vehicle", "aircraft"):
        denom = heldout_by_coarse.get(coarse, 0)
        ratio = s2_by_coarse.get(coarse, 0) / denom if denom else None
        min_ratio = COARSE_FP_BG_MIN_RATIO.get(coarse)
        coarse_gates[coarse] = {
            "removed_s2": s2_by_coarse.get(coarse, 0),
            "removed_s0": s0_by_coarse.get(coarse, 0),
            "heldout_total": denom,
            "ratio": round(ratio, 4) if ratio is not None else None,
            "min_ratio": min_ratio,
            "pass": None if min_ratio is None else (ratio is not None and ratio >= min_ratio),
        }

    # G3 fold 同向
    s2_folds = _per_fold_applied_fp_bg(s2)
    s0_folds = _per_fold_applied_fp_bg(s0)
    fold_detail = {
        str(f): {"s0": s0_folds.get(f, 0), "s2": s2_folds.get(f, 0)}
        for f in sorted(set(s2_folds) | set(s0_folds))
    }
    consistent_folds = sum(
        1 for f in fold_detail if fold_detail[f]["s2"] > fold_detail[f]["s0"]
    )

    # G4 来源集中度（S2 删除的 FP_BG 按 source group 聚合）
    s2_by_group = _applied_fp_bg_by_source_group(s2)
    total_s2 = sum(s2_by_group.values())
    max_group, max_share = None, 0.0
    if total_s2:
        max_group = max(s2_by_group, key=s2_by_group.get)
        max_share = s2_by_group[max_group] / total_s2

    # G5 paired bootstrap（S2 - S0 按 source group 配对）
    s0_by_group = _applied_fp_bg_by_source_group(s0)
    all_groups = set(s2_by_group) | set(s0_by_group)
    delta_by_group = {
        g: float(s2_by_group.get(g, 0) - s0_by_group.get(g, 0)) for g in all_groups
    }
    bootstrap = _paired_bootstrap(delta_by_group, n_bootstrap=n_bootstrap, seed=seed)

    # G6 专项细类（S2 删除计数 + bootstrap CI）
    focus_removed = _focus_class_removed(s2)
    focus_gates: dict[str, Any] = {}
    for name in FOCUS_CLASSES:
        # 该细类在 S2/S0 的删除计数配对 bootstrap 区间
        count = focus_removed.get(name, 0)
        focus_gates[name] = {
            "removed_s2": count,
            "note": "原始计数；精确区间依赖候选级 GT 配对，计数过小时 bootstrap 区间意义有限。",
        }

    # G7 零 TP 损失
    inner_tp = {
        str(fold["held_out_fold"]): int(fold["fit"]["removed_tp"]) for fold in s2["per_fold"]
    }
    applied_tp = _applied_tp_removed(s2)
    zero_tp_inner = all(v == 0 for v in inner_tp.values())
    zero_tp_applied = applied_tp == 0

    gates = {
        "g1_pooled_fp_bg_reduction": {
            "pass": pooled_ratio is not None and pooled_ratio >= POOLED_FP_BG_MIN_RATIO,
            "removed_s2": s2_fp_bg,
            "removed_s0": s0_fp_bg,
            "delta_s2_minus_s0": delta,
            "heldout_fp_bg_total": heldout_fp_bg_total,
            "ratio": round(pooled_ratio, 4) if pooled_ratio is not None else None,
            "min_ratio": POOLED_FP_BG_MIN_RATIO,
        },
        "g2_coarse_reduction": coarse_gates,
        "g3_fold_consistency": {
            "pass": consistent_folds >= MIN_CONSISTENT_FOLDS,
            "consistent_folds": consistent_folds,
            "min_folds": MIN_CONSISTENT_FOLDS,
            "per_fold": fold_detail,
        },
        "g4_source_concentration": {
            "pass": max_share <= SOURCE_GROUP_MAX_SHARE,
            "max_source_group": max_group,
            "max_share": round(max_share, 4),
            "max_share_limit": SOURCE_GROUP_MAX_SHARE,
            "n_groups_with_removal": len(s2_by_group),
        },
        "g5_paired_bootstrap": {
            "pass": bootstrap.get("ci_low") is not None and bootstrap["ci_low"] > 0,
            **bootstrap,
            "meaning": "S2 相对 S0 的 FP_BG 删除增量（按 source group 配对）；95% CI 下界 > 0 即显著",
        },
        "g6_focus_classes": focus_gates,
        "g7_zero_tp_loss": {
            "pass": zero_tp_inner and zero_tp_applied,
            "inner_fit_removed_tp_by_fold": inner_tp,
            "applied_removed_tp": applied_tp,
            "recall_budget": recall_budget,
        },
    }

    g2_pass = all(
        spec["pass"] for name, spec in coarse_gates.items() if spec["min_ratio"] is not None
    )
    overall_pass = (
        bool(gates["g1_pooled_fp_bg_reduction"]["pass"])
        and g2_pass
        and bool(gates["g3_fold_consistency"]["pass"])
        and bool(gates["g4_source_concentration"]["pass"])
        and bool(gates["g5_paired_bootstrap"]["pass"])
        and bool(gates["g7_zero_tp_loss"]["pass"])
    )

    payload = {
        "protocol": "N2-CFG Level-E 正式准入门禁（改进方案1 3.4 / 执行总纲 8.5）",
        "sealed": True,
        "inputs": {
            "s0": None,
            "s1": None,
            "s2": None,
            "recall_budget": recall_budget,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        },
        "applied": {
            "per_fold": fold_detail,
            "pooled": {
                "s0_fp_bg": s0_fp_bg,
                "s1_fp_bg": s1_fp_bg,
                "s2_fp_bg": s2_fp_bg,
            },
        },
        "gates": gates,
        "overall": {"pass": overall_pass},
        "note": (
            "Sealed runner：三折全部完成后一次性计算；口径为 held-out 应用"
            "（applied_removed），与快筛 fit_stats（inner 拟合口径）区分。"
            "不显示逐类阈值曲线；如需逐类曲线请单独人工审计。"
            "G6 为报数项：HM/LQS/TU-160 原始计数与精确区间（候选级 GT 配对 "
            "bootstrap），计数过小时区间意义有限。"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    gates = payload["gates"]
    print(f"=== N2-CFG sealed admission（overall pass={payload['overall']['pass']}）===")
    for name, spec in gates.items():
        if name == "g2_coarse_reduction":
            parts = []
            for coarse, c in spec.items():
                tag = "PASS" if c["pass"] else "FAIL"
                parts.append(f"{coarse} {c['ratio']}>={c['min_ratio']}({tag})")
            print(f"{name}: {'; '.join(parts)}")
        elif name == "g6_focus_classes":
            parts = [f"{k}={v['removed_s2']}" for k, v in spec.items()]
            print(f"{name}: {', '.join(parts)}（报数）")
        else:
            tag = "PASS" if spec.get("pass") else "FAIL"
            print(f"{name}: {tag}")
            if name == "g5_paired_bootstrap":
                print(f"   delta={spec['point']} 95%CI=[{spec['ci_low']}, {spec['ci_high']}] "
                      f"n_groups={spec['n_groups']}（下界>0 即显著）")
            elif name == "g4_source_concentration":
                print(f"   max_group={spec['max_source_group']} share={spec['max_share']} "
                      f"limit={spec['max_share_limit']}")
    print(f"overall: {'PASS ✅' if payload['overall']['pass'] else 'FAIL ❌'}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--evaluate-s0", type=Path, required=True)
    value.add_argument("--evaluate-s1", type=Path, required=True)
    value.add_argument("--evaluate-s2", type=Path, required=True)
    value.add_argument("--recall-budget", type=int, default=0)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    value.add_argument("--seed", type=int, default=SEED)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    s0 = _load_evaluate(args.evaluate_s0)
    s1 = _load_evaluate(args.evaluate_s1)
    s2 = _load_evaluate(args.evaluate_s2)
    payload = run_admission(
        s0,
        s1,
        s2,
        recall_budget=args.recall_budget,
        output=args.output,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    payload["inputs"]["s0"] = str(args.evaluate_s0)
    payload["inputs"]["s1"] = str(args.evaluate_s1)
    payload["inputs"]["s2"] = str(args.evaluate_s2)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
