#!/usr/bin/env python3
"""N2-CFG Level-E 快筛评估：S0 / S1 / S2。

三消融（《改进方案 1》第 3.2 节）：

- S0：仅 R1-6 score 的单调校准器（排除"只是重新调阈值"）；
- S1：shared foreground head + score；
- S2：coarse-conditioned head + score（主候选）。

流程为 leave-one-fold-out cross-fit：对每个 held-out fold，用另外两折
（inner_score_calib 角色）拟合低容量校准器 ``(alpha, beta, gamma_c, tau_drop)``，
只应用一次到 held-out fold，评估 FP_BG 删除与 Recall 变化。held-out fold
不参与任何参数选择。

本脚本的模型推理依赖 GPU；校准器拟合与指标计算为纯 CPU 逻辑，可本地单测。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsdet.analysis.background_gate import (
    COARSE_CLASSES,
    MODE_S0,
    MODE_S1,
    MODE_S2,
    GateCalibration,
    apply_gate,
    calibrate_q,
    coarse_of_category_id,
    make_calibration,
)


@dataclass(frozen=True)
class FitSample:
    """一个用于校准拟合的带标签候选。"""

    proposal_uid: str
    score: float
    fg_logit: float
    coarse: str
    is_tp: bool
    is_fp_bg: bool
    image_id: int | None = None
    category_id: int | None = None
    source_group: str = ""


def _gamma_grid(mode: str) -> list[Mapping[str, float]]:
    """按模式给出 gamma 网格（S0/S1 单一值，S2 舰船/车辆独立）。"""
    values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    if mode in {MODE_S0, MODE_S1}:
        return [{name: value for name in COARSE_CLASSES} for value in values]
    grids: list[Mapping[str, float]] = []
    for ship in values:
        for vehicle in values:
            grids.append({"ship": ship, "aircraft": 0.0, "vehicle": vehicle})
    return grids


def _beta_grid(mode: str) -> list[float]:
    if mode == MODE_S0:
        return [0.0]
    return [0.5, 1.0, 2.0]


def _alpha_grid(mode: str) -> list[float]:
    return [1.0]


def _best_tau_drop(
    samples: Sequence[FitSample],
    calibration_params: Mapping[str, Any],
    recall_budget: int,
) -> tuple[float, int, int]:
    """在给定 (alpha, beta, gamma) 下扫描 tau_drop，最大化删除 FP_BG。

    按 q 升序（低 q 先删）前缀扫描，删除的 TP 数不超过 recall_budget 时，
    记录可删除的最大 FP_BG 数。
    """
    alpha = float(calibration_params["alpha"])
    beta = float(calibration_params["beta"])
    gamma = dict(calibration_params["gamma"])
    scores: list[tuple[float, bool, bool]] = []
    for sample in samples:
        calib = make_calibration(
            mode=calibration_params["mode"],
            alpha=alpha,
            beta=beta,
            tau_drop=0.0,
            gamma=gamma,
        )
        q = calibrate_q(sample.score, sample.fg_logit, sample.coarse, calib)
        scores.append((q, sample.is_tp, sample.is_fp_bg))
    scores.sort(key=lambda item: item[0])  # 升序，低 q 在前

    best_fp_bg = 0
    best_tau = 0.0
    best_removed_tp = 0
    removed_tp = 0
    removed_fp_bg = 0
    for index, (q, is_tp, is_fp_bg) in enumerate(scores):
        if removed_tp + int(is_tp) > recall_budget:
            break
        removed_tp += int(is_tp)
        removed_fp_bg += int(is_fp_bg)
        if removed_fp_bg > best_fp_bg:
            best_fp_bg = removed_fp_bg
            best_removed_tp = removed_tp
            # tau_drop 取当前 q 与下一个 q 之间的中点，或当前 q。
            next_q = scores[index + 1][0] if index + 1 < len(scores) else 1.0
            best_tau = (q + next_q) / 2.0 if next_q > q else min(q + 1e-6, 1.0)
    return best_tau, best_fp_bg, best_removed_tp


def fit_gate_calibration(
    samples: Sequence[FitSample],
    *,
    mode: str,
    recall_budget: int,
) -> tuple[GateCalibration, dict[str, Any]]:
    """拟合低容量校准器，返回 (校准器, 拟合统计)。"""
    if mode not in {MODE_S0, MODE_S1, MODE_S2}:
        raise ValueError(f"未知 mode={mode!r}")
    best: tuple[int, int, GateCalibration] | None = None  # (fp_bg, -tp, calib)
    best_stats: dict[str, Any] = {}
    for alpha in _alpha_grid(mode):
        for beta in _beta_grid(mode):
            for gamma in _gamma_grid(mode):
                params = {"mode": mode, "alpha": alpha, "beta": beta, "gamma": gamma}
                tau, fp_bg, tp = _best_tau_drop(samples, params, recall_budget)
                calib = make_calibration(
                    mode=mode,
                    alpha=alpha,
                    beta=beta,
                    tau_drop=tau,
                    gamma=gamma,
                )
                rank = (fp_bg, -tp)
                if best is None or rank > (best[0], best[1]):
                    best = (fp_bg, tp, calib)
                    best_stats = {
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": dict(gamma),
                        "tau_drop": tau,
                        "removed_fp_bg": fp_bg,
                        "removed_tp": tp,
                    }
    assert best is not None
    return best[2], best_stats


def apply_gate_to_predictions(
    predictions: Mapping[int, list[dict[str, Any]]],
    fg_logits: Mapping[str, float],
    calibration: GateCalibration,
) -> tuple[dict[int, list[dict[str, Any]]], list[Mapping[str, Any]]]:
    """把门控应用到 R1-6 候选，返回 (保留预测, 被删除记录)。"""
    candidates: list[Mapping[str, Any]] = []
    for image_id, records in predictions.items():
        for prediction_index, record in enumerate(records):
            candidate = dict(record)
            candidate["_image_id"] = image_id
            candidate["_prediction_index"] = prediction_index
            if "proposal_uid" not in candidate:
                candidate["proposal_uid"] = f"r1-i{image_id}-p{prediction_index}"
            candidate["category_id"] = int(candidate["category_id"])
            candidate["score"] = float(candidate["score"])
            candidates.append(candidate)

    kept, removed = apply_gate(candidates, fg_logits, calibration)
    kept_predictions: dict[int, list[dict[str, Any]]] = {}
    for candidate in kept:
        image_id = int(candidate["_image_id"])
        record = {
            "image_id": image_id,
            "category_id": int(candidate["category_id"]),
            "score": float(candidate["score"]),
            "bbox_xyxy": list(candidate["bbox_xyxy"]),
            "proposal_uid": str(candidate["proposal_uid"]),
        }
        kept_predictions.setdefault(image_id, []).append(record)
    removed_records = [
        {
            "proposal_uid": str(candidate["proposal_uid"]),
            "image_id": int(candidate["_image_id"]),
            "coarse": str(candidate.get("_gate_coarse", "")),
            "score": float(candidate["score"]),
            "q": float(candidate.get("_gate_q", 0.0)),
        }
        for candidate in removed
    ]
    return kept_predictions, removed_records


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--predictions", type=Path, required=True)
    value.add_argument("--fg-logits", type=Path, required=True, help="proposal_uid -> fg_logit JSON")
    value.add_argument("--formal-crop-manifest", type=Path, required=True)
    value.add_argument("--image-ledger", type=Path, required=True)
    value.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--recall-budget", type=int, default=0, help="每折允许删除的最大 TP 数")
    value.add_argument("--mode", choices=(MODE_S0, MODE_S1, MODE_S2), required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)

    from rsdet.analysis.background_gate_manifest import (
        _load_image_ledger,
        _load_selected_predictions,
    )
    from rsdet.analysis.oof_detection import (
        decompose_official_errors,
        load_formal_ground_truth,
    )
    from rsdet.evaluation.official_metric import evaluate_predictions_with_trace
    from rsdet.evaluation.protocol import parse_evaluation_protocol
    from rsdet.utils.config import load_config

    predictions, _ = _load_selected_predictions(args.predictions)
    ledger = _load_image_ledger(args.image_ledger)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    formal = load_formal_ground_truth(
        args.formal_crop_manifest,
        expected_images=4481,
        expected_annotations=20933,
    )
    fg_logits = json.loads(args.fg_logits.read_text(encoding="utf-8"))

    _, trace = evaluate_predictions_with_trace(
        formal.boxes,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    tp_keys = {(match.image_id, match.prediction_index) for match in trace.matches}
    _, cases, _ = decompose_official_errors(
        formal,
        predictions,
        threshold=0.0,
        protocol=protocol,
        model_key="R1_6_FINAL_CHAIN",
        include_cases=True,
    )
    fp_bg_keys = {
        (int(case["image_id"]), int(str(case["item_uid"]).rsplit("-p", 1)[1]))
        for case in cases
        if case["case_side"] == "prediction" and case["reason"] == "FP_BG"
    }

    samples: list[FitSample] = []
    for image_id, records in predictions.items():
        for prediction_index, record in enumerate(records):
            proposal_uid = str(record.get("proposal_uid") or f"r1-i{image_id}-p{prediction_index}")
            coarse = coarse_of_category_id(int(record["category_id"]))
            ledger_entry = ledger.get(int(image_id)) or {}
            samples.append(
                FitSample(
                    proposal_uid=proposal_uid,
                    score=float(record["score"]),
                    fg_logit=_extract_fg_logit(
                        fg_logits.get(proposal_uid, {}), args.mode, coarse
                    ),
                    coarse=coarse,
                    is_tp=(image_id, prediction_index) in tp_keys,
                    is_fp_bg=(image_id, prediction_index) in fp_bg_keys,
                    image_id=int(image_id),
                    category_id=int(record["category_id"]),
                    source_group=str(ledger_entry.get("group_id", "")),
                )
            )

    # cross-fit：三折轮流 held-out。
    per_fold: list[dict[str, Any]] = []
    for held_out in (0, 1, 2):
        selection = [
            sample for sample in samples if _fold_of(sample, ledger) != held_out
        ]
        calibration, fit_stats = fit_gate_calibration(
            selection,
            mode=args.mode,
            recall_budget=args.recall_budget,
        )
        heldout_samples = [
            sample for sample in samples if _fold_of(sample, ledger) == held_out
        ]
        heldout_fp_bg = sum(1 for sample in heldout_samples if sample.is_fp_bg)
        heldout_tp = sum(1 for sample in heldout_samples if sample.is_tp)
        # held-out 应用拟合出的校准器：q < tau_drop 即被删除（OOF 泛化口径明细）。
        applied_removed: list[dict[str, Any]] = []
        for sample in sorted(heldout_samples, key=lambda s: s.score, reverse=True):
            q = calibrate_q(sample.score, sample.fg_logit, sample.coarse, calibration)
            if q < calibration.tau_drop:
                applied_removed.append(
                    {
                        "proposal_uid": sample.proposal_uid,
                        "image_id": sample.image_id,
                        "category_id": sample.category_id,
                        "coarse": sample.coarse,
                        "source_group": sample.source_group,
                        "score": sample.score,
                        "q": q,
                        "is_tp": sample.is_tp,
                        "is_fp_bg": sample.is_fp_bg,
                    }
                )
        heldout_fp_bg_by_coarse: dict[str, int] = {}
        heldout_tp_by_coarse: dict[str, int] = {}
        for coarse in COARSE_CLASSES:
            heldout_fp_bg_by_coarse[coarse] = sum(
                1 for sample in heldout_samples if sample.is_fp_bg and sample.coarse == coarse
            )
            heldout_tp_by_coarse[coarse] = sum(
                1 for sample in heldout_samples if sample.is_tp and sample.coarse == coarse
            )
        per_fold.append(
            {
                "held_out_fold": held_out,
                "fit": fit_stats,
                "heldout_counts": {
                    "fp_bg": heldout_fp_bg,
                    "tp": heldout_tp,
                    "fp_bg_by_coarse": heldout_fp_bg_by_coarse,
                    "tp_by_coarse": heldout_tp_by_coarse,
                },
                "applied_removed": applied_removed,
            }
        )

    result = {"mode": args.mode, "per_fold": per_fold}
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"evaluate_{args.mode}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _fold_of(sample: FitSample, ledger: Mapping[int, Mapping[str, Any]]) -> int:
    """从 proposal_uid 解析 fold（形如 ``m1-f{fold}-i{image}-p{idx}``）。"""
    for part in str(sample.proposal_uid).split("-"):
        if part.startswith("f") and part[1:].isdigit():
            return int(part[1:])
    raise ValueError(f"无法从 proposal_uid 解析 fold: {sample.proposal_uid}")


def _extract_fg_logit(entry: Mapping[str, Any], mode: str, coarse: str) -> float:
    """按模式从模型输出提取单一前景 logit。

    - S0：无前景证据（0.0）；
    - S1：shared head（``shared``）；
    - S2：coarse-conditioned head（``{coarse}`` = shared + residual）。
    """
    if mode == MODE_S0:
        return 0.0
    if not entry:
        return 0.0
    if mode == MODE_S1:
        return float(entry["shared"])
    if mode == MODE_S2:
        return float(entry[coarse])
    raise ValueError(f"未知 mode={mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
