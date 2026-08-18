#!/usr/bin/env python3
"""T4-GATED-INFER 决策框架：M3 门控推理准入评估（等待 gate4/gate5 数据后一键判定）。

总纲 5.4 停止条件（T4）：M3 门控推理仅当以下全部成立才保留进入最终系统：

1. 校准后额外净增 ≥30 TP（相对 M1 基线）；
2. Overall FDR ≤ 0.17（N2-CFG 门控后口径，gate4）；
3. 10K p95 ≤ 18 秒（组合时延，gate5）。

本脚本输入（待 N2-CFG / E 组合完成后的产物）：
- M1 与 M3 的候选预测（OOF，可选门控掩码）；
- gate4 数据：N2-CFG 后 FDR 结果（evaluate/admission JSON）；
- gate5 数据：E 组合 10K 测速（audit_combined.json 的 p95）。

输出：T4 准入判定 JSON + 报告（现在只做框架,无数据时输出 waiting）。

用法：:

    python scripts/evaluate_gated_infer.py \
      --m1 /path/evaluate_M1.json \
      --m3 /path/evaluate_M3.json \
      --gate4 /path/admission.json \
      --gate5 /path/audit_combined.json \
      --output /path/t4_decision.json

也可只给部分输入：缺失的 gate 记为 pending。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

GATE4_FDR_MAX = 0.17
GATE5_P95_MAX = 18.0
T4_MIN_TP_GAIN = 30


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _eval_recall_fdr(doc: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if doc is None or "pooled" not in doc:
        return None, None
    return float(doc["pooled"]["recall"]), float(doc["pooled"]["fdr"])


def _tp(doc: dict[str, Any] | None) -> int | None:
    if doc is None or "pooled" not in doc:
        return None
    return int(doc["pooled"]["tp"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-eval", type=Path, default=None, help="M1 evaluate JSON")
    parser.add_argument("--m3-eval", type=Path, default=None, help="M3 evaluate JSON")
    parser.add_argument("--gate4", type=Path, default=None, help="N2-CFG 后 admission/FDR JSON")
    parser.add_argument("--gate5", type=Path, default=None, help="E 组合 audit_combined JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    m1 = _load(args.m1_eval)
    m3 = _load(args.m3_eval)
    gate4_doc = _load(args.gate4)
    gate5_doc = _load(args.gate5)

    # TP 净增：M3 相对 M1（同一低阈值 OOF 口径；正式口径以校准后为准）
    m1_tp = _tp(m1)
    m3_tp = _tp(m3)
    tp_gain = (m3_tp - m1_tp) if (m1_tp is not None and m3_tp is not None) else None

    # gate4：N2-CFG 后 Overall FDR
    fdr_value: float | None = None
    if gate4_doc is not None:
        for key in ("overall_fdr", "fdr", "pooled_fdr"):
            if isinstance(gate4_doc, dict) and key in gate4_doc:
                fdr_value = float(gate4_doc[key])
                break
        if fdr_value is None and isinstance(gate4_doc, dict):
            for k, v in gate4_doc.items():
                if isinstance(v, dict) and "fdr" in v:
                    fdr_value = float(v["fdr"])
                    break

    # gate5：E 组合 10K p95
    p95: float | None = None
    if gate5_doc is not None:
        for key in ("combined_after_read_p95", "p95", "after_read_p95"):
            if key in gate5_doc:
                p95 = float(gate5_doc[key])
                break

    gates = {
        "tp_gain_ge_30": {
            "required": T4_MIN_TP_GAIN,
            "actual": tp_gain,
            "pass": (tp_gain >= T4_MIN_TP_GAIN) if tp_gain is not None else None,
        },
        "gate4_fdr_le_0.17": {
            "required": GATE4_FDR_MAX,
            "actual": fdr_value,
            "pass": (fdr_value <= GATE4_FDR_MAX) if fdr_value is not None else None,
        },
        "gate5_10k_p95_le_18": {
            "required": GATE5_P95_MAX,
            "actual": p95,
            "pass": (p95 <= GATE5_P95_MAX) if p95 is not None else None,
        },
    }
    decided = all(g["pass"] is not None for g in gates.values())
    pending = any(g["pass"] is None for g in gates.values())
    all_pass = decided and all(g["pass"] for g in gates.values())

    decision = {
        "experiment": "T4-GATED-INFER 决策",
        "status": "all_pass" if all_pass else ("pending" if pending else "not_admitted"),
        "gates": gates,
        "note": (
            "T4 仅当校准后净增TP>=30 且 N2-CFG后FDR<=0.17 且 10K p95<=18s 才保留；"
            "缺失数据时对应 gate 记 pending。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
