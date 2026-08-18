#!/usr/bin/env python3
"""创新实验 vs M1 基线统一对比汇总。

消费每个实验的 ``evaluate_<KEY>.json``（由 evaluate_experiment.py 产出，
run_innovation.sh 自动生成 Y3/Y4/Y5 的），输出：

- ``<output>.json``：结构化对比（pooled / macro / per_fold / error_breakdown /
  focus_classes 全部并列），供自动化消费；
- ``<output>.md``：人类可读对比表（Markdown），供报告直接引用。

纯 CPU。用法：:

    python scripts/compare_innovations.py \
      --baseline /path/evaluate_M1.json \
      --innovation Y3=/path/evaluate_Y3-HIER.json \
      --innovation Y5=/path/evaluate_Y5-ROT90.json \
      --innovation Y4=/path/evaluate_Y4-AFSS.json \
      --output outputs/innovation-comparison/comparison.json

三创新串行跑完后，从服务器拉回三个 evaluate_*.json 即可一键出对比。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

METRIC_LABELS = {
    "recall": "Recall",
    "fdr": "FDR",
    "tp": "TP",
    "fp": "FP",
    "fn": "FN",
}

ERROR_LABELS = {
    "FP_BG": "FP_BG(背景误检)",
    "FP_CLS": "FP_CLS(类别误检)",
    "FP_DUP": "FP_DUP(重复框)",
    "FP_LOC": "FP_LOC(定位误检)",
    "FN_MISS": "FN_MISS(漏检)",
    "FN_CLS": "FN_CLS(类别漏检)",
    "FN_LOC": "FN_LOC(定位漏检)",
}


def _load_eval(path: Path, key: str) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("model_key") not in (key, None):
        print(f"  [warn] {path.name}: model_key={doc.get('model_key')} != {key}")
    return doc


def _fmt(value: Any, nd: int = 4) -> str:
    return f"{value:.{nd}f}" if isinstance(value, float) else str(value)


def _delta(innov: float, base: float) -> str:
    d = innov - base
    sign = "+" if d > 0 else ""
    return f"{sign}{d:+.4f}"


def build_comparison(
    baseline_doc: dict[str, Any],
    innovations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def add_row(metric: str, label: str, getter) -> None:
        base = getter(baseline_doc)
        row: dict[str, Any] = {"metric": metric, "label": label, "M1": base}
        for key, doc in innovations.items():
            row[key] = getter(doc)
        rows.append(row)

    # pooled
    add_row("pooled_recall", "Pooled Recall", lambda d: d["pooled"]["recall"])
    add_row("pooled_fdr", "Pooled FDR", lambda d: d["pooled"]["fdr"])
    add_row("macro_recall", "Macro Recall(V1.6)", lambda d: d["macro_recall"])
    add_row("macro_fdr", "Macro FDR(V1.6)", lambda d: d["macro_fdr"])

    # per-fold
    for fold in range(3):
        add_row(f"fold{fold}_recall", f"Fold{fold} Recall",
                lambda d, f=fold: d["per_fold"][str(f)]["recall"])
        add_row(f"fold{fold}_fdr", f"Fold{fold} FDR",
                lambda d, f=fold: d["per_fold"][str(f)]["fdr"])

    # error breakdown
    fp = baseline_doc["error_breakdown"]["fp"]
    fn = baseline_doc["error_breakdown"]["fn"]
    fp_keys = list(fp.keys())
    fn_keys = list(fn.keys())
    for k in fp_keys:
        add_row(f"fp_{k}", ERROR_LABELS.get(k, k),
                lambda d, k=k: d["error_breakdown"]["fp"].get(k, 0))
    for k in fn_keys:
        add_row(f"fn_{k}", ERROR_LABELS.get(k, k),
                lambda d, k=k: d["error_breakdown"]["fn"].get(k, 0))

    # focus classes(材料 19 点名专项)
    if "focus_classes" in baseline_doc:
        focus = baseline_doc["focus_classes"]
        for cls, stats in focus.items():
            for stat_key in ("tp", "fp", "fn", "recall", "fdr"):
                add_row(f"focus_{cls}_{stat_key}", f"{cls} {stat_key}",
                        lambda d, cls=cls, stat_key=stat_key:
                        d.get("focus_classes", {}).get(cls, {}).get(stat_key, 0))

    # GT 尺寸分布
    if "gt_size_distribution" in baseline_doc:
        for size_key, val in baseline_doc["gt_size_distribution"].items():
            if isinstance(val, int):
                add_row(f"gt_size_{size_key}", f"GT 尺寸 {size_key}",
                        lambda d, size_key=size_key: d.get("gt_size_distribution", {}).get(size_key, 0))

    return {"rows": rows, "models": ["M1", *innovations.keys()]}


def render_markdown(comparison: dict[str, Any]) -> str:
    lines = ["# 创新实验 vs M1 基线对比", ""]
    models = comparison["models"]
    innov_names = models[1:]
    header = "| 指标 | " + " | ".join(models) + " | " + \
             " | ".join(f"Δ({name}-M1)" for name in innov_names) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(models) + len(innov_names) + 1))
    for row in comparison["rows"]:
        base = row.get("M1")
        vals = [_fmt(row.get(m)) for m in models]
        deltas: list[str] = []
        for name in innov_names:
            val = row.get(name)
            if isinstance(base, (int, float)) and isinstance(val, (int, float)):
                deltas.append(_delta(val, base))
            else:
                deltas.append("-")
        lines.append(f"| {row['label']} | " + " | ".join(vals) + " | " + " | ".join(deltas) + " |")
    lines.append("")
    lines.append("> Δ 列：该创新相对 M1 的变化（+ 为更高/更多）。FDR 越低越好；Recall 越高越好；FP_*/FN_* 计数越低越好。")
    lines.append("")
    lines.append("## 判读要点（根据结果对照）")
    lines.append("")
    lines.append("- **Y3-HIER（层次损失）**：预期 FP_CLS 下降；若 Recall 下降过多说明 coarse 权重过大。")
    lines.append("- **Y5-ROT90（旋转增强）**：预期车辆 Recall 提升（车辆常旋转排布）；观察是否伤其他类。")
    lines.append("- **Y4-AFSS（充分度采样）**：预期低充分度图上的 FP_BG/FN_MISS 改善；观察最低 10% 图错误占比。")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="M1 基线 evaluate_M1.json")
    parser.add_argument("--innovation", action="append", default=[],
                        help="KEY=path，可多次；如 Y3=/path/evaluate_Y3-HIER.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        raise FileNotFoundError(args.baseline)
    innovations: dict[str, dict[str, Any]] = {}
    for spec in args.innovation:
        key, _, path_str = spec.partition("=")
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(path)
        innovations[key] = _load_eval(path, key)
    if not innovations:
        raise ValueError("至少提供一个 --innovation")

    baseline = _load_eval(args.baseline, "M1")
    comparison = build_comparison(baseline, innovations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(render_markdown(comparison), encoding="utf-8")
    print(f"comparison -> {args.output}")
    print(f"markdown   -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
