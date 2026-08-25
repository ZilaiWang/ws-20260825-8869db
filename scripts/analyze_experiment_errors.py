#!/usr/bin/env python3
"""实验错误诊断接口：定位"具体哪里出了问题"，供方法改进。

消费 evaluate_experiment.py 落盘的 `.cases.json`，结合 cv3 划分与 GT 尺寸，
按 类 / 尺寸 / fold / source-group 聚合错误，输出可读的诊断报告：

  - 每个错误类型（FP_BG/FP_CLS/FN_MISS/...）主要落在哪些细类；
  - 每个错误类型主要落在哪些尺寸档（tiny/small/medium/large）；
  - 每个错误类型的 fold 分布（折间是否一致）；
  - 每个错误类型贡献最多的 source group（来源稳健性：单一组是否过度集中）。

输出 `<output>.json` + 人类可读打印。纯 CPU。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


def _short_edge(box: list[float]) -> float:
    return min(box[2] - box[0], box[3] - box[1])


def _size_bin(short_edge: float) -> str:
    if short_edge < 32:
        return "tiny"
    if short_edge < 96:
        return "small"
    if short_edge < 256:
        return "medium"
    return "large"


def _parse_box(text: str) -> list[float]:
    return [float(v) for v in str(text).split()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--split", type=Path,
                        default=Path("data/splits/cv3_airport_proxy_k60_v2.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    doc = json.loads(args.split.read_text(encoding="utf-8"))
    split = {int(s["image_id"]): (int(s["fold"]), str(s["group_id"])) for s in doc["samples"]}

    by_reason_class: dict[str, Counter] = defaultdict(Counter)
    by_reason_size: dict[str, Counter] = defaultdict(Counter)
    by_reason_fold: dict[str, Counter] = defaultdict(Counter)
    by_reason_group: dict[str, Counter] = defaultdict(Counter)

    for c in cases:
        reason = c["reason"]
        image_id = int(c["image_id"])
        fold, group = split.get(image_id, (-1, "unknown"))
        box = _parse_box(c["bbox_xyxy"])
        size = _size_bin(_short_edge(box))
        by_reason_class[reason][c["class_name"]] += 1
        by_reason_size[reason][size] += 1
        by_reason_fold[reason][fold] += 1
        by_reason_group[reason][group] += 1

    report: dict[str, Any] = {
        "by_reason_class": {k: dict(sorted(v.items(), key=lambda x: -x[1]))
                            for k, v in sorted(by_reason_class.items())},
        "by_reason_size": {k: dict(sorted(v.items(), key=lambda x: -x[1]))
                           for k, v in sorted(by_reason_size.items())},
        "by_reason_fold": {k: dict(sorted(v.items())) for k, v in sorted(by_reason_fold.items())},
        # 来源稳健性：每个错误类型 top-5 source group（材料 18 门禁：单一组 <40%）
        "top_source_groups": {
            k: dict(v.most_common(5)) for k, v in sorted(by_reason_group.items())
        },
        "source_group_concentration": {
            k: round(v.most_common(1)[0][1] / sum(v.values()), 4) if v else None
            for k, v in sorted(by_reason_group.items())
        },
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 人类可读打印
    print("=== 错误类型 → 细类 Top5 ===")
    for reason, dist in sorted(report["by_reason_class"].items()):
        top = list(dist.items())[:5]
        print(f"  {reason}: {', '.join(f'{k}={v}' for k, v in top)}")
    print("\n=== 错误类型 → 尺寸档 ===")
    for reason, dist in sorted(report["by_reason_size"].items()):
        print(f"  {reason}: {dist}")
    print("\n=== 错误类型 → source-group 集中度（top1 占比）===")
    for reason, conc in sorted(report["source_group_concentration"].items()):
        print(f"  {reason}: {conc}")
    print(f"\n诊断写入: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
