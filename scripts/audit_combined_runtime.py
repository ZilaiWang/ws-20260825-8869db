#!/usr/bin/env python3
"""E 组合系统(combined)10K 时延审计:预算加总口径。

组合系统 = M1 + M3 串行流水线(先 M1 推理, 再 M3 推理, 融合)。
「组合 after-read 预算」= M1 after-read + M3 after-read(共享同一张 10K 图,
两模型分别 read/tile/infer/fuse)。因此组合审计 = 对同一图的两个模型
after-read 逐 run 相加, 再取 p50/p95/max。

输入: M1 与 M3 各自的 capture/runtime_samples.jsonl(由 run_e_benchmark.sh
分别产出, 每文件含 measured runs)。两文件 image 集合一致(同一 10K 图集)。

产出:
- p50/p95/max 组合 after-read(逐图相加), 门禁 <=20s
- 组合 wall 预算 = M1 wall + M3 wall
- 各 run 明细(图 id, m1_after_read, m3_after_read, combined)

纯 CPU。用法::

    python scripts/audit_combined_runtime.py \
      --m1-samples /path/M1/capture/runtime_samples.jsonl \
      --m3-samples /path/M3/capture/runtime_samples.jsonl \
      --output /path/combined/audit_combined.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

MAX_COMBINED_AFTER_READ_SECONDS = 20.0


def _load_samples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _as_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _after_read(sample: dict[str, Any]) -> float:
    if "total_after_read" in sample:
        return _as_seconds(sample["total_after_read"])
    for key in ("after_read_seconds", "after_read", "phase_after_read_seconds"):
        if key in sample:
            return _as_seconds(sample[key])
    if "phases" in sample and isinstance(sample["phases"], dict):
        # 兜底: after_read = image_read 之后所有 phase 之和
        excluded = {"image_read", "read"}
        return sum(_as_seconds(v) for k, v in sample["phases"].items()
                   if k not in excluded and isinstance(v, (int, float)))
    raise KeyError(f"无法从 sample 提取 after_read: {list(sample.keys())[:8]}")


def _wall(sample: dict[str, Any]) -> float:
    for key in ("wall_seconds", "total_seconds", "wall"):
        if key in sample:
            return _as_seconds(sample[key])
    # wall = after_read + image_read
    if "phases" in sample and isinstance(sample["phases"], dict):
        return _after_read(sample) + _as_seconds(sample["phases"].get("image_read", 0.0))
    return _after_read(sample)


def _image_id(sample: dict[str, Any]) -> str:
    for key in ("image_id", "image", "path", "sample_id"):
        if key in sample:
            return str(sample[key])
    return "unknown"


def audit_combined(m1_path: Path, m3_path: Path) -> dict[str, Any]:
    m1 = _load_samples(m1_path)
    m3 = _load_samples(m3_path)
    if not m1 or not m3:
        raise ValueError("M1/M3 runtime_samples 为空")

    m1_by_image = {_image_id(s): s for s in m1}
    m3_by_image = {_image_id(s): s for s in m3}
    common = sorted(set(m1_by_image) & set(m3_by_image))
    if not common:
        raise ValueError("M1 与 M3 无共同图(不同图集, 无法组合)")

    combined_rows = []
    for img in common:
        a = _after_read(m1_by_image[img])
        b = _after_read(m3_by_image[img])
        combined_rows.append(
            {
                "image_id": img,
                "m1_after_read_seconds": round(a, 4),
                "m3_after_read_seconds": round(b, 4),
                "combined_after_read_seconds": round(a + b, 4),
                "m1_wall_seconds": round(_wall(m1_by_image[img]), 4),
                "m3_wall_seconds": round(_wall(m3_by_image[img]), 4),
                "combined_wall_seconds": round(_wall(m1_by_image[img]) + _wall(m3_by_image[img]), 4),
            }
        )

    combined_after = [r["combined_after_read_seconds"] for r in combined_rows]
    combined_wall = [r["combined_wall_seconds"] for r in combined_rows]
    p50 = statistics.median(combined_after)
    p95 = sorted(combined_after)[min(len(combined_after) - 1, int(round(0.95 * len(combined_after))) - 1)]
    result = {
        "contract_version": "runtime_10k_benchmark_v1",
        "combination": "M1 + M3 serial budget addition",
        "measured_runs": len(combined_rows),
        "combined_after_read_p50": round(p50, 4),
        "combined_after_read_p95": round(p95, 4),
        "combined_after_read_max": round(max(combined_after), 4),
        "combined_wall_p50": round(statistics.median(combined_wall), 4),
        "combined_after_read_passed_max_20s": bool(max(combined_after) <= MAX_COMBINED_AFTER_READ_SECONDS),
        "combined_after_read_p50_passed_20s": bool(p50 <= MAX_COMBINED_AFTER_READ_SECONDS),
        "maximum_combined_after_read_seconds": MAX_COMBINED_AFTER_READ_SECONDS,
        "m1_after_read_p50": round(statistics.median([r["m1_after_read_seconds"] for r in combined_rows]), 4),
        "m3_after_read_p50": round(statistics.median([r["m3_after_read_seconds"] for r in combined_rows]), 4),
        "per_run": combined_rows,
        "input_sha256": {
            "m1_samples": _sha256(m1_path),
            "m3_samples": _sha256(m3_path),
        },
    }
    return result


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-samples", type=Path, required=True)
    parser.add_argument("--m3-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    result = audit_combined(args.m1_samples, args.m3_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    passed = bool(result["combined_after_read_p50_passed_20s"])
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
