#!/usr/bin/env python3
"""官方评分方案 V1.6 七项排名与二次排序 CLI。

用法：
    # 逐个传评估 JSON（每个必须是 scripts/evaluate.py --output 的产物，
    # 且含 official_ranking.seven_ranking_metrics_v1_6 块）
    PYTHONPATH=src python scripts/rank_official.py \
        --results outputs/M1/result.json --tag M1 \
        --results outputs/M3/result.json --tag M3

    # 或提供一个含全部队伍的汇总 JSON（列表形式）
    PYTHONPATH=src python scripts/rank_official.py --teams teams.json --output rank.json

teams.json 结构：
    [
      {"team_id": "M1", "ship_recall": 0.72, "ship_fdr": 0.52, ...,
       "latency_seconds": 8.1},
      ...
    ]

输出：每支队伍 7 项指标、7 项排名、排名和、二次排序名次/百分比、
官方三项打分区间；默认打印文本表，可用 --output 写 JSON。
"""

import argparse
import json
import math
import sys
from pathlib import Path

from rsdet.evaluation.official_ranking import (
    RankingItem,
    compute_official_rankings,
    format_ranking_table,
    ranking_result_to_dicts,
)
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="rank_official")

_SEVEN_KEYS = (
    "ship_recall",
    "ship_fdr",
    "aircraft_recall",
    "aircraft_fdr",
    "vehicle_recall",
    "vehicle_fdr",
    "latency_seconds",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="官方 V1.6 七项排名与二次排序")
    parser.add_argument(
        "--results",
        action="append",
        type=Path,
        default=[],
        help="scripts/evaluate.py --output 的评估 JSON（可多次传）",
    )
    parser.add_argument("--tag", action="append", default=[], help="对应 --results 的队名")
    parser.add_argument(
        "--teams",
        type=Path,
        default=None,
        help="包含全部队伍指标的汇总 JSON（与 --results 二选一）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="排名结果 JSON",
    )
    parser.add_argument(
        "--percentile-fraction-scale",
        type=float,
        default=0.5,
        help="二次排序百分比取法：0.5=position/n（默认），0.0=(position-1)/(n-1)",
    )
    return parser.parse_args(argv)


def _build_items(
    results: list[Path],
    tags: list[str],
    teams: Path | None,
) -> tuple[list[RankingItem], list[tuple[str, str]]]:
    """从评估 JSON 或队伍汇总 JSON 构造 RankingItem。"""
    if teams is not None:
        return _item_from_teams_file(teams)
    if len(results) != len(tags):
        raise ValueError("--results 与 --tag 数量必须一致")
    items: list[RankingItem] = []
    provenance: list[tuple[str, str]] = []
    for path, tag in zip(results, tags):
        item = _item_from_eval_json(path, tag)
        items.append(item)
        provenance.append((tag, str(path)))
    return items, provenance


def _item_from_eval_json(path: Path, tag: str) -> RankingItem:
    """从 evaluate.py 的 JSON 产物构造单队指标。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data.get("official_ranking", {}).get("seven_ranking_metrics_v1_6")
    if not isinstance(block, dict):
        raise ValueError(
            f"{path}: 缺少 official_ranking.seven_ranking_metrics_v1_6 块，"
            "请用 scripts/evaluate.py --output 生成"
        )
    return _item_from_mapping(tag, block, path)


def _item_from_teams_file(path: Path) -> tuple[list[RankingItem], list[tuple[str, str]]]:
    """从队伍汇总 JSON 构造全部指标。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: 必须是非空队伍列表")
    items: list[RankingItem] = []
    provenance: list[tuple[str, str]] = []
    for entry in data:
        team_id = entry.get("team_id")
        if not isinstance(team_id, str) or not team_id:
            raise ValueError(f"{path}: 每支队伍必须有非空 team_id")
        item = _item_from_mapping(team_id, entry, path)
        items.append(item)
        provenance.append((team_id, str(path)))
    return items, provenance


def _item_from_mapping(team_id: str, mapping: dict, source: Path) -> RankingItem:
    """从任意映射构造 RankingItem，缺失键按 None 处理。"""
    kwargs: dict[str, object] = {"team_id": team_id}
    for key in _SEVEN_KEYS:
        value = mapping.get(key)
        if value is not None:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{source}: {team_id} 的 {key} 不是有限数")
        kwargs[key] = value
    try:
        return RankingItem(**kwargs)
    except TypeError as error:
        raise ValueError(f"{source}: {team_id} 指标缺失 ({error})") from error


def main(argv: list[str] | None = None) -> int:
    """运行排名。"""
    args = parse_args(argv)
    try:
        items, _ = _build_items(args.results, args.tag, args.teams)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("输入无效: %s", error)
        return 1
    results = compute_official_rankings(
        items,
        percentile_fraction_scale=args.percentile_fraction_scale,
    )
    for line in format_ranking_table(results):
        logger.info(line)
    for result in results:
        if result.incomplete:
            logger.warning(
                "队伍 %s 缺时延项，未参与完整 7 项排名（只计 6 项）",
                result.team_id,
            )

    if args.output:
        output_data = {
            "contract": "official_ranking_v1_6",
            "metric_names": list(_SEVEN_KEYS),
            "percentile_fraction_scale": args.percentile_fraction_scale,
            "teams": ranking_result_to_dicts(results),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("排名结果已保存: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
