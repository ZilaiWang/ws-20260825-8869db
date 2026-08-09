"""官方 V1.6 七项排名与二次排序模块测试。"""

import pytest

from rsdet.evaluation.official_ranking import (
    RankingItem,
    TeamRankingResult,
    compute_official_rankings,
    compute_score_range_from_percentile,
    format_ranking_table,
)


def _item(
    team_id,
    ship_recall=0.8,
    ship_fdr=0.2,
    aircraft_recall=0.8,
    aircraft_fdr=0.2,
    vehicle_recall=0.8,
    vehicle_fdr=0.2,
    latency_seconds=10.0,
):
    return RankingItem(
        team_id=team_id,
        ship_recall=ship_recall,
        ship_fdr=ship_fdr,
        aircraft_recall=aircraft_recall,
        aircraft_fdr=aircraft_fdr,
        vehicle_recall=vehicle_recall,
        vehicle_fdr=vehicle_fdr,
        latency_seconds=latency_seconds,
    )


class TestRankingItem:
    def test_metric_values_includes_latency(self):
        item = _item("A")
        values = item.metric_values()
        assert values["latency_seconds"] == 10.0
        assert len(values) == 7

    def test_metric_values_omits_missing_latency(self):
        item = _item("A", latency_seconds=None)
        values = item.metric_values()
        assert "latency_seconds" not in values
        assert len(values) == 6
        assert item.incomplete is True

    def test_incomplete_flag_false_with_latency(self):
        assert _item("A").incomplete is False


class TestComputeOfficialRankings:
    def test_single_team_uses_default_percentile(self):
        """单队伍时二次排序百分比取 0.5，区间 [3, 7]。"""
        results = compute_official_rankings([_item("A")])
        assert len(results) == 1
        assert results[0].rank_sum == 7
        assert results[0].rank_count == 7
        assert results[0].second_order_percentile == pytest.approx(0.5)
        assert results[0].score_min == pytest.approx(3.0)
        assert results[0].score_max == pytest.approx(7.0)

    def test_all_better_team_ranks_first(self):
        """全面领先的队伍应排第一。"""
        items = [
            _item("bad", ship_recall=0.5, aircraft_recall=0.5, vehicle_recall=0.5),
            _item("good", ship_recall=0.9, aircraft_recall=0.9, vehicle_recall=0.9),
        ]
        results = compute_official_rankings(items)
        assert results[0].team_id == "good"
        assert results[1].team_id == "bad"
        assert results[0].rank_sum < results[1].rank_sum

    def test_competition_ranking_ties(self):
        """并列指标使用竞赛排名（1,2,2,4），排名和只计参与项。"""
        items = [
            _item("a", ship_recall=0.9),
            _item("b", ship_recall=0.9),
            _item("c", ship_recall=0.7),
        ]
        results = compute_official_rankings(items)
        ranks = {result.team_id: result.metric_ranks["ship_recall"] for result in results}
        assert ranks["a"] == 1
        assert ranks["b"] == 1
        assert ranks["c"] == 3

    def test_missing_latency_does_not_crash(self):
        """缺时延的队伍只按 6 项排名，且标记 incomplete。"""
        items = [
            _item("a", latency_seconds=None),
            _item("b"),
        ]
        results = compute_official_rankings(items)
        by_id = {result.team_id: result for result in results}
        assert by_id["a"].rank_count == 6
        assert by_id["a"].incomplete is True
        assert by_id["b"].rank_count == 7
        assert by_id["b"].incomplete is False

    def test_lower_is_better_for_fdr_and_latency(self):
        """FDR 和时延越低排名越靠前。"""
        items = [
            _item("slow", latency_seconds=19.0, ship_fdr=0.5),
            _item("fast", latency_seconds=2.0, ship_fdr=0.1),
        ]
        results = compute_official_rankings(items)
        by_id = {result.team_id: result for result in results}
        assert by_id["fast"].metric_ranks["latency_seconds"] == 1
        assert by_id["slow"].metric_ranks["latency_seconds"] == 2
        assert by_id["fast"].metric_ranks["ship_fdr"] == 1
        assert by_id["slow"].metric_ranks["ship_fdr"] == 2

    def test_duplicate_team_ids_rejected(self):
        with pytest.raises(ValueError, match="team_id 必须唯一"):
            compute_official_rankings([_item("a"), _item("a")])

    def test_empty_input_returns_empty(self):
        assert compute_official_rankings([]) == []

    def test_score_range_matches_official_example(self):
        """官方示例：第 30% -> 区间 5-9 分。"""
        results = compute_official_rankings(
            [_item("a"), _item("b"), _item("c"), _item("d"), _item("e"), _item("f"), _item("g"), _item("h"), _item("i"), _item("j")],
        )
        # 第 3 名（position=3, n=10, p=0.3）
        assert results[2].second_order_percentile == pytest.approx(0.3)
        assert results[2].score_min == pytest.approx(5.0)
        assert results[2].score_max == pytest.approx(9.0)

    def test_percentile_scale_zero_uses_position_minus_one(self):
        items = [_item(f"t{i}") for i in range(5)]
        results = compute_official_rankings(
            items, percentile_fraction_scale=0.0
        )
        assert results[0].second_order_percentile == pytest.approx(0.0)
        assert results[4].second_order_percentile == pytest.approx(1.0)


class TestComputeScoreRangeFromPercentile:
    def test_percentile_30(self):
        assert compute_score_range_from_percentile(0.3) == pytest.approx((5.0, 9.0))

    def test_percentile_0(self):
        assert compute_score_range_from_percentile(0.0) == pytest.approx((8.0, 10.0))

    def test_percentile_100(self):
        assert compute_score_range_from_percentile(1.0) == pytest.approx((0.0, 2.0))

    def test_invalid_percentile(self):
        with pytest.raises(ValueError, match="percentile"):
            compute_score_range_from_percentile(1.5)


class TestFormatRankingTable:
    def test_table_has_header_and_rows(self):
        results = compute_official_rankings([_item("a"), _item("b")])
        lines = format_ranking_table(results)
        # header + 分隔线 + 每支队伍一行
        assert len(lines) == 4
        assert "team" in lines[0]
        assert "a" in lines[2]
        assert "b" in lines[3]

    def test_incomplete_row_marked(self):
        results = compute_official_rankings(
            [_item("a", latency_seconds=None), _item("b")]
        )
        lines = format_ranking_table(results)
        assert any("[incomplete]" in line for line in lines)


class TestTeamRankingResultSerialization:
    def test_to_dict_contains_all_fields(self):
        results = compute_official_rankings([_item("a")])
        data = results[0].to_dict()
        assert data["team_id"] == "a"
        assert data["rank_count"] == 7
        assert "second_order_percentile" in data
        assert "score_min" in data
        assert "score_max" in data
