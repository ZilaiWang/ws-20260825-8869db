from rsdet.evaluation.official_metric import CoarseMacroMetrics, RankingMetrics
from rsdet.evaluation.platform_protocol import build_platform_observed_metrics


def test_platform_gate_is_equal_weighted_three_coarse_macro() -> None:
    ranking = RankingMetrics(
        per_coarse={
            "ship": CoarseMacroMetrics(0.874969, 0.320177, 0.99, 0.01, 4, [0, 1, 2, 3]),
            "aircraft": CoarseMacroMetrics(0.967641, 0.064691, 0.10, 0.90, 20, list(range(4, 24))),
            "vehicle": CoarseMacroMetrics(0.852632, 0.325, 0.10, 0.90, 1, [24]),
        }
    )
    result = build_platform_observed_metrics(ranking, latency_seconds=2.473167)
    assert abs(result.gate_recall - (0.874969 + 0.967641 + 0.852632) / 3) < 1e-12
    assert abs(result.gate_fdr - (0.320177 + 0.064691 + 0.325) / 3) < 1e-12
    assert result.recall_pass
    assert not result.fdr_pass
    assert result.absolute_score is not None
