import json
from pathlib import Path

from scripts.build_cv3_oof_pseudo_10k import excluded_sources_from_ground_truth
from scripts.build_pseudo_10k_mosaics import _largest_remainder_quotas


def test_largest_remainder_quotas_preserve_total() -> None:
    quotas = _largest_remainder_quotas(
        {"ship": 77.4, "aircraft": 18.4, "vehicle": 4.2}, 100
    )
    assert sum(quotas.values()) == 100
    assert quotas == {"ship": 78, "aircraft": 18, "vehicle": 4}


def test_excluded_sources_from_ground_truth(tmp_path: Path) -> None:
    source = tmp_path / "image.jpg"
    gt = tmp_path / "ground_truth.json"
    gt.write_text(
        json.dumps({"images": [{"source_images": [str(source)]}]}), encoding="utf-8"
    )
    assert excluded_sources_from_ground_truth([gt]) == {str(source.resolve())}
