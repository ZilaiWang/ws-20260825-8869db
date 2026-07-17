"""P03 多随机种子稳定性分析的关键规则测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_script_module():
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "analyze_p03_seed_stability.py"
    spec = importlib.util.spec_from_file_location("p03_seed_stability_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Run:
    def __init__(self, value: float) -> None:
        self.metrics = {
            "macro_recall": value,
            "macro_f1": value,
            "accuracy": value,
        }


def test_two_way_decomposition_separates_seed_and_fold_effects() -> None:
    module = _load_script_module()
    groups = {}
    # 只有 fold 效应，没有 seed 效应或交互。
    for seed in module.SEEDS:
        for condition in module.CONDITIONS:
            groups[(seed, condition)] = [_Run(0.8), _Run(0.9), _Run(1.0)]
    rows = module._two_way_rows(groups)
    row = next(
        value
        for value in rows
        if value["condition"] == "clean" and value["metric"] == "macro_recall"
    )
    assert np.isclose(row["grand_mean_9_runs"], 0.9)
    assert np.isclose(row["seed_mean_range"], 0.0)
    assert np.isclose(row["fold_mean_range"], 0.2)
    assert np.isclose(row["descriptive_ss_fold_share"], 1.0)
    assert np.isclose(row["residual_rmse"], 0.0)


def test_checkpoint_inventory_requires_all_six_seed_fold_pairs(tmp_path: Path) -> None:
    module = _load_script_module()
    rows = []
    sizes = []
    for seed in (3407, 202625):
        for fold in range(3):
            path = (
                f"/workspace/results/P03-TASK-04/"
                f"ft-natural-tight-224-seed{seed}-fold{fold}/best_checkpoint.pt"
            )
            rows.append(f"{'a' * 64}  {path}")
            sizes.append(f"111 {path}")
    (tmp_path / "CHECKPOINTS_SHA256.txt").write_text("\n".join(rows) + "\n")
    (tmp_path / "CHECKPOINTS_SIZES.txt").write_text("\n".join(sizes) + "\n")
    result = module._parse_checkpoint_inventory(tmp_path)
    assert result["count"] == 6
    assert result["all_sizes_equal"] is True
    assert result["size_bytes"] == [111]
