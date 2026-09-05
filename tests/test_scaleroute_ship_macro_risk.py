import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "analyze_scaleroute_ship_macro_risk_cv3.py"
    spec = importlib.util.spec_from_file_location("analyze_scaleroute_ship_macro_risk_cv3", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = load_script()


def test_compose_fold_changes_only_ship_thresholds() -> None:
    primary = {
        1: [
            {"category_id": 0, "score": 0.4},
            {"category_id": 4, "score": 0.6},
            {"category_id": 24, "score": 0.9},
        ]
    }
    expert = {1: [{"category_id": 24, "score": 0.7}]}
    result = module.compose_fold(
        image_ids={1},
        primary=primary,
        expert=expert,
        primary_threshold=0.5,
        expert_threshold=0.6,
        ship_thresholds={0: 0.3, 1: 0.5, 2: 0.5, 3: 0.5},
    )
    assert [(row["category_id"], row["score"]) for row in result[1]] == [
        (0, 0.4),
        (4, 0.6),
        (24, 0.7),
    ]
