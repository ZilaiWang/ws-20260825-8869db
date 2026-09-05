import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "analyze_background_resolution_route.py"
    spec = importlib.util.spec_from_file_location("analyze_background_resolution_route", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = load_script()


def test_background_route_is_class_disjoint() -> None:
    manifest = [{"image_id": 1, "width": 1000, "height": 1000}]
    primary = {
        1: [
            {"category_id": 0, "score": 0.8},
            {"category_id": 24, "score": 0.8},
        ]
    }
    expert = {
        1: [
            {"category_id": 0, "score": 0.9},
            {"category_id": 24, "score": 0.9},
        ]
    }
    mapping = {index: ("vehicle" if index == 24 else "ship") for index in range(25)}
    result = module.analyze(
        manifest=manifest,
        primary=primary,
        expert=expert,
        primary_labels=frozenset(range(24)),
        expert_labels=frozenset({24}),
        primary_threshold=0.85,
        expert_threshold=0.85,
        category_mapping=mapping,
    )
    assert result["route"]["false_positives"] == 1
    assert result["route"]["per_coarse_false_positives"] == {"ship": 0, "vehicle": 1}
