import importlib.util
from pathlib import Path


def test_comparison_script_imports() -> None:
    path = Path(__file__).parents[1] / "scripts" / "compare_metric_risk_stages.py"
    spec = importlib.util.spec_from_file_location("compare_metric_risk_stages", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = {
        "frontiers": {
            "0.150": {
                "crossfit": {
                    "recall": 0.9,
                    "fdr": 0.1,
                    "tp": 9,
                    "fp": 1,
                    "per_coarse": {},
                }
            }
        }
    }
    assert module._point(payload, 0.15)["tp"] == 9
