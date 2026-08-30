import importlib.util
from pathlib import Path


def test_voi_script_imports() -> None:
    path = Path(__file__).parents[1] / "scripts" / "apply_voi_dual_scores.py"
    spec = importlib.util.spec_from_file_location("apply_voi_dual_scores", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
