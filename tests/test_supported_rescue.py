import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("supported", Path(__file__).parents[1] / "scripts/calibrate_p40_supported_rescue.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_support_features_and_abstention():
    row = {"category_id": 24, "score": .8, "bbox_xyxy": [0, 0, 10, 20]}
    same = dict(row, score=.25)
    wrong = dict(row, category_id=0, score=.99)
    f = module.features(row, [same, wrong])
    np.testing.assert_allclose(f, [.8, 1, .25, np.log1p(10), np.log1p(20)])
    assert not module.accept_record({"status": "abstain_insufficient_support"}, f)


def test_fixed_probability_and_no_incumbent_deletion():
    model = {"status": "fit", "mean": [0]*5, "scale": [1]*5,
             "coef": [0]*5, "intercept": 3}
    assert module.accept_record(model, [0]*5)
    assert not module.accept_record(dict(model, intercept=2), [0]*5)
    models = {str(f): {name: model for name in module.COARSE} for f in range(3)}
    row = {"category_id": 24, "score": .8, "bbox_xyxy": [0, 0, 10, 20]}
    novel = dict(row, bbox_xyxy=[100, 100, 110, 120])
    base, aux = {1: [row]}, {1: [row, novel]}
    output, counts = module.apply_models(base, aux, {1: 0}, {0: .5, 1: .5, 2: .5}, models)
    assert output == {1: [row, novel]}
    assert counts == {"candidates": 1, "accepted": 1}
