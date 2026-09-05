import numpy as np

from rsdet.contracts import InferenceSample, Prediction
from rsdet.models.rot90_view import Rot90ViewDetector
from rsdet.postprocess.complementary_rescue import append_class_rescue


def test_rectangular_continuous_box_inverse_and_pixels():
    pixels = np.zeros((60, 100, 3), dtype=np.uint8)
    pixels[10:30, 20:50] = 255

    class Detector:
        def predict(self, batch):
            s = batch[0]
            assert (s.width, s.height) == (60, 100)
            assert np.all(s.image[50:80, 10:30] == 255)
            return [Prediction(7, [[10, 50, 30, 80]], [0.8], [24])]

    got = Rot90ViewDetector(Detector()).predict([InferenceSample(7, pixels, 100, 60)])
    assert got[0].boxes_xyxy == [[20, 10, 50, 30]]
    assert got[0].scores == [0.8] and got[0].labels == [24]


def test_class_rescue_never_cross_class_suppresses_or_replaces():
    box = {"category_id": 0, "score": 0.9, "bbox_xyxy": [0, 0, 10, 10]}
    aux = [dict(box, category_id=1), dict(box, score=0.99),
           dict(box, category_id=24)]
    result, counts = append_class_rescue({1: [box]}, {1: aux},
                                         category_iou={0: 0.5, 1: 0.5})
    assert result == {1: [box, aux[0]]}
    assert counts == {"auxiliary_candidates": 2, "suppressed_overlap": 1, "added": 1}
