"""目标裁剪与 HPR 输入预处理测试。"""

import numpy as np
import pytest

from rsdet.data.object_crops import crop_and_resize


def test_crop_and_resize_keeps_rgb_and_square_shape() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[20:60, 40:90] = [10, 100, 200]

    crop = crop_and_resize(
        image,
        [40, 20, 90, 60],
        output_size=64,
        context_ratio=0.1,
    )

    assert crop.shape == (64, 64, 3)
    assert crop.dtype == np.uint8
    assert crop[..., 2].max() == 200


def test_invalid_crop_is_rejected() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="非法目标框"):
        crop_and_resize(image, [10, 10, 5, 15])
