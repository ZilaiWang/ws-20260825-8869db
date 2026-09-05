from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from rsdet.experiments.class_task_vector import (  # noqa: E402
    assert_same_architecture,
    final_class_conv_candidates,
    interpolate_class_rows,
)


class TinyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.names = {index: str(index) for index in range(3)}
        self.body = nn.Conv2d(2, 2, 1)
        self.cls = nn.Conv2d(2, 3, 1)


def test_class_task_vector_changes_only_selected_row() -> None:
    base = TinyHead()
    donor = TinyHead()
    donor.load_state_dict(base.state_dict())
    with torch.no_grad():
        donor.cls.weight[2].add_(2.0)
        donor.cls.bias[2].add_(4.0)
        donor.body.weight.add_(99.0)
    merged, matched = interpolate_class_rows(
        base,
        donor,
        alpha=0.25,
        num_classes=3,
        class_ids=[2],
        module_regex=r"cls",
        expected_module_count=1,
    )
    assert matched == ["cls"]
    assert torch.equal(merged.body.weight, base.body.weight)
    assert torch.equal(merged.cls.weight[:2], base.cls.weight[:2])
    assert torch.allclose(merged.cls.weight[2], base.cls.weight[2] + 0.5)
    assert torch.allclose(merged.cls.bias[2], base.cls.bias[2] + 1.0)


def test_alpha_zero_is_bitwise_identity() -> None:
    base = TinyHead()
    donor = TinyHead()
    merged, _ = interpolate_class_rows(
        base,
        donor,
        alpha=0.0,
        num_classes=3,
        class_ids=[2],
        module_regex=r"cls",
        expected_module_count=1,
    )
    for key, value in base.state_dict().items():
        assert torch.equal(value, merged.state_dict()[key])


def test_candidate_listing_and_architecture_mismatch() -> None:
    base = TinyHead()
    assert final_class_conv_candidates(base, 3) == ["cls"]
    donor = TinyHead()
    donor.names[2] = "different"
    with pytest.raises(ValueError, match="name tables"):
        assert_same_architecture(base, donor)
