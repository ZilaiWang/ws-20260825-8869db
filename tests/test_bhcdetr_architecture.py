"""Shape and task-branch isolation tests for the decoupled decoder."""

from __future__ import annotations

import subprocess
import sys

import pytest

torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    check=False,
    capture_output=True,
    text=True,
)
if torch_probe.returncode != 0:  # pragma: no cover - environment dependent
    reason = (torch_probe.stderr or torch_probe.stdout).strip().splitlines()
    pytest.skip(
        f"PyTorch is unavailable: {reason[-1] if reason else 'import failed'}",
        allow_module_level=True,
    )

import torch  # noqa: E402

from rsdet.models.bhcdetr import BHCDetrConfig, DecoupledDecoderLayer  # noqa: E402


def _small_config() -> BHCDetrConfig:
    return BHCDetrConfig(
        image_size=32,
        backbone_pretrained=False,
        hidden_dim=8,
        num_queries=4,
        encoder_layers=1,
        decoder_layers=1,
        nheads=2,
        dim_feedforward=16,
        projection_dim=4,
        dropout=0.0,
    )


def test_decoupled_layer_preserves_paired_query_shapes() -> None:
    layer = DecoupledDecoderLayer(_small_config())
    classification = torch.randn(2, 4, 8)
    localization = torch.randn(2, 4, 8)
    memory = torch.randn(2, 6, 8)
    position = torch.randn(2, 6, 8)

    output_classification, output_localization = layer(
        classification,
        localization,
        memory,
        position,
        None,
    )

    assert output_classification.shape == (2, 4, 8)
    assert output_localization.shape == (2, 4, 8)


def test_classification_output_does_not_use_localization_specific_branch() -> None:
    layer = DecoupledDecoderLayer(_small_config())
    classification = torch.randn(1, 4, 8, requires_grad=True)
    localization = torch.randn(1, 4, 8, requires_grad=True)
    memory = torch.randn(1, 6, 8, requires_grad=True)
    position = torch.randn(1, 6, 8)

    output_classification, _ = layer(classification, localization, memory, position, None)
    output_classification.square().sum().backward()

    assert any(
        parameter.grad is not None
        for parameter in layer.classification_branch.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in layer.localization_branch.parameters()
    )
    assert any(parameter.grad is not None for parameter in layer.self_attn.parameters())


def test_config_rejects_invalid_attention_and_position_dimensions() -> None:
    with pytest.raises(ValueError, match="nheads"):
        BHCDetrConfig(hidden_dim=8, nheads=0)
    with pytest.raises(ValueError, match="divisible by 4"):
        BHCDetrConfig(hidden_dim=6, nheads=3)


def test_uhr_decoder_uses_local_memory_and_backpropagates() -> None:
    config = BHCDetrConfig(
        image_size=32,
        backbone_pretrained=False,
        hidden_dim=8,
        num_queries=4,
        encoder_layers=1,
        decoder_layers=1,
        nheads=2,
        dim_feedforward=16,
        projection_dim=4,
        dropout=0.0,
        uhr_enabled=True,
        uhr_patch_size=16,
        uhr_patch_budget=1,
        uhr_max_local_tokens=4,
    )
    layer = DecoupledDecoderLayer(config)
    classification = torch.randn(1, 4, 8)
    localization = torch.randn(1, 4, 8)
    memory = torch.randn(1, 6, 8)
    position = torch.randn(1, 6, 8)
    local_memory = torch.randn(1, 3, 8, requires_grad=True)
    local_position = torch.randn(1, 3, 8)

    classification_output, localization_output = layer(
        classification,
        localization,
        memory,
        position,
        None,
        local_memory,
        local_position,
        None,
    )
    (classification_output.square().mean() + localization_output.square().mean()).backward()

    assert local_memory.grad is not None
    assert torch.isfinite(local_memory.grad).all()
    assert any(
        parameter.grad is not None
        for parameter in layer.classification_branch.local_cross_attn.parameters()
    )
