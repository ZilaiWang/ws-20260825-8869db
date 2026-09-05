"""Audited class-row task vectors for same-architecture YOLO checkpoints.

The deployment experiment is intentionally narrow: it changes only explicitly
selected rows in final class-logit convolutions.  Box geometry, objectness,
normalization state, and every non-selected class row remain inherited from the
incumbent checkpoint.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from typing import Any


def resolve_inference_model(checkpoint: dict[str, Any]) -> tuple[str, Any]:
    """Resolve the module Ultralytics will use for inference."""

    try:
        from torch import nn
    except ImportError as error:  # pragma: no cover - exercised on GPU hosts
        raise RuntimeError("class task vectors require PyTorch") from error
    for key in ("ema", "model"):
        value = checkpoint.get(key)
        if isinstance(value, nn.Module):
            return key, value
    raise KeyError("checkpoint has no nn.Module under 'ema' or 'model'")


def assert_same_architecture(base: Any, donor: Any) -> None:
    """Reject mismatched state keys, tensor shapes, and fine-class names."""

    base_state = base.state_dict()
    donor_state = donor.state_dict()
    if list(base_state) != list(donor_state):
        differences = sorted(set(base_state) ^ set(donor_state))
        raise ValueError(f"state keys differ: {differences[:20]}")
    mismatched = [
        key for key in base_state if base_state[key].shape != donor_state[key].shape
    ]
    if mismatched:
        raise ValueError(f"state shapes differ: {mismatched[:20]}")
    base_names = getattr(base, "names", None)
    donor_names = getattr(donor, "names", None)
    if base_names != donor_names:
        raise ValueError("fine-class name tables differ")


def final_class_conv_candidates(model: Any, num_classes: int) -> list[str]:
    """List every Conv2d whose output dimension equals the class count."""

    from torch import nn

    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d) and module.out_channels == num_classes
    ]


def interpolate_class_rows(
    base: Any,
    donor: Any,
    *,
    alpha: float,
    num_classes: int,
    class_ids: Sequence[int],
    module_regex: str,
    expected_module_count: int | None = None,
) -> tuple[Any, list[str]]:
    """Apply ``base + alpha * (donor-base)`` to selected logit rows only."""

    import torch
    from torch import nn

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    ids = tuple(int(value) for value in class_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("class_ids must be non-empty and unique")
    if any(value < 0 or value >= num_classes for value in ids):
        raise ValueError("class_ids contain an out-of-range value")
    assert_same_architecture(base, donor)
    pattern = re.compile(module_regex)
    result = copy.deepcopy(base).cpu()
    donor_modules = dict(donor.named_modules())
    matched: list[str] = []
    for name, module in result.named_modules():
        if not pattern.fullmatch(name):
            continue
        if not isinstance(module, nn.Conv2d) or module.out_channels != num_classes:
            raise ValueError(
                f"matched module {name!r} is not Conv2d(out_channels={num_classes})"
            )
        other = donor_modules.get(name)
        if not isinstance(other, nn.Conv2d):
            raise ValueError(f"donor module missing or incompatible: {name!r}")
        if module.weight.shape != other.weight.shape:
            raise ValueError(f"donor weight shape mismatch: {name!r}")
        if (module.bias is None) != (other.bias is None):
            raise ValueError(f"donor bias contract mismatch: {name!r}")
        with torch.no_grad():
            for class_id in ids:
                base_row = module.weight[class_id].float()
                donor_row = other.weight[class_id].detach().cpu().float()
                module.weight[class_id].copy_(
                    (base_row + alpha * (donor_row - base_row)).to(module.weight.dtype)
                )
                if module.bias is not None and other.bias is not None:
                    base_bias = module.bias[class_id].float()
                    donor_bias = other.bias[class_id].detach().cpu().float()
                    module.bias[class_id].copy_(
                        (base_bias + alpha * (donor_bias - base_bias)).to(module.bias.dtype)
                    )
        matched.append(name)
    if not matched:
        raise ValueError("module_regex matched no final class convolutions")
    if expected_module_count is not None and len(matched) != expected_module_count:
        raise ValueError(
            f"matched {len(matched)} modules; expected {expected_module_count}: {matched}"
        )
    assert_only_selected_rows_changed(
        base,
        result,
        matched_modules=matched,
        class_ids=ids,
    )
    return result, matched


def assert_only_selected_rows_changed(
    base: Any,
    candidate: Any,
    *,
    matched_modules: Sequence[str],
    class_ids: Sequence[int],
) -> None:
    """Bitwise audit that no tensor outside the allow-listed rows changed."""

    import torch

    allowed: dict[str, set[int]] = {}
    for module_name in matched_modules:
        allowed[f"{module_name}.weight"] = set(class_ids)
        allowed[f"{module_name}.bias"] = set(class_ids)
    base_state = base.state_dict()
    candidate_state = candidate.state_dict()
    for key, before in base_state.items():
        after = candidate_state[key]
        rows = allowed.get(key)
        if rows is None:
            if not torch.equal(before.detach().cpu(), after.detach().cpu()):
                raise AssertionError(f"non-target tensor changed: {key}")
            continue
        for row in range(before.shape[0]):
            if row in rows:
                continue
            if not torch.equal(before[row].detach().cpu(), after[row].detach().cpu()):
                raise AssertionError(f"non-target row changed: {key}[{row}]")


__all__ = [
    "assert_only_selected_rows_changed",
    "assert_same_architecture",
    "final_class_conv_candidates",
    "interpolate_class_rows",
    "resolve_inference_model",
]
