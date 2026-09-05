"""Train-source-only BN experiment. No optimizer, fusion, or test-time adaptation.

The caller supplies an unfused model and equally sized batches. Running moments
use the standard cumulative average of batch estimates, not a claim of exact
population variance. Only BN buffers may change; every parameter is conserved.
Torch is imported lazily so dataset/contract preflight remains GPU-independent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def state_digest(state: dict) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def bn_buffer_names(model) -> set[str]:
    from torch.nn.modules.batchnorm import _BatchNorm

    names = set()
    for name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            if not module.track_running_stats or module.running_mean is None:
                raise ValueError("BN without running statistics is unsupported")
            prefix = f"{name}." if name else ""
            names.update(prefix + s for s in ("running_mean", "running_var", "num_batches_tracked"))
    if not names:
        raise ValueError("no BN buffers: fused checkpoints cannot be recalibrated")
    return names


def verify_only_bn_changed(before: dict, after: dict, allowed: set[str]) -> dict:
    import torch

    if set(before) != set(after) or not allowed <= set(before):
        raise ValueError("checkpoint state structure changed")
    changed = []
    for name, value in after.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite model state: {name}")
        same = before[name].dtype == value.dtype and torch.equal(before[name].cpu(), value.cpu())
        if not same:
            if name not in allowed:
                raise ValueError(f"non-BN state changed: {name}")
            changed.append(name)
        if name in allowed and name.endswith("running_var") and (value < 0).any():
            raise ValueError(f"negative BN variance: {name}")
    unchanged = {k: v for k, v in before.items() if k not in allowed}
    return {
        "non_bn_state_sha256": state_digest(unchanged),
        "changed_bn_buffers": sorted(changed),
        "all_non_bn_state_bitwise_equal": True,
    }


def reestimate_bn(model, batches: Iterable, *, batch_size: int, expected_batches: int) -> dict:
    import torch
    from torch.nn.modules.batchnorm import _BatchNorm

    if batch_size < 2 or expected_batches < 1:
        raise ValueError("positive batch count and batch_size >= 2 required")
    allowed = bn_buffer_names(model)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    verify_only_bn_changed(before, before, allowed)
    modules = list(model.modules())
    modes = [m.training for m in modules]
    bns = [m for m in modules if isinstance(m, _BatchNorm)]
    momenta = [m.momentum for m in bns]
    parameters = list(model.parameters())
    gradients = [p.requires_grad for p in parameters]
    processed = 0
    try:
        model.eval()  # Detect, Dropout and all other layers stay in evaluation mode.
        for p in parameters:
            p.requires_grad_(False)
        for bn in bns:
            bn.reset_running_stats()
            bn.momentum = None
            bn.train()
        with torch.no_grad():
            for inputs in batches:
                if processed >= expected_batches:
                    raise ValueError("more batches than declared")
                if (
                    inputs.ndim != 4 or inputs.shape[0] != batch_size
                    or inputs.shape[1] != 3 or inputs.dtype != torch.float32
                    or not torch.isfinite(inputs).all()
                    or inputs.min() < 0 or inputs.max() > 1
                ):
                    raise ValueError("expected equal NCHW float32 RGB batches in [0,1]")
                model(inputs)
                processed += 1
        if processed != expected_batches:
            raise ValueError("fewer batches than declared")
        if any(int(b.num_batches_tracked) != processed for b in bns):
            raise ValueError("BN layers were skipped or reused; unsupported model graph")
        audit = verify_only_bn_changed(before, model.state_dict(), allowed)
        return {**audit, "bn_layers": len(bns), "batches": processed,
                "images": processed * batch_size, "estimator": "cumulative_equal_batch_estimates"}
    except BaseException:
        model.load_state_dict(before)  # A failed call never leaves half-updated moments.
        raise
    finally:
        for bn, momentum in zip(bns, momenta):
            bn.momentum = momentum
        for m, mode in zip(modules, modes):
            m.training = mode
        for p, required in zip(parameters, gradients):
            p.requires_grad_(required)


def copy_bn_buffers(source, destination) -> dict:
    """Copy moments back to the parent's original dtype; preserve weights exactly."""
    import torch

    allowed = bn_buffer_names(destination)
    if bn_buffer_names(source) != allowed:
        raise ValueError("BN architecture mismatch")
    before = {k: v.detach().cpu().clone() for k, v in destination.state_dict().items()}
    dst, src = destination.state_dict(), source.state_dict()
    if any(dst[name].shape != src[name].shape for name in allowed):
        raise ValueError("BN shape mismatch")
    with torch.no_grad():
        for name in allowed:
            dst[name].copy_(src[name].to(device=dst[name].device, dtype=dst[name].dtype))
    try:
        return verify_only_bn_changed(before, destination.state_dict(), allowed)
    except BaseException:
        destination.load_state_dict(before)
        raise
