"""Quality-aware and hard-negative-focused loss injection for YOLO26.

YOLO26 already assigns soft positive scores with task-aligned matching, but its
default classification criterion is unweighted BCE.  The loss below applies a
Varifocal-style negative weighting elementwise while preserving Ultralytics'
existing normalisation and end-to-end one-to-many/one-to-one schedule. Positive
anchors can use their soft quality target (Varifocal) or unit weight (the
recall-protecting hard-negative focal ablation).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def quality_aware_classification_loss(
    alpha: float = 0.75,
    gamma: float = 2.0,
    *,
    positive_weighting: str = "quality",
    focus_class_indices: Sequence[int] | None = None,
) -> Any:
    """Return an elementwise quality-aware BCE module.

    The returned tensor has the same shape as the logits so the surrounding
    Ultralytics criterion keeps its original ``target_scores_sum``
    normalisation.  Positive anchors are weighted by their soft task-aligned
    quality target (``quality``) or unit weight (``unit``); negative anchors are
    weighted by predicted confidence to the power ``gamma``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    if positive_weighting not in {"quality", "unit"}:
        raise ValueError("positive_weighting must be 'quality' or 'unit'")
    normalized_focus = None
    if focus_class_indices is not None:
        normalized_focus = tuple(sorted({int(index) for index in focus_class_indices}))
        if not normalized_focus or normalized_focus[0] < 0:
            raise ValueError("focus_class_indices must contain non-negative indices")

    import torch
    from torch import nn
    from torch.nn import functional as F

    class _QualityAwareClassificationLoss(nn.Module):
        def forward(self, logits: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
            target_scores = target_scores.to(dtype=logits.dtype)
            labels = target_scores.gt(0).to(dtype=logits.dtype)
            probabilities = logits.sigmoid()
            positive_weights = target_scores if positive_weighting == "quality" else labels
            weights = alpha * probabilities.pow(gamma) * (1.0 - labels) + (
                positive_weights * labels
            )
            if normalized_focus is not None:
                if normalized_focus[-1] >= logits.shape[-1]:
                    raise ValueError("focus class index exceeds logits class dimension")
                focus_mask = torch.zeros(
                    logits.shape[-1], dtype=torch.bool, device=logits.device
                )
                focus_mask[list(normalized_focus)] = True
                weights = torch.where(focus_mask, weights, torch.ones_like(weights))
            return F.binary_cross_entropy_with_logits(
                logits.float(), target_scores.float(), reduction="none"
            ).to(dtype=logits.dtype) * weights

    return _QualityAwareClassificationLoss()


def quality_aware_trainer(
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
    positive_weighting: str = "quality",
    focus_class_indices: Sequence[int] | None = None,
    base_trainer: type | None = None,
) -> type:
    """Return a DetectionTrainer that replaces only classification BCE.

    ``base_trainer`` allows this injection to compose with the audited external
    coarse-to-fine head-transfer trainer.  No architecture, assignment, box
    loss, DFL loss, augmentation, or inference code is changed.
    """
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    trainer_base = base_trainer or DetectionTrainer

    class _QualityAwareTrainer(trainer_base):
        def _setup_train(self) -> None:
            super()._setup_train()
            model = unwrap_model(self.model)
            criterion = getattr(model, "criterion", None)
            if criterion is None and hasattr(model, "init_criterion"):
                criterion = model.init_criterion()
                model.criterion = criterion
            replacement = quality_aware_classification_loss(
                alpha=alpha,
                gamma=gamma,
                positive_weighting=positive_weighting,
                focus_class_indices=focus_class_indices,
            )
            if hasattr(criterion, "one2many") and hasattr(criterion, "one2one"):
                criterion.one2many.bce = replacement
                criterion.one2one.bce = quality_aware_classification_loss(
                    alpha=alpha,
                    gamma=gamma,
                    positive_weighting=positive_weighting,
                    focus_class_indices=focus_class_indices,
                )
            elif hasattr(criterion, "bce"):
                criterion.bce = replacement
            else:
                raise TypeError(
                    f"unsupported detection criterion for quality-aware loss: "
                    f"{type(criterion).__name__}"
                )

    _QualityAwareTrainer.__name__ = "QualityAwareTrainer"
    return _QualityAwareTrainer


def selective_classifier_trainer(
    *,
    focus_class_indices: Sequence[int],
    base_trainer: type,
    max_weight_relative_delta: float | None = None,
    max_bias_delta: float | None = None,
) -> type:
    """Freeze a trained detector and update only selected classifier rows.

    This wrapper is intentionally narrow: only the final 1x1 classification
    convolutions in the one-to-many and one-to-one YOLO26 heads are trainable.
    Gradient hooks keep every non-selected class row byte-stable, while box
    regression, backbone, neck, and all BatchNorm statistics remain frozen.
    The caller must use zero weight decay so optimizer-side decay cannot move
    zero-gradient rows.  Optional row-wise projection bounds implement a
    conservative residual update around the mature input checkpoint; they are
    applied after every optimizer step to both the live model and its EMA.
    """
    normalized_focus = tuple(sorted({int(index) for index in focus_class_indices}))
    if not normalized_focus or normalized_focus[0] < 0:
        raise ValueError("focus_class_indices must contain non-negative indices")
    if (max_weight_relative_delta is None) != (max_bias_delta is None):
        raise ValueError("weight and bias projection bounds must be set together")
    if max_weight_relative_delta is not None and max_weight_relative_delta <= 0:
        raise ValueError("max_weight_relative_delta must be positive")
    if max_bias_delta is not None and max_bias_delta <= 0:
        raise ValueError("max_bias_delta must be positive")

    from torch import nn
    from ultralytics.utils.torch_utils import unwrap_model

    class _SelectiveClassifierTrainer(base_trainer):
        _selective_classifier_modules: tuple[nn.Conv2d, ...] = ()
        _selective_classifier_names: tuple[str, ...] = ()
        _selective_classifier_anchors: dict[str, tuple[Any, Any]] = {}

        def _setup_train(self) -> None:
            super()._setup_train()
            model = unwrap_model(self.model)
            head = model.model[-1]
            branches = []
            branch_names = []
            for branch_name in ("cv3", "one2one_cv3"):
                branch = getattr(head, branch_name, None)
                if branch is None:
                    raise TypeError(f"YOLO26 head is missing {branch_name}")
                for scale_index, scale in enumerate(branch):
                    classifier = scale[-1]
                    if not isinstance(classifier, nn.Conv2d) or classifier.kernel_size != (1, 1):
                        raise TypeError("expected final 1x1 Conv2d classifier")
                    if normalized_focus[-1] >= classifier.out_channels:
                        raise ValueError("focus class index exceeds classifier output channels")
                    branches.append(classifier)
                    branch_names.append(
                        f"model.{len(model.model) - 1}.{branch_name}.{scale_index}.2"
                    )

            for parameter in model.parameters():
                parameter.requires_grad = False
            for classifier in branches:
                classifier.weight.requires_grad = True
                if classifier.bias is not None:
                    classifier.bias.requires_grad = True
                row_mask = classifier.weight.new_zeros(classifier.out_channels)
                row_mask[list(normalized_focus)] = 1.0
                classifier.weight.register_hook(
                    lambda grad, mask=row_mask: grad * mask[:, None, None, None]
                )
                if classifier.bias is not None:
                    classifier.bias.register_hook(
                        lambda grad, mask=row_mask: grad * mask
                    )
            self._selective_classifier_modules = tuple(branches)
            self._selective_classifier_names = tuple(branch_names)
            self._selective_classifier_anchors = {
                name: (
                    module.weight.detach().clone(),
                    None if module.bias is None else module.bias.detach().clone(),
                )
                for name, module in zip(branch_names, branches, strict=True)
            }

        def _project_selective_rows(self, target_model: Any) -> None:
            if max_weight_relative_delta is None or max_bias_delta is None:
                return
            import torch

            modules = dict(target_model.named_modules())
            indices = list(normalized_focus)
            with torch.no_grad():
                for name in self._selective_classifier_names:
                    module = modules[name]
                    anchor_weight, anchor_bias = self._selective_classifier_anchors[name]
                    anchor_weight = anchor_weight.to(
                        device=module.weight.device, dtype=module.weight.dtype
                    )
                    current_rows = module.weight[indices]
                    anchor_rows = anchor_weight[indices]
                    delta = current_rows - anchor_rows
                    delta_norm = delta.flatten(1).norm(dim=1)
                    allowed_norm = (
                        anchor_rows.flatten(1).norm(dim=1)
                        * max_weight_relative_delta
                    )
                    scale = torch.clamp(
                        allowed_norm / delta_norm.clamp_min(1e-12), max=1.0
                    )
                    projected = anchor_rows + delta * scale[:, None, None, None]
                    module.weight.copy_(anchor_weight)
                    module.weight[indices] = projected
                    if module.bias is not None and anchor_bias is not None:
                        anchor_bias = anchor_bias.to(
                            device=module.bias.device, dtype=module.bias.dtype
                        )
                        projected_bias = anchor_bias[indices] + (
                            module.bias[indices] - anchor_bias[indices]
                        ).clamp(-max_bias_delta, max_bias_delta)
                        module.bias.copy_(anchor_bias)
                        module.bias[indices] = projected_bias

        def optimizer_step(self) -> None:
            super().optimizer_step()
            from ultralytics.utils.torch_utils import unwrap_model

            self._project_selective_rows(unwrap_model(self.model))
            if self.ema is not None:
                self._project_selective_rows(unwrap_model(self.ema.ema))

        def _model_train(self) -> None:
            super()._model_train()
            model = unwrap_model(self.model)
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
            for classifier in self._selective_classifier_modules:
                classifier.train()

    _SelectiveClassifierTrainer.__name__ = "SelectiveClassifierTrainer"
    return _SelectiveClassifierTrainer


def spatial_classifier_residual_trainer(
    *,
    focus_class_indices: Sequence[int],
    base_trainer: type,
    max_branch_relative_delta: float,
    max_final_weight_relative_delta: float,
    max_final_bias_delta: float,
) -> type:
    """Adapt classifier spatial features in a bounded mature-model trust region.

    The complete ``cv3`` and ``one2one_cv3`` classification branches are
    trainable, while their final classifiers still update only the requested
    output rows. Every trainable tensor is projected around the mature input
    checkpoint after each optimizer step. This permits background supervision
    to change spatial context features without unrestricted forgetting.
    """
    normalized_focus = tuple(sorted({int(index) for index in focus_class_indices}))
    if not normalized_focus or normalized_focus[0] < 0:
        raise ValueError("focus_class_indices must contain non-negative indices")
    if max_branch_relative_delta <= 0:
        raise ValueError("max_branch_relative_delta must be positive")
    if max_final_weight_relative_delta <= 0:
        raise ValueError("max_final_weight_relative_delta must be positive")
    if max_final_bias_delta <= 0:
        raise ValueError("max_final_bias_delta must be positive")

    from torch import nn
    from ultralytics.utils.torch_utils import unwrap_model

    class _SpatialClassifierResidualTrainer(base_trainer):
        _anchors: dict[str, Any] = {}
        _final_names: frozenset[str] = frozenset()

        def _setup_train(self) -> None:
            super()._setup_train()
            model = unwrap_model(self.model)
            head = model.model[-1]
            head_index = len(model.model) - 1
            final_names: set[str] = set()
            branch_prefixes: list[str] = []
            for branch_name in ("cv3", "one2one_cv3"):
                branch = getattr(head, branch_name, None)
                if branch is None:
                    raise TypeError(f"YOLO26 head is missing {branch_name}")
                branch_prefixes.append(f"model.{head_index}.{branch_name}.")
                for scale_index, scale in enumerate(branch):
                    classifier = scale[-1]
                    if not isinstance(classifier, nn.Conv2d) or classifier.kernel_size != (1, 1):
                        raise TypeError("expected final 1x1 Conv2d classifier")
                    if normalized_focus[-1] >= classifier.out_channels:
                        raise ValueError("focus class index exceeds classifier output channels")
                    final_names.add(f"model.{head_index}.{branch_name}.{scale_index}.2")

            for parameter in model.parameters():
                parameter.requires_grad = False
            for name, parameter in model.named_parameters():
                if any(name.startswith(prefix) for prefix in branch_prefixes):
                    parameter.requires_grad = True

            modules = dict(model.named_modules())
            for name in sorted(final_names):
                classifier = modules[name]
                row_mask = classifier.weight.new_zeros(classifier.out_channels)
                row_mask[list(normalized_focus)] = 1.0
                classifier.weight.register_hook(
                    lambda grad, mask=row_mask: grad * mask[:, None, None, None]
                )
                if classifier.bias is not None:
                    classifier.bias.register_hook(lambda grad, mask=row_mask: grad * mask)

            self._anchors = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            self._final_names = frozenset(final_names)

        def _project(self, target_model: Any) -> None:
            import torch

            parameters = dict(target_model.named_parameters())
            focus = list(normalized_focus)
            with torch.no_grad():
                for name, anchor_stored in self._anchors.items():
                    parameter = parameters[name]
                    anchor = anchor_stored.to(device=parameter.device, dtype=parameter.dtype)
                    module_name, field = name.rsplit(".", 1)
                    if module_name in self._final_names:
                        current_rows = parameter[focus]
                        anchor_rows = anchor[focus]
                        delta = current_rows - anchor_rows
                        if field == "bias":
                            projected = anchor_rows + delta.clamp(
                                -max_final_bias_delta, max_final_bias_delta
                            )
                        else:
                            delta_norm = delta.flatten(1).norm(dim=1)
                            allowed = (
                                anchor_rows.flatten(1).norm(dim=1)
                                * max_final_weight_relative_delta
                            )
                            scale = (allowed / delta_norm.clamp_min(1e-12)).clamp(max=1.0)
                            projected = anchor_rows + delta * scale[:, None, None, None]
                        parameter.copy_(anchor)
                        parameter[focus] = projected
                    else:
                        delta = parameter - anchor
                        delta_norm = delta.norm()
                        allowed = anchor.norm() * max_branch_relative_delta
                        scale = (allowed / delta_norm.clamp_min(1e-12)).clamp(max=1.0)
                        parameter.copy_(anchor + delta * scale)

        def optimizer_step(self) -> None:
            super().optimizer_step()
            self._project(unwrap_model(self.model))
            if self.ema is not None:
                self._project(unwrap_model(self.ema.ema))

        def _model_train(self) -> None:
            super()._model_train()
            model = unwrap_model(self.model)
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()

    _SpatialClassifierResidualTrainer.__name__ = "SpatialClassifierResidualTrainer"
    return _SpatialClassifierResidualTrainer


__all__ = [
    "quality_aware_classification_loss",
    "quality_aware_trainer",
    "selective_classifier_trainer",
    "spatial_classifier_residual_trainer",
]
