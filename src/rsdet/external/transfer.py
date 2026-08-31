"""Ultralytics trainer factory for audited coarse-to-fine head transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_ARCHITECTURE_FIELDS = (
    "backbone",
    "head",
    "scales",
    "scale",
    "depth_multiple",
    "width_multiple",
    "channels",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def model_architecture_sha256(model_yaml: dict[str, Any]) -> str:
    """Hash architecture fields while deliberately excluding class-count metadata."""
    payload = {
        key: _jsonable(model_yaml[key])
        for key in _ARCHITECTURE_FIELDS
        if key in model_yaml
    }
    if "backbone" not in payload or "head" not in payload:
        raise ValueError("model YAML must contain backbone and head architecture fields")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _name_list_sha256(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode("utf-8")).hexdigest()


def external_head_transfer_trainer(
    source_state: dict[str, Any],
    audit_path: Path,
    *,
    expected_target_nc: int,
    reset_seed: int,
    source_model_yaml: dict[str, Any],
) -> type:
    """Build a trainer that audits backbone/neck loading and resets the full Detect head."""
    import torch
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import unwrap_model

    frozen_source = {name: value.detach().cpu().clone() for name, value in source_state.items()}
    source_architecture_sha256 = model_architecture_sha256(source_model_yaml)

    class _ExternalHeadTransferTrainer(DetectionTrainer):
        def setup_model(self):
            """Reset before optimizer/EMA creation, after Ultralytics loaded weights."""
            checkpoint = super().setup_model()
            if getattr(self, "_external_transfer_complete", False):
                return checkpoint
            model = unwrap_model(self.model)
            layers = getattr(model, "model", None)
            if layers is None or len(layers) < 2:
                raise ValueError("Ultralytics model lacks an auditable layer list")
            head_index = len(layers) - 1
            target_nc = int(
                getattr(
                    model,
                    "nc",
                    getattr(layers[head_index], "nc", getattr(model, "yaml", {}).get("nc", -1)),
                )
            )
            if target_nc != expected_target_nc:
                raise ValueError(
                    f"target dataset/model class count {target_nc} != {expected_target_nc}"
                )
            head_prefix = f"model.{head_index}."
            target_state = model.state_dict()
            shape_compatible = []
            loaded_equal = []
            for name, value in target_state.items():
                source = frozen_source.get(name)
                if source is None or tuple(source.shape) != tuple(value.shape):
                    continue
                shape_compatible.append(name)
                if torch.equal(source, value.detach().cpu()):
                    loaded_equal.append(name)
            backbone_neck_keys = [
                name for name in target_state if not name.startswith(head_prefix)
            ]
            backbone_neck_compatible = [
                name for name in shape_compatible if not name.startswith(head_prefix)
            ]
            backbone_neck_loaded = [
                name for name in loaded_equal if not name.startswith(head_prefix)
            ]
            if set(backbone_neck_keys) != set(backbone_neck_compatible):
                missing = sorted(set(backbone_neck_keys) - set(backbone_neck_compatible))
                extra = sorted(set(backbone_neck_compatible) - set(backbone_neck_keys))
                raise RuntimeError(
                    "source checkpoint does not cover the exact target backbone/neck tensor set: "
                    f"missing_first={missing[:10]}, extra_first={extra[:10]}"
                )
            if set(backbone_neck_compatible) != set(backbone_neck_loaded):
                missing = sorted(set(backbone_neck_compatible) - set(backbone_neck_loaded))
                raise RuntimeError(
                    "not all shape-compatible backbone/neck tensors were loaded: "
                    f"first={missing[:10]}"
                )
            target_architecture_sha256 = model_architecture_sha256(model.yaml)
            if target_architecture_sha256 != source_architecture_sha256:
                raise RuntimeError(
                    "source and target architecture YAML signatures differ: "
                    f"source={source_architecture_sha256}, "
                    f"target={target_architecture_sha256}"
                )
            torch.manual_seed(reset_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(reset_seed)
            # A generic recursive reset destroys Detect.bias_init() and can
            # yield catastrophic first-epoch classification loss.  Build a
            # native fresh model instead, then transplant its complete head.
            fresh_model = DetectionModel(
                model.yaml,
                ch=3,
                nc=target_nc,
                verbose=False,
            )
            fresh_head_state = fresh_model.model[head_index].state_dict()
            layers[head_index].load_state_dict(fresh_head_state, strict=True)
            head_parameter_names = [
                name
                for name, _parameter in model.named_parameters()
                if name.startswith(head_prefix)
            ]
            post_reset_state = model.state_dict()
            head_parameters_equal_source_after_reset = [
                name
                for name in head_parameter_names
                if name in frozen_source
                and tuple(frozen_source[name].shape) == tuple(post_reset_state[name].shape)
                and torch.equal(frozen_source[name], post_reset_state[name].detach().cpu())
            ]
            audit = {
                "status": "pass",
                "protocol": "external_coarse_backbone_neck_transfer_full_head_reset_v1",
                "target_nc": target_nc,
                "head_layer_index": head_index,
                "head_prefix": head_prefix,
                "source_tensor_count": len(frozen_source),
                "target_tensor_count": len(target_state),
                "shape_compatible_tensor_count": len(shape_compatible),
                "loaded_equal_tensor_count_before_head_reset": len(loaded_equal),
                "backbone_neck_tensor_count": len(backbone_neck_keys),
                "backbone_neck_shape_compatible_count": len(backbone_neck_compatible),
                "backbone_neck_loaded_equal_count": len(backbone_neck_loaded),
                "backbone_neck_tensor_names_sha256": _name_list_sha256(
                    backbone_neck_keys
                ),
                "backbone_neck_tensor_names": sorted(backbone_neck_keys),
                "source_architecture_sha256": source_architecture_sha256,
                "target_architecture_sha256": target_architecture_sha256,
                "native_fresh_head_tensor_count": len(fresh_head_state),
                "head_parameter_tensor_count": len(head_parameter_names),
                "head_parameters_equal_source_after_reset_count": len(
                    head_parameters_equal_source_after_reset
                ),
                "reset_seed": reset_seed,
                "head_policy": (
                    "entire final Detect module replaced from a native freshly constructed "
                    "target-nc model after Ultralytics bias_init"
                ),
            }
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            self._external_transfer_complete = True
            return checkpoint

    _ExternalHeadTransferTrainer.__name__ = "ExternalHeadTransferTrainer"
    return _ExternalHeadTransferTrainer


__all__ = ["external_head_transfer_trainer", "model_architecture_sha256"]
