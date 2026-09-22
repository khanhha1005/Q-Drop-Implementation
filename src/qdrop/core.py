"""Framework-agnostic Q-Drop unit helpers."""

from __future__ import annotations

import numpy as np
from typing import Any

from .types import QDropDropoutState, QDropTensorSpec


def _is_torch_value(value: Any) -> bool:
    return type(value).__module__.startswith("torch")


def _is_tensorflow_value(value: Any) -> bool:
    return type(value).__module__.startswith("tensorflow")


def _zeros_like(value: Any, *, dtype: Any = None) -> Any:
    if _is_torch_value(value):
        import torch

        if dtype is not None:
            return torch.zeros_like(value, dtype=dtype)
        return torch.zeros_like(value)
    if _is_tensorflow_value(value):
        import tensorflow as tf

        if dtype is not None:
            return tf.zeros_like(value, dtype=dtype)
        return tf.zeros_like(value)
    raise TypeError(f"Unsupported tensor type for zeros_like: {type(value)!r}")


def _ones_like(value: Any, *, dtype: Any = None) -> Any:
    if _is_torch_value(value):
        import torch

        if dtype is not None:
            return torch.ones_like(value, dtype=dtype)
        return torch.ones_like(value)
    if _is_tensorflow_value(value):
        import tensorflow as tf

        if dtype is not None:
            return tf.ones_like(value, dtype=dtype)
        return tf.ones_like(value)
    raise TypeError(f"Unsupported tensor type for ones_like: {type(value)!r}")


def _clone_like(value: Any) -> Any:
    if _is_torch_value(value):
        return value.detach().clone()
    if _is_tensorflow_value(value):
        import tensorflow as tf

        return tf.identity(value)
    raise TypeError(f"Unsupported tensor type for clone_like: {type(value)!r}")


def _add_values(left: Any, right: Any) -> Any:
    return left + right


def _logical_or(left: Any, right: Any) -> Any:
    if _is_torch_value(left):
        import torch

        return torch.logical_or(left, right)
    if _is_tensorflow_value(left):
        import tensorflow as tf

        return tf.logical_or(left, right)
    raise TypeError(f"Unsupported tensor type for logical_or: {type(left)!r}")


def _apply_nan_to_num(value: Any) -> Any:
    if _is_torch_value(value):
        import torch

        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    if _is_tensorflow_value(value):
        import tensorflow as tf

        finite = tf.where(tf.math.is_finite(value), value, tf.zeros_like(value))
        return tf.where(tf.math.is_nan(finite), tf.zeros_like(finite), finite)
    raise TypeError(f"Unsupported tensor type for nan_to_num: {type(value)!r}")


def _where(mask: Any, keep: Any, drop: Any) -> Any:
    if _is_torch_value(mask):
        import torch

        return torch.where(mask, keep, drop)
    if _is_tensorflow_value(mask):
        import tensorflow as tf

        return tf.where(mask, keep, drop)
    raise TypeError(f"Unsupported tensor type for where: {type(mask)!r}")


def _assign_parameter(parameter: Any, value: Any) -> None:
    if _is_torch_value(parameter):
        parameter.data.copy_(value)
        return
    if _is_tensorflow_value(parameter):
        parameter.assign(value)
        return
    raise TypeError(f"Unsupported tensor type for assign: {type(parameter)!r}")


def _reshape_like(template: Any, values: np.ndarray) -> Any:
    if _is_torch_value(template):
        import torch

        return torch.as_tensor(values, device=template.device).view_as(template)
    if _is_tensorflow_value(template):
        import tensorflow as tf

        return tf.reshape(tf.convert_to_tensor(values), tf.shape(template))
    raise TypeError(f"Unsupported tensor type for reshape_like: {type(template)!r}")


def _to_numpy(value: Any) -> np.ndarray:
    if _is_torch_value(value):
        return value.detach().cpu().numpy()
    if _is_tensorflow_value(value):
        return value.numpy()
    raise TypeError(f"Unsupported tensor type for to_numpy: {type(value)!r}")


class QDropUnit:
    """Own one quantum tensor and dummy Q-Drop state."""

    def __init__(self, spec: QDropTensorSpec):
        self.spec = spec
        self.accumulated_grad = _zeros_like(spec.parameter)
        self.prune_ratio = 0.8

    def accumulate(self, grad: Any) -> None:
        if grad is None:
            return
        self.accumulated_grad = _add_values(self.accumulated_grad, _clone_like(grad))

    def reset_accumulated_grad(self) -> None:
        self.accumulated_grad = _zeros_like(self.spec.parameter)

    def build_prune_mask(self) -> Any:
        accumulated_np = _to_numpy(self.accumulated_grad).reshape(-1)
        if accumulated_np.size == 0:
            return _ones_like(self.spec.parameter, dtype=bool)

        epsilon = 1e-8
        grad_min = float(accumulated_np.min())
        grad_max = float(accumulated_np.max())
        normalized = (accumulated_np - grad_min) / (grad_max - grad_min + epsilon)
        logits = np.log(normalized + epsilon)
        stabilized = logits - logits.max()
        probabilities = np.exp(stabilized)
        probabilities_sum = probabilities.sum()
        if not np.isfinite(probabilities_sum) or probabilities_sum <= 0:
            probabilities = np.ones_like(probabilities) / max(len(probabilities), 1)
        else:
            probabilities = probabilities / probabilities_sum

        num_samples = max(1, int(self.prune_ratio * accumulated_np.size))
        sampled_indices = np.random.choice(accumulated_np.size, size=num_samples, replace=True, p=probabilities)
        keep_mask = np.zeros(accumulated_np.size, dtype=bool)
        keep_mask[sampled_indices] = True
        return _reshape_like(self.spec.parameter, keep_mask)

    def build_pruned_gradient(self) -> Any:
        prune_mask = self.build_prune_mask()
        return self.apply_gradient_mask(self.accumulated_grad, prune_mask, keep_mask=True)

    def build_dropout_mask(self, dropout_state: QDropDropoutState) -> Any:
        if not dropout_state.enabled or not dropout_state.dropped_wires:
            return _zeros_like(self.spec.parameter, dtype=bool)

        if self.spec.wire_masks is not None:
            mask = _zeros_like(self.spec.parameter, dtype=bool)
            for wire_id in dropout_state.dropped_wires:
                wire_mask = self.spec.wire_masks.get(wire_id)
                if wire_mask is None:
                    continue
                mask = _logical_or(mask, wire_mask)
            return mask

        if self.spec.mask_builder is None:
            raise ValueError(
                f"Tensor spec '{self.spec.tensor_id}' cannot build dropout masks without wire metadata."
            )

        return self.spec.mask_builder(dropout_state.dropped_wires)

    def apply_gradient_mask(self, grad: Any, mask: Any, *, keep_mask: bool = False) -> Any:
        if grad is None:
            return None

        if keep_mask:
            return _where(mask, grad, _zeros_like(grad))
        return _where(mask, _zeros_like(grad), grad)

    def sanitize_tensor(self) -> None:
        _assign_parameter(self.spec.parameter, _apply_nan_to_num(self.spec.parameter))

    def sanitize_gradient(self, grad: Any) -> Any:
        if grad is None:
            return None
        return _apply_nan_to_num(grad)
