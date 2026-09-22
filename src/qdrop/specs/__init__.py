"""Spec helpers for framework-specific quantum layers."""

from .base import resolve_qdrop_layer_specs
from .pennylane_tf import PennyLaneTensorFlowSpecFactory, TensorFlowQuantumTensorAdapter
from .pennylane_torch import PennyLaneTorchSpecFactory

__all__ = [
    "PennyLaneTensorFlowSpecFactory",
    "PennyLaneTorchSpecFactory",
    "TensorFlowQuantumTensorAdapter",
    "resolve_qdrop_layer_specs",
]
