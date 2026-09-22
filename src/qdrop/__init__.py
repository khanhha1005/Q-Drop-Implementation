"""Simple factory-based Q-Drop API shared across project paths."""

from .backends.tensorflow_runtime import TensorFlowQDropRuntime
from .backends.torch_runtime import TorchQDropRuntime
from .core import QDropUnit
from .factories import QDropRuntimeFactory, QDropSpecFactory
from .session import QDropSession
from .types import QDropConfig, QDropDropoutState, QDropLayerSpec, QDropTensorSpec, SupportsQDropSpec

__all__ = [
    "QDropConfig",
    "QDropDropoutState",
    "QDropLayerSpec",
    "QDropRuntimeFactory",
    "QDropSession",
    "QDropSpecFactory",
    "QDropTensorSpec",
    "TensorFlowQDropRuntime",
    "TorchQDropRuntime",
    "QDropUnit",
    "SupportsQDropSpec",
]
