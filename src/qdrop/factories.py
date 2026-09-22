"""Q-Drop factories for specs and runtimes."""

from __future__ import annotations

from .backends.tensorflow_runtime import TensorFlowQDropRuntime
from .backends.torch_runtime import TorchQDropRuntime
from .specs.base import resolve_qdrop_layer_specs
from .types import QDropConfig


class QDropSpecFactory:
    @staticmethod
    def resolve(quantum_layers):
        return resolve_qdrop_layer_specs(quantum_layers)


class QDropRuntimeFactory:
    @staticmethod
    def create_torch(quantum_layers, config: QDropConfig) -> TorchQDropRuntime:
        return TorchQDropRuntime(QDropSpecFactory.resolve(quantum_layers), config)

    @staticmethod
    def create_tensorflow(quantum_layers, config: QDropConfig) -> TensorFlowQDropRuntime:
        return TensorFlowQDropRuntime(QDropSpecFactory.resolve(quantum_layers), config)
