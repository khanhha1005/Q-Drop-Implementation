"""Framework-specific Q-Drop runtimes."""

from .tensorflow_runtime import TensorFlowQDropRuntime
from .torch_runtime import TorchQDropRuntime

__all__ = ["TensorFlowQDropRuntime", "TorchQDropRuntime"]
