"""Public Q-Drop types and contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

SUPPORTED_QDROP_ALGORITHMS = {"baseline", "pruning", "dropout", "both"}

MaskBuilder = Callable[[Tuple[int, ...]], Any]
ForwardMaskSetter = Callable[[Optional["QDropDropoutState"]], None]


@dataclass
class QDropConfig:
    algorithm: str = "baseline"
    accumulate_window: int = 10
    prune_window: int = 8
    prune_ratio: float = 0.8
    schedule: bool = True
    dropout_prob: float = 0.5
    n_drop_wires: int = 1
    enable_forward_mask: bool = True
    sanitize_quantum_gradients: bool = True
    sanitize_quantum_parameters: bool = True

    def __post_init__(self) -> None:
        if self.algorithm not in SUPPORTED_QDROP_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{self.algorithm}'. "
                f"Expected one of {sorted(SUPPORTED_QDROP_ALGORITHMS)}."
            )


@dataclass(frozen=True)
class QDropDropoutState:
    enabled: bool
    dropped_wires: Tuple[int, ...] = ()


@dataclass
class QDropTensorSpec:
    tensor_id: str
    parameter: Any
    num_wires: int
    supports_gradient_mask: bool = True
    supports_forward_mask: bool = False
    wire_masks: Optional[Dict[int, Any]] = None
    mask_builder: Optional[MaskBuilder] = None

    def __post_init__(self) -> None:
        if self.num_wires <= 0:
            raise ValueError("QDropTensorSpec requires num_wires > 0.")
        if self.wire_masks is None and self.mask_builder is None:
            raise ValueError(
                f"QDropTensorSpec '{self.tensor_id}' requires either wire_masks or mask_builder."
            )


@dataclass
class QDropLayerSpec:
    layer_id: str
    tensor_specs: list[QDropTensorSpec]
    set_forward_mask: Optional[ForwardMaskSetter] = None

    def __post_init__(self) -> None:
        if not self.layer_id:
            raise ValueError("QDropLayerSpec requires a non-empty layer_id.")
        if not self.tensor_specs:
            raise ValueError("QDropLayerSpec requires at least one tensor spec.")


@runtime_checkable
class SupportsQDropSpec(Protocol):
    def qdrop_layer_spec(self) -> QDropLayerSpec:
        """Return the Q-Drop spec owned by this quantum-capable layer."""
