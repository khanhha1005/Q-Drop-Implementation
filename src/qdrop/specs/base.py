"""Base spec resolution helpers."""

from __future__ import annotations

from typing import Iterable, List

from ..types import QDropLayerSpec


def resolve_qdrop_layer_specs(quantum_layers: Iterable[object]) -> List[QDropLayerSpec]:
    resolved_specs: List[QDropLayerSpec] = []
    for layer in quantum_layers:
        if isinstance(layer, QDropLayerSpec):
            resolved_specs.append(layer)
            continue

        if not hasattr(layer, "qdrop_layer_spec"):
            raise TypeError(
                f"Q-Drop layer '{layer}' must be a QDropLayerSpec or expose qdrop_layer_spec()."
            )

        resolved_specs.append(layer.qdrop_layer_spec())

    return resolved_specs
