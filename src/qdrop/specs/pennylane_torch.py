"""PennyLane Torch spec factory."""

from __future__ import annotations

from ..types import QDropLayerSpec, QDropTensorSpec


class PennyLaneTorchSpecFactory:
    @staticmethod
    def from_adapter(adapter) -> QDropLayerSpec:
        layer_id = getattr(adapter, "qdrop_name", adapter.__class__.__name__)
        return QDropLayerSpec(
            layer_id=layer_id,
            tensor_specs=[
                QDropTensorSpec(
                    tensor_id="weights",
                    parameter=adapter.quantum_layer.weights,
                    num_wires=adapter.n_qubits,
                    supports_gradient_mask=True,
                    supports_forward_mask=True,
                    mask_builder=adapter.mask_builder,
                )
            ],
            set_forward_mask=adapter.set_forward_mask,
        )
