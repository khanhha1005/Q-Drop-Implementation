"""PennyLane TensorFlow spec helpers."""

from __future__ import annotations

from ..types import QDropLayerSpec, QDropTensorSpec


class TensorFlowQuantumTensorAdapter:
    def __init__(
        self,
        layer_id,
        parameter,
        num_wires,
        *,
        tensor_id="quantum_weights",
        wire_masks=None,
        mask_builder=None,
        set_forward_mask=None,
        supports_forward_mask=False,
    ):
        self._layer_id = layer_id
        self._parameter = parameter
        self._num_wires = num_wires
        self._tensor_id = tensor_id
        self._wire_masks = wire_masks
        self._mask_builder = mask_builder
        self._set_forward_mask = set_forward_mask
        self._supports_forward_mask = supports_forward_mask

    def qdrop_layer_spec(self) -> QDropLayerSpec:
        return QDropLayerSpec(
            layer_id=self._layer_id,
            tensor_specs=[
                QDropTensorSpec(
                    tensor_id=self._tensor_id,
                    parameter=self._parameter,
                    num_wires=self._num_wires,
                    supports_gradient_mask=True,
                    supports_forward_mask=self._supports_forward_mask,
                    wire_masks=self._wire_masks,
                    mask_builder=self._mask_builder,
                )
            ],
            set_forward_mask=self._set_forward_mask,
        )


class PennyLaneTensorFlowSpecFactory:
    @staticmethod
    def create_adapter(**kwargs) -> TensorFlowQuantumTensorAdapter:
        return TensorFlowQuantumTensorAdapter(**kwargs)
