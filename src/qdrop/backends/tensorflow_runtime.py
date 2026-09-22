"""Thin TensorFlow runtime for Q-Drop sessions."""

from __future__ import annotations

from typing import Iterable, List

from ..session import QDropSession
from ..types import QDropConfig


def _variable_key(variable) -> object:
    if hasattr(variable, "ref"):
        return variable.ref()
    return id(variable)


class TensorFlowQDropRuntime:
    def __init__(self, layer_specs: Iterable, config: QDropConfig):
        self.layer_specs = list(layer_specs)
        self.config = config
        self.session = QDropSession(self.layer_specs, config)
        self._variable_map = {}
        for layer_spec in self.layer_specs:
            for tensor_spec in layer_spec.tensor_specs:
                self._variable_map[_variable_key(tensor_spec.parameter)] = (
                    layer_spec.layer_id,
                    tensor_spec.tensor_id,
                )

    @property
    def quantum_param_count(self) -> int:
        return self.session.quantum_param_count

    def start_epoch(self, epoch: int) -> None:
        self.session.start_epoch(epoch)

    def process_gradients(self, gradients, variables) -> List:
        self.session.begin_train_step()
        processed_gradients = []
        for grad, variable in zip(gradients, variables):
            spec_key = self._variable_map.get(_variable_key(variable))
            if spec_key is None or grad is None:
                processed_gradients.append(grad)
                continue

            layer_id, tensor_id = spec_key
            processed_gradients.append(self.session.process_tensor_grad(layer_id, tensor_id, grad))

        self.session.end_train_step()
        return processed_gradients

    def after_step(self) -> None:
        self.session.after_optimizer_step()

    def clear_forward_masks(self) -> None:
        self.session.clear_forward_masks()
