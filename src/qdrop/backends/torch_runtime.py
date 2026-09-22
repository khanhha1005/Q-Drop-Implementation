"""Thin Torch runtime for Q-Drop sessions."""

from __future__ import annotations

from typing import Iterable

from ..session import QDropSession
from ..types import QDropConfig


class TorchQDropRuntime:
    def __init__(self, layer_specs: Iterable, config: QDropConfig):
        self.layer_specs = list(layer_specs)
        self.config = config
        self.session = QDropSession(self.layer_specs, config)

    @property
    def quantum_param_count(self) -> int:
        return self.session.quantum_param_count

    @property
    def active_dropout_states(self):
        return self.session.active_dropout_states

    @property
    def dropout_enabled(self) -> bool:
        return self.session.dropout_enabled

    @property
    def accumulate_phase(self) -> bool:
        return self.session.accumulate_phase

    @property
    def pruning_step_count(self) -> int:
        return self.session.pruning_step_count

    @property
    def current_prune_ratio(self) -> float:
        return self.session.current_prune_ratio

    def start_epoch(self, epoch: int) -> None:
        self.session.start_epoch(epoch)

    def after_backward(self) -> None:
        self.session.begin_train_step()
        for layer_spec in self.layer_specs:
            for tensor_spec in layer_spec.tensor_specs:
                if tensor_spec.parameter.grad is None:
                    continue
                tensor_spec.parameter.grad = self.session.process_tensor_grad(
                    layer_spec.layer_id,
                    tensor_spec.tensor_id,
                    tensor_spec.parameter.grad,
                )
        self.session.end_train_step()

    def after_step(self) -> None:
        self.session.after_optimizer_step()

    def clear_forward_masks(self) -> None:
        self.session.clear_forward_masks()

    def apply(self) -> None:
        self.after_backward()

    def sanitize_parameters(self) -> None:
        self.after_step()
