"""Simple Q-Drop session with dummy pruning and epoch dropout state."""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, Tuple

from .core import QDropUnit
from .types import QDropConfig, QDropDropoutState, QDropLayerSpec


class QDropSession:
    """Coordinate many tensor units with shared phase/dropout state."""

    def __init__(self, layer_specs: Iterable[QDropLayerSpec], config: QDropConfig):
        self.config = config
        self.layer_specs = list(layer_specs)
        self.units: Dict[Tuple[str, str], QDropUnit] = {}
        self.active_dropout_states: Dict[str, QDropDropoutState] = {}

        for layer_spec in self.layer_specs:
            for tensor_spec in layer_spec.tensor_specs:
                unit = QDropUnit(tensor_spec)
                unit.prune_ratio = float(config.prune_ratio)
                self.units[(layer_spec.layer_id, tensor_spec.tensor_id)] = unit

        self.accumulate_phase = True
        self.accumulate_count = config.accumulate_window
        self.prune_count = config.prune_window
        self.current_prune_ratio = float(config.prune_ratio)
        self.pruning_step_count = 0
        self.current_epoch = 0
        self.dropout_enabled = False
        self._active_step_mode = "passthrough"

    @property
    def quantum_param_count(self) -> int:
        return len(self.units)

    def clear_forward_masks(self) -> None:
        self.dropout_enabled = False
        self.active_dropout_states = {}
        for layer_spec in self.layer_specs:
            if layer_spec.set_forward_mask is not None:
                layer_spec.set_forward_mask(None)

    def _sample_dropout_state(self, layer_spec: QDropLayerSpec) -> QDropDropoutState | None:
        num_wires = max((tensor_spec.num_wires for tensor_spec in layer_spec.tensor_specs), default=0)
        if num_wires <= 0:
            return None

        n_drop = max(1, min(self.config.n_drop_wires, num_wires))
        dropped_wires = tuple(sorted(random.sample(range(num_wires), k=n_drop)))
        return QDropDropoutState(enabled=True, dropped_wires=dropped_wires)

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        self.clear_forward_masks()

        if self.config.algorithm not in {"dropout", "both"}:
            return
        if epoch <= 1:
            return
        if random.random() >= self.config.dropout_prob:
            return

        self.dropout_enabled = True
        for layer_spec in self.layer_specs:
            dropout_state = self._sample_dropout_state(layer_spec)
            if dropout_state is None:
                continue
            self.active_dropout_states[layer_spec.layer_id] = dropout_state
            if self.config.enable_forward_mask and layer_spec.set_forward_mask is not None:
                layer_spec.set_forward_mask(dropout_state)

    def _update_phase(self) -> None:
        if self.accumulate_count == 0:
            self.accumulate_count = self.config.accumulate_window
            self.accumulate_phase = False
        elif self.prune_count == 0:
            self.prune_count = self.config.prune_window
            self.accumulate_phase = True

    def begin_train_step(self) -> None:
        self._active_step_mode = "passthrough"

        if self.config.algorithm not in {"pruning", "both"}:
            return

        self._update_phase()
        if self.accumulate_phase:
            self.accumulate_count -= 1
            self._active_step_mode = "accumulate"
            return

        self.prune_count -= 1
        self._active_step_mode = "prune"

    def process_tensor_grad(self, layer_id: str, tensor_id: str, grad):
        unit = self.units[(layer_id, tensor_id)]
        result = grad

        if self.config.sanitize_quantum_gradients:
            result = unit.sanitize_gradient(result)

        if self.config.algorithm in {"dropout", "both"}:
            dropout_state = self.active_dropout_states.get(layer_id)
            if dropout_state is not None and dropout_state.enabled:
                drop_mask = unit.build_dropout_mask(dropout_state)
                result = unit.apply_gradient_mask(result, drop_mask, keep_mask=False)

        if self._active_step_mode == "accumulate":
            unit.accumulate(result)
            return result

        if self._active_step_mode == "prune":
            unit.prune_ratio = self.current_prune_ratio
            return unit.build_pruned_gradient()

        return result

    def end_train_step(self) -> None:
        if self._active_step_mode != "prune":
            return

        for unit in self.units.values():
            unit.reset_accumulated_grad()

        self.pruning_step_count += 1
        if self.config.schedule and self.pruning_step_count % 5 == 0:
            self.current_prune_ratio = min(1.0, self.current_prune_ratio * math.exp(0.1))

    def after_optimizer_step(self) -> None:
        if not self.config.sanitize_quantum_parameters:
            return

        for unit in self.units.values():
            unit.sanitize_tensor()
