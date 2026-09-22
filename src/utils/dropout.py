"""Backward-compatible TensorFlow dropout shim over the new Q-Drop unit API."""

from __future__ import annotations

import tensorflow as tf

from qdrop.core import QDropUnit
from qdrop.types import QDropDropoutState, QDropTensorSpec


class QuantumDynamicDropoutManager:
    def __init__(self, quantum_weights, theta_wire_0, theta_wire_1, n_drop, drop_flag):
        self.quantum_weights = quantum_weights
        self.theta_wire_0 = theta_wire_0
        self.theta_wire_1 = theta_wire_1
        self.n_drop = n_drop
        self.drop_flag = drop_flag
        self.unit = QDropUnit(
            QDropTensorSpec(
                tensor_id="legacy_dropout_tensor",
                parameter=quantum_weights,
                num_wires=2,
                wire_masks={
                    0: tf.cast(tf.reshape(theta_wire_0, tf.shape(quantum_weights)), tf.bool),
                    1: tf.cast(tf.reshape(theta_wire_1, tf.shape(quantum_weights)), tf.bool),
                },
            )
        )

    def sanitize_gradients(self, gradients):
        sanitized = []
        for grad in gradients:
            if grad is None:
                sanitized.append(grad)
            else:
                sanitized.append(tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad))
        return sanitized

    def apply_dropout(self, gradients, trainable_variables):
        quantum_index = None
        for index, variable in enumerate(trainable_variables):
            if variable is self.quantum_weights:
                quantum_index = index
                break

        if quantum_index is None:
            raise ValueError("Quantum weights variable not found in trainable_variables.")
        if not bool(self.drop_flag.numpy()):
            return list(gradients)

        n_drop = int(self.n_drop.numpy()) if hasattr(self.n_drop, "numpy") else int(self.n_drop)
        dropped_wires = tuple(range(max(0, min(n_drop, 2))))
        dropout_state = QDropDropoutState(enabled=True, dropped_wires=dropped_wires)
        dropout_mask = self.unit.build_dropout_mask(dropout_state)

        new_gradients = list(gradients)
        quantum_grad = new_gradients[quantum_index]
        if quantum_grad is not None:
            new_gradients[quantum_index] = self.unit.apply_gradient_mask(
                quantum_grad,
                dropout_mask,
                keep_mask=False,
            )
        return new_gradients
