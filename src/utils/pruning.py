"""Backward-compatible TensorFlow pruning shim over the new Q-Drop runtime."""

from __future__ import annotations

import tensorflow as tf

from qdrop import QDropConfig, QDropRuntimeFactory
from qdrop.specs.pennylane_tf import PennyLaneTensorFlowSpecFactory


class ScheduledGradientPruning:
    def __init__(
        self,
        quantum_weights: tf.Variable,
        accumulate_window: int = 10,
        prune_window: int = 8,
        prune_ratio: float = 0.8,
        seed: int = 42,
        dtype: tf.dtypes.DType = tf.float64,
        schedule: bool = False,
    ):
        del seed
        self.quantum_weights = quantum_weights
        self.dtype = dtype
        self.prune_ratio = tf.Variable(prune_ratio, trainable=False, dtype=dtype)
        self.accumulated_grads = tf.Variable(tf.zeros_like(quantum_weights), trainable=False, dtype=dtype)
        self.adapter = PennyLaneTensorFlowSpecFactory.create_adapter(
            layer_id="legacy_pruning_layer",
            parameter=quantum_weights,
            num_wires=max(int(quantum_weights.shape[-1] or 1), 1),
            mask_builder=lambda wire_ids: tf.zeros_like(quantum_weights, dtype=tf.bool),
            supports_forward_mask=False,
        )
        self.runtime = QDropRuntimeFactory.create_tensorflow(
            quantum_layers=[self.adapter],
            config=QDropConfig(
                algorithm="pruning",
                accumulate_window=accumulate_window,
                prune_window=prune_window,
                prune_ratio=prune_ratio,
                schedule=schedule,
            ),
        )

    def apply(self, quantum_grad, optimizer, gradients, trainable_variables):
        del quantum_grad
        processed_gradients = self.runtime.process_gradients(list(gradients), list(trainable_variables))
        optimizer.apply_gradients(zip(processed_gradients, trainable_variables))
        self.runtime.after_step()
        self.prune_ratio.assign(self.runtime.session.current_prune_ratio)
