import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # This silences INFO and WARNING messages
import warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import pennylane as qml
from utils.rbs_gate import *
import random as rd

from qdrop import QDropConfig, QDropRuntimeFactory
from qdrop.specs.pennylane_tf import PennyLaneTensorFlowSpecFactory

# =============================================================================
# HybridModel Definition with OOP Train Step
# =============================================================================
class HybridModel(tf.keras.Model):
    def __init__(self, random :int, algorithm :str, algorithm_params :dict):
        super(HybridModel, self).__init__()
        
        # Set seeds for reproducibility
        rd.seed(random)
        np.random.seed(random)
        tf.random.set_seed(random)
        qml.numpy.random.seed(random)
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(6, activation='linear', dtype=tf.float64)
        self.quantum_weights = self.add_weight(
            shape=(15,),
            initializer="zeros",
            trainable=True,
            dtype=tf.float64
        )
        
        # Quantum device with 6 wires
        self.dev = qml.device('default.qubit.tf', wires=6)
        
        # Build the quantum node (QNode)
        @qml.qnode(self.dev, interface='tf', diff_method='backprop')
        def quantum_circuit(inputs, weights):
            inputs = tf.cast(inputs, tf.float32)
            weights = tf.cast(weights, tf.float32)
            vector_loader(convert_array(inputs), wires=range(6))
            pyramid_circuit(weights, wires=range(6))
            return [qml.expval(qml.PauliZ(wire)) for wire in range(6)]
        self.quantum_circuit = quantum_circuit
        self.qdrop_forward_mask = tf.Variable(tf.ones((6,), dtype=tf.float64), trainable=False)

        # Additional classical NN layer
        self.classical_nn_2 = tf.keras.layers.Dense(1, activation='sigmoid', dtype=tf.float64)
        self.qdrop_quantum_layer = PennyLaneTensorFlowSpecFactory.create_adapter(
            layer_id="orthogonal_quantum_layer",
            parameter=self.quantum_weights,
            num_wires=6,
            mask_builder=self._build_qdrop_mask,
            set_forward_mask=self._set_qdrop_forward_mask,
            supports_forward_mask=True,
        )
        self.qdrop_runtime = None
        if algorithm in {'pruning', 'dropout', 'both'}:
            self.qdrop_runtime = QDropRuntimeFactory.create_tensorflow(
                quantum_layers=self.qdrop_layers(),
                config=QDropConfig(
                    algorithm=algorithm,
                    accumulate_window=algorithm_params['accumulate_window'],
                    prune_window=algorithm_params['prune_window'],
                    prune_ratio=algorithm_params['prune_ratio'],
                    schedule=algorithm_params['schedule'],
                ),
            )

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float64)
        flattened_inputs = self.flatten(inputs)
        classical_output = self.dense(flattened_inputs)
        
        # Run quantum circuit for each output from the dense layer
        quantum_outputs = tf.map_fn(
            lambda x: tf.stack(self.quantum_circuit(x, self.quantum_weights)),
            classical_output,
            fn_output_signature=tf.TensorSpec(shape=(6,), dtype=tf.float64)
        )
        quantum_outputs = quantum_outputs * self.qdrop_forward_mask
        # Replace any NaN values with zeros
        quantum_outputs = tf.where(tf.math.is_nan(quantum_outputs), 
                                   tf.zeros_like(quantum_outputs), quantum_outputs)
        quantum_outputs = tf.reshape(quantum_outputs, [-1, 6])
        nn_output = self.classical_nn_2(quantum_outputs)
        return nn_output

    def _build_qdrop_mask(self, wire_ids):        
        indices = [wire_id for wire_id in wire_ids if 0 <= wire_id < int(self.quantum_weights.shape[0])]
        updates = tf.ones((len(indices),), dtype=tf.bool)
        mask = tf.zeros_like(self.quantum_weights, dtype=tf.bool)
        if not indices:
            return mask
        scatter_indices = tf.constant([[wire_id] for wire_id in indices], dtype=tf.int32)
        return tf.tensor_scatter_nd_update(mask, scatter_indices, updates)

    def _set_qdrop_forward_mask(self, dropout_state):
        mask = np.ones((6,), dtype=np.float64)
        if dropout_state is not None and dropout_state.enabled:
            for wire_id in dropout_state.dropped_wires:
                if 0 <= wire_id < 6:
                    mask[wire_id] = 0.0
        self.qdrop_forward_mask.assign(mask)

    def qdrop_layers(self):
        return [self.qdrop_quantum_layer]

    def train_step(self, data):
        x, y = data  # Unpack the data

        if self.qdrop_runtime is not None and self.optimizer is not None:
            self.qdrop_runtime.start_epoch(int(self.optimizer.iterations.numpy()) + 1)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Compute gradients for all trainable variables
        gradients = tape.gradient(loss, self.trainable_variables)

        if self.qdrop_runtime is not None:
            gradients = self.qdrop_runtime.process_gradients(gradients, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        if self.qdrop_runtime is not None:
            self.qdrop_runtime.after_step()
        
        # Sanitize all weights: Replace any NaNs with zeros
        for var in self.trainable_variables:
            sanitized_var = tf.where(tf.math.is_nan(var), tf.zeros_like(var), var)
            var.assign(sanitized_var)
        
        # Update metrics and return a dictionary mapping metric names to current values
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
