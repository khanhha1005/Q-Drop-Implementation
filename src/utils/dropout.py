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

# =============================================================================
# QuantumDynamicDropout
# This class encapsulates gradient sanitization and quantum dropout.
# =============================================================================
class QuantumDynamicDropout:
    def __init__(self, 
                 quantum_weights : tf.Variables, 
                 theta_wire_0, theta_wire_1, n_drop, drop_flag):
        """
        Args:
            quantum_weights: The trainable quantum weight variable.
            theta_wire_0: A tensor mask for dropping parameters on wire 0.
            theta_wire_1: A tensor mask for dropping parameters on wire 1.
            n_drop: A tf.constant indicating how many wires to drop (e.g., 1 or 2).
            drop_flag: A tf.Variable (bool) that turns dropout on/off.
        """
        self.quantum_weights = quantum_weights
        self.theta_wire_0 = theta_wire_0
        self.theta_wire_1 = theta_wire_1
        self.n_drop = n_drop
        self.drop_flag = drop_flag

    @tf.function
    def sanitize_gradients(self, gradients):
        """
        Replace any NaN values in gradients with zeros.
        """
        sanitized = []
        for grad in gradients:
            if grad is not None:
                sanitized.append(tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad))
            else:
                sanitized.append(grad)
        return sanitized

    @tf.function
    def apply_dropout(self, gradients, trainable_variables):
        """
        Applies the quantum dropout mask to the gradient corresponding to the
        quantum weights. This method locates the quantum weight gradient in the
        gradients list (by comparing references) and then applies one of the
        dropout strategies.
        """
        # Find the index corresponding to the quantum weights variable.
        quantum_index = None
        for i, var in enumerate(trainable_variables):
            if var is self.quantum_weights:
                quantum_index = i
                break
        if quantum_index is None:
            raise ValueError("Quantum weights variable not found in trainable_variables.")

        quantum_grad = gradients[quantum_index]

        # Define the dropout functions:
        def one_wire_drop():
            # For one-wire dropout, set to zero those elements where theta_wire_0 is 1.
            return tf.where(self.theta_wire_0 == 1, 0.0, quantum_grad)

        def two_wire_drop():
            # For two-wire dropout, apply both masks sequentially.
            dropped = tf.where(self.theta_wire_0 == 1, 0.0, quantum_grad)
            return tf.where(self.theta_wire_1 == 1, 0.0, dropped)

        # Use nested tf.cond to choose which dropout to apply.
        def dropout_fn():
            return tf.cond(
                tf.equal(self.n_drop, 1),
                lambda: one_wire_drop(),
                lambda: tf.cond(
                    tf.equal(self.n_drop, 2),
                    two_wire_drop,
                    lambda: quantum_grad  # If n_drop is not 1 or 2, leave unchanged.
                )
            )

        new_quantum_grad = tf.cond(self.drop_flag, dropout_fn, lambda: quantum_grad)
        new_gradients = list(gradients)
        new_gradients[quantum_index] = new_quantum_grad
        return new_gradients

# =============================================================================
# HybridModel Definition with QuantumDynamicDropout
# =============================================================================
class HybridModel(tf.keras.Model):
    def __init__(self, apply_quantum_dropout):
        super(HybridModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(6, activation='linear', dtype=tf.float64)
        self.quantum_weights2 = self.add_weight(
            shape=(15,),
            initializer='zeros',
            trainable=True,
            dtype=tf.float32
        )
        self.theta_locked = self.add_weight(
            shape=(15,),
            initializer='zeros',
            trainable=False,
            dtype=tf.float32
        )
        
        # Quantum device and circuit definition.
        self.dev = qml.device('default.qubit.tf', wires=6)

        @qml.qnode(self.dev, interface='tf', diff_method='backprop')
        def quantum_circuit(inputs, weights):
            inputs = tf.cast(inputs, tf.float32)
            weights = tf.cast(weights, tf.float32)
            vector_loader(convert_array(inputs), wires=range(6))
            pyramid_circuit(weights, wires=range(6))
            return [qml.expval(qml.PauliZ(wire)) for wire in range(6)]
        self.quantum_circuit = quantum_circuit
        
        self.classical_nn_2 = tf.keras.layers.Dense(2, activation='sigmoid', dtype=tf.float64)
        
        # Define dropout masks.
        self.theta_wire_0 = tf.constant([1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1], dtype=tf.int32)
        self.theta_wire_1 = tf.constant([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=tf.int32)
        self.n_drop = tf.constant(1, dtype=tf.int32)  # Change to 2 for two-wire dropout.
        self.drop_flag = tf.Variable(apply_quantum_dropout, trainable=False)
        
        # Instantiate the dropout manager with the quantum parameters.
        self.dropout_manager = QuantumDynamicDropout(
            quantum_weights=self.quantum_weights2,
            theta_wire_0=self.theta_wire_0,
            theta_wire_1=self.theta_wire_1,
            n_drop=self.n_drop,
            drop_flag=self.drop_flag
        )

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float64)
        flattened_inputs = self.flatten(inputs)
        classical_output = self.dense(flattened_inputs)
        quantum_outputs = tf.map_fn(
            lambda x: tf.stack(self.quantum_circuit(x, self.quantum_weights2)),
            classical_output,
            fn_output_signature=tf.TensorSpec(shape=(6,), dtype=tf.float64)
        )
        # Optionally zero out the first wire if dropout is active.
        quantum_outputs = tf.cond(
            self.drop_flag,
            lambda: tf.concat([
                tf.zeros((tf.shape(quantum_outputs)[0], 1), dtype=tf.float64),
                quantum_outputs[:, 1:]
            ], axis=1),
            lambda: quantum_outputs
        )
        quantum_outputs = tf.where(tf.math.is_nan(quantum_outputs),
                                   tf.zeros_like(quantum_outputs),
                                   quantum_outputs)
        quantum_outputs = tf.reshape(quantum_outputs, [-1, 6])
        nn_output = self.classical_nn_2(quantum_outputs)
        return nn_output

    @tf.function
    def train_step(self, data):
        x, y = data  # Unpack the data

        # "Lock" the current quantum weights (this could be useful for debugging or further processing).
        self.theta_locked.assign(tf.identity(self.quantum_weights2))

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Compute gradients for all trainable variables.
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Sanitize gradients and apply quantum dynamic dropout.
        sanitized_gradients = self.dropout_manager.sanitize_gradients(gradients)
        final_gradients = self.dropout_manager.apply_dropout(sanitized_gradients, self.trainable_variables)
        
        # Apply the processed gradients.
        self.optimizer.apply_gradients(zip(final_gradients, self.trainable_variables))
        
        # Sanitize model variables: replace any NaNs with zeros.
        for var in self.trainable_variables:
            var.assign(tf.where(tf.math.is_nan(var), tf.zeros_like(var), var))
        
        # Update metrics and return the metric results.
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
