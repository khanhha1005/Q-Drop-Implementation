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
from utils.pruning import ScheduledGradientPruning
import random as rd

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
        
        # Additional classical NN layer
        self.classical_nn_2 = tf.keras.layers.Dense(1, activation='sigmoid', dtype=tf.float64)
        
        # Instantiate the ScheduledGradientPruning for the quantum_weights
        if algorithm == 'pruning':
            self.algorithm = ScheduledGradientPruning(
                self.quantum_weights, 
                accumulate_window=algorithm_params['accumulate_window'], 
                prune_window=algorithm_params['prune_window'], 
                prune_ratio=algorithm_params['prune_ratio'], 
                seed=random,
                dtype=tf.float64
            )
        elif algorithm == 'dropout':
            #TODO: Implement dropout algorithm
            pass

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
        # Replace any NaN values with zeros
        quantum_outputs = tf.where(tf.math.is_nan(quantum_outputs), 
                                   tf.zeros_like(quantum_outputs), quantum_outputs)
        quantum_outputs = tf.reshape(quantum_outputs, [-1, 6])
        nn_output = self.classical_nn_2(quantum_outputs)
        return nn_output

    @tf.function
    def train_step(self, data):
        x, y = data  # Unpack the data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Compute gradients for all trainable variables
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Locate quantum_weights gradient by matching names
        quantum_grad = None
        for idx, var in enumerate(self.trainable_variables):
            if var.name == self.quantum_weights.name:
                quantum_grad = gradients[idx]
                break
        if quantum_grad is None:
            raise ValueError("Quantum weights not found in trainable_variables")
        
        # Use the ScheduledGradientPruning to update quantum weights and apply the rest of the gradients
        self.algorithm.apply(quantum_grad, self.optimizer, gradients, self.trainable_variables)
        
        # Sanitize all weights: Replace any NaNs with zeros
        for var in self.trainable_variables:
            sanitized_var = tf.where(tf.math.is_nan(var), tf.zeros_like(var), var)
            var.assign(sanitized_var)
        
        # Update metrics and return a dictionary mapping metric names to current values
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


