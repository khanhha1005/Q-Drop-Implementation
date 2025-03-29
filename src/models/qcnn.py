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

class QCNNModel(tf.keras.Model):
    """
    Quantum Convolutional Neural Network (QCNN) implementation with optional pruning.
    Combines ideas from the QCNN implementation with the orthogonal network approach.
    """
    def __init__(self, 
                 random_seed=42, 
                 algorithm=None, 
                 algorithm_params=None,
                 n_qubits=6,
                 n_layers=2,
                 embedding_type='angle',  # 'angle' or 'amplitude'
                ):
        super(QCNNModel, self).__init__()
        
        # Set seeds for reproducibility
        rd.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        qml.numpy.random.seed(random_seed)
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding_type = embedding_type
        
        # Classical preprocessing layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense_preprocess = tf.keras.layers.Dense(n_qubits, activation='linear', dtype=tf.float64)
        
        # QCNN parameter count:
        # - Convolutional parameters: 3 parameters per pair of qubits per layer
        # - Pooling parameters: 3 parameters per qubit 
        total_conv_params = 3 * (n_qubits // 2) * n_layers
        total_pool_params = 3 * (n_qubits // 2)
        total_params = total_conv_params + total_pool_params
        
        # Quantum circuit parameters
        self.quantum_weights = self.add_weight(
            shape=(total_params,),
            initializer="random_normal",
            trainable=True,
            dtype=tf.float64
        )
        
        # Quantum device
        self.dev = qml.device('default.qubit.tf', wires=n_qubits)
        
        # Build the QCNN quantum node
        @qml.qnode(self.dev, interface='tf', diff_method='backprop')
        def qcnn_circuit(inputs, weights):
            inputs = tf.cast(inputs, tf.float32)
            weights = tf.cast(weights, tf.float32)
            
            # Data embedding
            self._embed_data(inputs)
            
            # Convolutional layers
            weight_idx = 0
            for l in range(self.n_layers):
                # Apply convolutional layer
                for i in range(0, self.n_qubits, 2):
                    if i + 1 < self.n_qubits:
                        # Two-qubit gate for convolution
                        qml.RY(weights[weight_idx], wires=i)
                        weight_idx += 1
                        qml.RY(weights[weight_idx], wires=i+1)
                        weight_idx += 1
                        qml.CNOT(wires=[i, i+1])
                        qml.RY(weights[weight_idx], wires=i+1)
                        weight_idx += 1
                        qml.CNOT(wires=[i, i+1])
            
            # Pooling layer 
            for i in range(0, self.n_qubits, 2):
                if i + 1 < self.n_qubits:
                    # Two-qubit operations for pooling
                    qml.RY(weights[weight_idx], wires=i)
                    weight_idx += 1
                    qml.RY(weights[weight_idx], wires=i+1)
                    weight_idx += 1
                    qml.CNOT(wires=[i, i+1])
                    qml.RY(weights[weight_idx], wires=i+1)
                    weight_idx += 1
                    qml.CNOT(wires=[i, i+1])
            
            # Measure first half of qubits for the pooled result
            return [qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits // 2)]
        
        self.qcnn_circuit = qcnn_circuit
        
        # Output classification layer
        self.classical_output = tf.keras.layers.Dense(1, activation='sigmoid', dtype=tf.float64)
        
        # Initialize pruning algorithm if specified
        if algorithm == 'pruning' and algorithm_params is not None:
            self.algorithm = ScheduledGradientPruning(
                self.quantum_weights, 
                accumulate_window=algorithm_params.get('accumulate_window', 100), 
                prune_window=algorithm_params.get('prune_window', 200), 
                prune_ratio=algorithm_params.get('prune_ratio', 0.5), 
                seed=random_seed,
                dtype=tf.float64,
                schedule=algorithm_params.get('schedule', False)
            )
        else:
            self.algorithm = None

    def _embed_data(self, inputs):
        """Embed classical data into quantum states"""
        if self.embedding_type == 'angle':
            # Angle embedding
            for i in range(self.n_qubits):
                qml.RY(inputs[i % len(inputs)], wires=i)
        else:
            # Amplitude embedding - normalize first
            inputs_normalized = inputs / tf.norm(inputs)
            # Pad if necessary
            padding_size = 2**self.n_qubits - tf.shape(inputs_normalized)[0]
            if padding_size > 0:
                inputs_padded = tf.pad(inputs_normalized, [[0, padding_size]])
                qml.AmplitudeEmbedding(inputs_padded, wires=range(self.n_qubits), normalize=True)
            else:
                # Truncate if input is too long
                qml.AmplitudeEmbedding(inputs_normalized[:2**self.n_qubits], 
                                      wires=range(self.n_qubits), normalize=True)

    def call(self, inputs, training=None):
        inputs = tf.cast(inputs, tf.float64)
        flattened_inputs = self.flatten(inputs)
        preprocessed = self.dense_preprocess(flattened_inputs)
        
        # Run quantum circuit for each preprocessed input
        quantum_outputs = tf.map_fn(
            lambda x: tf.stack(self.qcnn_circuit(x, self.quantum_weights)),
            preprocessed,
            fn_output_signature=tf.TensorSpec(shape=(self.n_qubits // 2,), dtype=tf.float64)
        )
        
        # Replace NaN values with zeros
        quantum_outputs = tf.where(tf.math.is_nan(quantum_outputs), 
                                   tf.zeros_like(quantum_outputs), quantum_outputs)
        
        # Reshape and pass through classical output layer
        quantum_outputs = tf.reshape(quantum_outputs, [-1, self.n_qubits // 2])
        output = self.classical_output(quantum_outputs)
        return output

    @tf.function
    def train_step(self, data):
        x, y = data  # Unpack the data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Apply pruning if algorithm is specified
        if self.algorithm is not None:
            # Locate quantum_weights gradient by name matching
            quantum_grad = None
            for idx, var in enumerate(self.trainable_variables):
                if var.name == self.quantum_weights.name:
                    quantum_grad = gradients[idx]
                    break
            if quantum_grad is None:
                raise ValueError("Quantum weights not found in trainable_variables")
            
            # Use pruning algorithm
            self.algorithm.apply(quantum_grad, self.optimizer, gradients, self.trainable_variables)
        else:
            # Apply gradients normally without pruning
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Sanitize weights: Replace any NaNs with zeros
        for var in self.trainable_variables:
            sanitized_var = tf.where(tf.math.is_nan(var), tf.zeros_like(var), var)
            var.assign(sanitized_var)
        
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics} 