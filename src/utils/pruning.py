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
# New Class: ScheduledGradientPruning
# This class encapsulates the gradient accumulation and pruning algorithm
# =============================================================================
class ScheduledGradientPruning:
    def __init__(self, quantum_weights, accumulate_window=10, prune_window=8, 
                 prune_ratio=0.8, seed=42, dtype=tf.float64):
        # Quantum weights variable reference
        self.quantum_weights = quantum_weights
        self.dtype = dtype

        # Initialize the accumulated gradient variable (non-trainable)
        self.accumulated_grads = tf.Variable(tf.zeros_like(quantum_weights),
                                             trainable=False, dtype=dtype)
        # Boolean flag: True for accumulation phase, False for pruning phase
        self.accumulate_flag = tf.Variable(True, trainable=False)
        
        # Window lengths for accumulation and pruning phases
        self.accumulate_window = tf.constant(accumulate_window)
        self.prune_window = tf.constant(prune_window)
        
        # Pruning ratio (can be updated dynamically)
        self.prune_ratio = tf.Variable(prune_ratio, trainable=False, dtype=dtype)
        
        # Counters for how many steps remain in each phase
        self.accumulate_count = tf.Variable(accumulate_window, dtype=tf.int32, trainable=False)
        self.prune_count = tf.Variable(prune_window, dtype=tf.int32, trainable=False)
        
        # Set seeds for reproducibility (if desired)
        rd.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        qml.numpy.random.seed(seed)

    @tf.function
    def update_phase(self):
        """Update the phase flags and counters based on current counts."""
        # Switch phases if counter reaches zero
        if self.accumulate_count == 0:
            self.accumulate_count.assign(self.accumulate_window)
            self.accumulate_flag.assign(False)
        elif self.prune_count == 0:
            self.prune_count.assign(self.prune_window)
            self.accumulate_flag.assign(True)

    @tf.function
    def apply(self, quantum_grad, optimizer, gradients, trainable_variables):
        """
        Applies the accumulation or pruning algorithm based on the phase.
        This function must be called inside a tf.GradientTape context.
        """
        # Update phase flags based on counters
        self.update_phase()

        if tf.equal(self.accumulate_flag, True):
            # ---------------------------
            # Accumulation Phase
            # ---------------------------
            self.accumulate_count.assign_sub(1)
            if quantum_grad is not None:
                self.accumulated_grads.assign_add(quantum_grad)
            # Apply the gradients normally for all variables
            optimizer.apply_gradients(zip(gradients, trainable_variables))
        else:
            # ---------------------------
            # Pruning Phase
            # ---------------------------
            self.prune_count.assign_sub(1)
            epsilon = tf.constant(1e-8, dtype=self.dtype)
            grad_min = tf.reduce_min(self.accumulated_grads)
            grad_max = tf.reduce_max(self.accumulated_grads)
            # Normalize the accumulated gradients
            norm_grads = (self.accumulated_grads - grad_min) / (grad_max - grad_min + epsilon)
            norm_grads_with_epsilon = norm_grads + epsilon
            logits = tf.math.log(norm_grads_with_epsilon)

            num_params = tf.shape(self.quantum_weights)[0]
            # Compute number of parameters to sample based on prune_ratio
            num_samples = tf.maximum(1, tf.cast(self.prune_ratio * tf.cast(num_params, self.dtype), tf.int32))
            
            # Draw random indices according to the logits as sampling weights
            indices = tf.random.categorical(tf.expand_dims(logits, 0), num_samples=num_samples)
            indices = tf.clip_by_value(indices, 0, self.quantum_weights.shape[0] - 1)
            indices = tf.cast(tf.reshape(indices, [-1, 1]), tf.int32)
            
            # Create a mask that is True for selected indices and False elsewhere
            mask = tf.zeros_like(self.quantum_weights, dtype=tf.bool)
            updates = tf.ones([tf.shape(indices)[0]], dtype=tf.bool)
            mask = tf.tensor_scatter_nd_update(mask, indices, updates)
            
            # Use the mask to prune (zero out) gradients that are not selected
            pruned_grad = tf.where(mask, self.accumulated_grads, tf.zeros_like(self.accumulated_grads))
            
            # Apply the pruned gradient only to the quantum weights
            optimizer.apply_gradients([(pruned_grad, self.quantum_weights)])
            
            # Apply gradients for all other variables
            other_gradients = []
            other_variables = []
            for grad, var in zip(gradients, trainable_variables):
                if var is not self.quantum_weights and grad is not None:
                    other_gradients.append(grad)
                    other_variables.append(var)
            optimizer.apply_gradients(zip(other_gradients, other_variables))
            
            # Reset the accumulator after pruning
            self.accumulated_grads.assign(tf.zeros_like(self.accumulated_grads))

# =============================================================================
# Custom Callback for Prune Ratio Scheduling (optional)
# =============================================================================
global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

class PruneScheduler(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        global_step.assign(0)
    def on_train_batch_end(self, batch, logs=None):
        global_step.assign_add(1)
        current_step = global_step.numpy()
        if current_step % 5 == 0:
            # Increase the pruning ratio dynamically, capped at 1.0
            current_prune_ratio = self.model.quantum_grad_manager.prune_ratio
            new_prune_ratio = min(current_prune_ratio * math.exp(0.1), 1.0)
            self.model.quantum_grad_manager.prune_ratio.assign(new_prune_ratio)
            tf.print("Updated prune_ratio to", new_prune_ratio)