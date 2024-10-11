# Q-Drop: Optimizing Quantum Orthogonal Networks with Gradient Pruning and Dynamic Dropout

This repository contains the code and experimental setups for the project *Q-Drop*, where we investigate how quantum principles—such as superposition, entanglement, and quantum parallelism—can be applied to optimize vision transformers, specifically focusing on techniques like gradient pruning and dynamic dropout.

## Repository Structure

- **`experiment_backprop_PGP/`**  
  Contains experimental setups and Jupyter notebooks that explore the backpropagation and the use of quantum gradient pruning (PGP) technique on different datasets. Each subfolder includes tests on the following datasets:
  - MNIST (2-class)
  - Fashion MNIST (2-class)a
  - Pneumonia MedMNIST
  - Retina MedMNIST (2-class)

- **`experiment_dropout/`**  
  Contains the experimental setup that focuses on dynamic dropout applied to quantum orthogonal networks. Each experiment mirrors the dataset structure mentioned above.

- **`experiment_pshift_PGP/`**  
  Contains the experimental setups that implement parameter shift rules and the use quantum gradient pruning (PGP) on the above datasets
- **`QuantumOrthoNN_Model_Pennylane_ipynb.ipynb`**  
  This notebook contains the core implementation of Quantum Orthogonal Networks using the Pennylane framework. It includes the model definitions, training procedures, and evaluation metrics.

- **`README.md`**  
  Initial setup and documentation for navigating the project.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Q-Drop.git
