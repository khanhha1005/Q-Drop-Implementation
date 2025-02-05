# Q-Drop: Optimizing Quantum Orthogonal Networks with Statistical Pruning and Dynamic Dropout

This repository contains the code and experimental setups for the project *Q-Drop*, where we investigate how quantum principles—such as superposition, entanglement, and quantum parallelism—can be applied to optimize an Orthogonal Neural Network, specifically focusing on techniques like gradient pruning and dynamic dropout.

## Repository Structure

- **`experiment_pruning/`**  
  Contains experimental setups and Jupyter notebooks that explore the backpropagation and the use of quantum gradient pruning technique on different datasets. Each subfolder includes tests on the following datasets:
  - MNIST (2-class)
  - Fashion MNIST (2-class)
  - Pneumonia MedMNIST
  - Retina MedMNIST (2-class)

- **`experiment_dropout/`**  
  Contains the experimental setup that focuses on dynamic dropout applied to quantum orthogonal networks. Each experiment mirrors the dataset structure mentioned above.

- **`QuantumOrthoNN_Model_Pennylane_ipynb.ipynb`**  
  This notebook contains the core implementation of Quantum Orthogonal Networks using the Pennylane framework. It includes the model definitions, training procedures, and evaluation metrics.

- **`README.md`**  
  Initial setup and documentation for navigating the project.

## Getting Started

1. Clone the repository:
  ```
   git clone https://github.com/yourusername/Q-Drop.git
  ```
2. Create conda environment:
  ```
  conda env create -f penny_env.yml
  ```
## Citation
```

```