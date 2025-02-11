# Q-Drop: Optimizing Quantum Orthogonal Networks with Statistical Pruning and Dynamic Dropout

This repository contains the source code and experimental setups for the project *Q-Drop*, where we investigate and develop algorithms for quantum machine learning inspired by classical dropout and pruning technique. Many quantum principles-such as superposition, entanglement, and quantum parallelism can be applied to optimize Neural Network, in this research we specifically focusing on Quantum Orthogonal Neural Network with our new proposed methods: schuduled gradients pruning and dynamic quantum dropout.

Initially, we run simulations via Pennylane and tensorflow since real quantum machine is difficult to achieve. Further, we aim to migrate and fine-tune the algorithms in order to fit the real quantum machine. 
## Repository Structure
```
Q-Drop/
├── data/*
├── notebooks/
│   ├── experiment_pruning/*
│   └── experiment_dropout/*
├── src/
│   ├── main.py
│   ├── utils/
│   │   ├── rbs_gate.py
│   │   ├── scheduled_pruning.py  
│   │   ├── dynamic_dropout.py   
│   │   └── __init__.py
│   └── models/
│       ├── orthogonal_nn.py
│       └── __init__.py
├── tmp/*
├── .gitignore
├── README.md
└── penny_env.yml
```

- **`experiment_pruning/`**  
  Contains experimental setups and Jupyter notebooks that explore the backpropagation and the use of quantum gradient pruning technique on different datasets. Each subfolder includes tests on the following datasets:
  - MNIST (2-class)
  - Fashion MNIST (2-class)
  - Pneumonia MedMNIST
  - Retina MedMNIST (2-class)

- **`experiment_dropout/`**  
  Contains the experimental setup that focuses on dynamic dropout applied to quantum orthogonal networks. Each experiment mirrors the dataset structure mentioned above.

- **`README.md`**  
  Initial setup and documentation for navigating the project.

## Getting Started

1. Clone the repository:
  ```
   git clone https://github.com/khanhha1005/Q-Drop-Implementation.git
  ```
2. Create conda environment:
  ```
  conda env create -f penny_env.yml
  ```
3. Activate environment:
  ```
  conda activate Penny2
  ```
## Citation
```

```