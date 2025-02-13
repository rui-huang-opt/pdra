# Private Distributed Resource Allocation Without Constraint Violations

This project is for the numerical experiments of the algorithm proposed in Private Distributed Resource Allocation Without Constraint Violations.

## Usage
### 1. Installation
To get started, first clone the repository and install the required dependencies:
```bash
git clone https://github.com/rui-huang-opt/pdra-exp.git
cd pdra-exp
```

#### Installing Dependency
You can install the necessary dependencies using `pip`.
 Make sure to use a virtual environment for isolation:
 ```bash
 pip install -r requirements.txt
 ```
Alternatively, if you are using a different package manager or want to install dependencies manually, ensure that the packages in `requirements.txt` are installed.
### 2. Configuration
Before running the experiments, configure the input parameters in the `config.toml` file.
Below are the key sections and settings that you may want to customize:
#### 1) General Plotting Settings:
Control whether to plot the graph and customize plot options:
```toml
PLOT_MODE = true  # Set to true to plot the graph, false to only run the experiment.

[GRAPH_PLOT_OPTIONS]
with_labels = true         # Set to true to display labels on the nodes.
font_size = 20             # Font size for the labels.
node_color = "white"       # Color of the nodes.
node_size = 1000           # Size of the nodes.
edgecolors = "black"       # Color of the edges.
linewidths = 1.5           # Line width of the edges.
width = 1.5                # Width of the edges.
```
#### 2) DQP Configuration
Distributed Quadratic Programming (DQP) configuration including nodes, edges, node positions, and algorithm parameters.
```toml
[DQP]
NODES = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]  # List of nodes.
EDGES = [
    ["1", "2"],
    ["2", "3"],
    ["3", "4"],
    ["3", "6"],
    ["3", "7"],
    ["4", "5"],
    ["6", "8"],
    ["7", "9"]
]

[DQP.NODES_POS]
"1" = [-2.0, 1.0]
"2" = [-1.0, 0.5]
"3" = [0.0, 0.0]
"4" = [-1.0, -0.5]
"5" = [-2.0, -1.0]
"6" = [1.0, 0.5]
"7" = [1.0, -0.5]
"8" = [2.0, 1.0]
"9" = [2.0, -1.0]

[DQP.NODE_CONFIG]
iterations = 2000       # Number of iterations to run the DQP algorithm.
step_size = 3.0         # Step size for optimization.
method = "AGM"          # The method to use for updating the auxiliary variable.
solver = "OSQP"         # Solver to use for local optimization.
result_path = "../data/dqp/result"  # Path where the results will be saved.
```
#### 3) CP Configuration
The Collaborative Production (CP) Configuration is similar to the DQP configuration, with differences mainly in nodes, edges, node positions, and algorithm parameters. The structure and format are the same.

### 3. Running the Experiment
To run the experiment, make sure you're in the project's root directory. Follow these steps:

1. Navigate to the `scripts` directory:
```bash
cd scripts
``` 

2. run the experiment script using Python:
```bash
python distributed_qp.py
# python co_production.py
```

**Note: Since the code utilizes multiprocessing, we recommend running it on a Linux system as it handles multiprocessing more efficiently.**