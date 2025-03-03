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
#### 1) **dqp** Configuration
Distributed Quadratic Programming (DQP) configuration including node_names, edge_pairs, and algorithm parameters.

**Core parameters:**
```toml
[dqp]
run_type = "plot"  # Set to "plot" to plot the results, "cen" to run the centralized optimization, "dis" to run the distributed optimization.
algorithm = "rsdd" # Set to "core" to run the algorithm proposed in the paper， "rsdd" to run relaxation and successive distributed decomposition (RSDD) algorithm.
```
#### 2) CP Configuration
The Collaborative Production (CP) Configuration is similar to the DQP configuration.
However, Only the proposed algorithm is used in this problem.

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