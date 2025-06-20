# Private Distributed Resource Allocation Without Constraint Violations

A privacy-preserving distributed optimization library for resource allocation problems with **provable constraint satisfaction guarantees**.

## Installation

### Method 1: Install directly from GitHub (Recommended)
```bash
pip install git+https://github.com/rui-huang-opt/pdra.git
```

### Method 2: Clone repository and install
```bash
git clone https://github.com/rui-huang-opt/pdra.git
cd pdra
pip install .
```

## Requirements
- Python >= 3.10
- cvxpy>=1.4.3
- topolink @ git+https://github.com/rui-huang-opt/topolink.git
- matplotlib, networkx, toml (for visualization and testing)

All dependencies are specified in `pyproject.toml` and will be installed automatically with `pip install .` or `pip install .[test]` for optional testing/visualization features.

## (Optional) Using a Virtual Environment

It is recommended to use a virtual environment to avoid dependency conflicts:

```bash
# On Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.venv\Scripts\activate
```

After activating the environment, proceed with the installation steps above.

## Testing

To run the test suite and visualization examples, install the optional dependencies:
```bash
pip install .[test]
```

## References

[1] [Notarnicola, I., & Notarstefano, G. (2019). Constraint-coupled distributed optimization: A relaxation and duality approach. *IEEE Transactions on Control of Network Systems*, 7(1), 483-492.](https://ieeexplore.ieee.org/abstract/document/8746216)

[2] [Simonetto, A., & Jamali-Rad, H. (2016). Primal recovery from consensus-based dual decomposition for distributed convex optimization. Journal of Optimization Theory and Applications, 168, 172-197.](https://link.springer.com/article/10.1007/s10957-015-0758-0)