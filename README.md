# Private Distributed Resource Allocation Without Constraint Violations

This repository contains the experimental code accompanying the paper [*Private Distributed Resource Allocation without Constraint Violations*](https://ieeexplore.ieee.org/abstract/document/11120444), published in **IEEE Transactions on Control of Network Systems (TCNS)**.

It provides a privacy-preserving distributed optimization library for resource allocation problems, ensuring **provable constraint satisfaction** throughout the optimization process.

## Installation

### Method 1: Install directly from GitHub (Recommended)
```bash
pip install git+https://github.com/rui-huang-opt/pdra.git
```

### Method 2: Clone repository and install for development
```bash
git clone https://github.com/rui-huang-opt/pdra.git
cd pdra
pip install -e .
```

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

## Truncated Laplace Noise

This library provides a standalone implementation of the **truncated Laplace noise** distribution. The `TruncatedLaplace` class allows you to sample from a Laplace distribution that is truncated to a specified interval.

### What is the Truncated Laplace Distribution?

The truncated Laplace distribution is a Laplace distribution restricted to a given interval $[\text{low}, \text{high}]$. Unlike the standard Laplace distribution, the truncated version only takes values within $[\text{low}, \text{high}]$—the probability outside this interval is set to zero, and the probability density is renormalized.

Its probability density function (PDF) is:

$$
f(x) = \frac{1}{Z} \cdot \frac{1}{2 \text{scale}} \exp\left(-\frac{|x-\text{loc}|}{\text{scale}}\right), \quad x \in [\text{low}, \text{high}]
$$

where:
- $\text{loc}$ is the location parameter (mean),
- $\text{scale}$ is the scale parameter,
- $Z$ is the normalization constant ensuring the total probability integrates to 1, given by

$$
Z = \int_{\text{low}} \frac{1}{2 \text{scale}} \exp\left(-\frac{|x-\text{loc}|}{\text{scale}}\right) dx
$$

The truncated Laplace distribution is commonly used in differential privacy and related applications, as it allows strict control over the output range while preserving privacy guarantees.

### Example usage:

```python
import numpy as np
import matplotlib.pyplot as plt
from pdra import TruncatedLaplace

low = -2
high = 1
loc = 0
scale = 1
n_samples = 50000
n_bins = 50
n_points = 1000

tl = TruncatedLaplace(low, high, loc, scale)

samples = tl.sample(n_samples)

fig, ax = plt.subplots()

ax.hist(samples, bins=n_bins, density=True)

x = np.linspace(low, high, n_points, dtype=np.float64)

ax.plot(x, tl.pdf(x), "r", lw=2)

ax.set_title("Truncated Laplace Distribution")
ax.set_xlabel("x")
ax.set_ylabel("pdf(x)")

plt.show()
```

![Truncated Laplace distribution](docs/images/truncated_laplace.png)

## Distributed Resource Allocation
This library provides tools for solving distributed resource allocation problems where multiple agents collaboratively optimize their local objectives subject to global and local constraints.

The distributed resource allocation optimization problem aims to coordinate multiple agents (or nodes) to optimize their individual objective functions while satisfying both global and local constraints. The standard formulation is:

$$
\begin{align*}
\min_{x_1, \ldots, x_N} \quad & \sum_{i=1}^N f_i(x_i) \\
\text{s.t.} \quad & x_i \in X_i, \quad \forall i=1,\ldots,N \\
& \sum_{i=1}^N A_i x_i \leq b
\end{align*}
$$

where $f_i$ is the local objective function of agent $i$, $X_i$ is its feasible set, and $\sum_{i=1}^N A_i x_i \leq b$ is the global resource constraint.
This problem commonly arises in scenarios such as energy allocation and communication networks.

### Example: Distributed Quadratic Programming

Suppose $5$ agents collaboratively solve a resource allocation problem where each agent $i$ minimizes a local quadratic cost:

$$
f_i(x_i) = \frac{1}{2} x_i^\top Q_i x_i + c_i x_i
$$

subject to no local constraints and a global resource constraint:

$$
\sum_{i=1}^N A_i x_i \leq b
$$

With this library, you can set up and solve such problems in a distributed manner.
For a complete example, please refer to the Jupyter notebooks in [`examples/distributed_qp`](examples/distributed_qp).

### Network Topology with `topolink`

This package leverages [`topolink`](https://github.com/rui-huang-opt/topolink) to easily construct distributed network topologies for multi-agent optimization.
This package enables the construction of real network topologies across multiple machines or processes, primarily for building undirected graphs.

To set up a network, please refer to the [topolink repository](https://github.com/rui-huang-opt/topolink) for detailed usage and examples.

## Baseline Methods

This package also implements two baseline algorithms for benchmarking purposes:

- **Relaxation and Successive Distributed Decomposition (RSDD)**: Based on [[1]](#references).
- **Dual Decomposition**: Based on [[2]](#references).

You can use these baselines to compare the performance of the proposed privacy-preserving algorithms. Example usage and benchmarking scripts are provided in the [`tests`](tests) directory.

## Baseline Methods

We consider two baseline methods for comparison:

1. **RSDD (Relaxation and Successive Distributed Decomposition)** [1].

2. **Consensus-Based Dual Decomposition with Primal Recovery** [2]

**References**  

[1] [Notarnicola, I., & Notarstefano, G. (2019). Constraint-coupled distributed optimization: A relaxation and duality approach. *IEEE Transactions on Control of Network Systems*, 7(1), 483-492.](https://ieeexplore.ieee.org/abstract/document/8746216)

[2] [Simonetto, A., & Jamali-Rad, H. (2016). Primal recovery from consensus-based dual decomposition for distributed convex optimization. Journal of Optimization Theory and Applications, 168, 172-197.](https://link.springer.com/article/10.1007/s10957-015-0758-0)