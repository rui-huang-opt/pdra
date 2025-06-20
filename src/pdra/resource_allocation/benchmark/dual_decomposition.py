import numpy as np
import cvxpy as cp
from typing import Callable
from numpy.typing import NDArray
from ..resource_allocation import Node


class DualDecompositionNode(Node, key="dual_decomp"):
    """
    The local optimization problem is defined as:

    min  f_i(x_i) + mu_i @ (a_i @ x_i - b_i)

    s.t. g_i(x_i) <= 0 (if exists).

    The value of b_i will be set to 0 if the node don't receive the information of the resource.
    """

    def __init__(
        self,
        name: str,
        f_i: Callable[[cp.Variable], cp.Expression],
        a_i: NDArray[np.float64],
        g_i: Callable[[cp.Variable], cp.Expression] | None = None,
        max_iter: int = 1000,
        results_prefix: str = "results",
        sever_address: str = "localhost:5555",
        solver: str = "OSQP",
        step_size: float = 0.1,
        comm_weight: float = 0.5,
    ):
        super().__init__(name, f_i, a_i, g_i, max_iter, results_prefix, sever_address)

        self._x_i.value = np.zeros(self.n_dim)
        self._x_i_wave = cp.Variable(self.n_dim)
        self._mu_i = cp.Parameter(self.n_ccons, value=np.zeros(self.n_ccons))

        self._solver = solver
        self._step_size = step_size
        self._comm_weight = comm_weight

    def setup_local_problem(self) -> cp.Problem:
        cost = self._f_i(self._x_i_wave) + self._mu_i @ (
            self._a_i @ self._x_i_wave - self._b_i
        )
        constraints: list[cp.Constraint] = (
            [] if self._g_i is None else [self._g_i(self._x_i_wave) <= 0]
        )

        return cp.Problem(cp.Minimize(cost), constraints)

    def perform_iteration(self, k: int, local_problem: cp.Problem):
        local_problem.solve(solver=self._solver)

        x_i = self._x_i.value
        x_i_wave = self._x_i_wave.value
        mu_i = self._mu_i.value

        if x_i is None or x_i_wave is None or mu_i is None:
            raise ValueError("Local problem may not be solved correctly.")

        self._x_i.value = (x_i * k + x_i_wave) / (k + 1)

        local_update = mu_i + self._step_size / np.sqrt(1 + k) * (
            self._a_i @ x_i_wave - self._b_i
        )

        consensus_error = self.node_handle.laplacian(local_update)
        combined_update = local_update - self._comm_weight * consensus_error

        self._mu_i.value = np.maximum(combined_update, 0)
