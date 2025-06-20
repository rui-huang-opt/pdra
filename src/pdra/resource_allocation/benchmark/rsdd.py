import numpy as np
import cvxpy as cp
from typing import Callable, Dict
from numpy.typing import NDArray
from ..resource_allocation import Node


class RSDDNode(Node, key="rsdd"):
    """
    The local optimization problem is defined as:

    min  f_i(x_i)

    s.t. rho_i >= 0,
            a_i @ x_i + l_i @ y <= b_i + rho_i * [1, 1, ..., 1],
            g_i(x_i) <= 0 (if exists).

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
        decay_rate: float = 0.9,
        penalty_factor: float = 1e3,
    ):
        super().__init__(name, f_i, a_i, g_i, max_iter, results_prefix, sever_address)

        self._rho_i = cp.Variable()
        self._mu_i = np.zeros(self.n_ccons)

        self._lambda_ij_dict: Dict[int | str, NDArray[np.float64]] = {}
        self._lambda_ji_dict: Dict[int | str, NDArray[np.float64]] = {}

        self._bias_lambda = cp.Parameter(self.n_ccons)

        self._solver = solver
        self._step_size = step_size
        self._decay_rate = decay_rate
        self._penalty_factor = penalty_factor

    def setup_local_problem(self) -> cp.Problem:
        cost = self._f_i(self._x_i) + self._penalty_factor * self._rho_i
        constraints: list[cp.Constraint] = [
            self._a_i @ self._x_i + self._bias_lambda
            <= self._b_i + self._rho_i * np.ones(self.n_ccons),
            self._rho_i >= 0,
        ]

        if self._g_i is not None:
            constraints.append(self._g_i(self._x_i) <= 0)

        return cp.Problem(cp.Minimize(cost), constraints)

    def perform_iteration(self, k: int, local_problem: cp.Problem):
        for j in self.node_handle.neighbor_names:
            lambda_ij = self._lambda_ij_dict.get(j, np.zeros(self.n_ccons))
            self.node_handle.send(j, lambda_ij)

        for j in self.node_handle.neighbor_names:
            self._lambda_ji_dict[j] = self.node_handle.recv(j)

        sum_lambda_ij = sum(self._lambda_ij_dict.values())
        sum_lambda_ji = sum(self._lambda_ji_dict.values())
        self._bias_lambda.value = sum_lambda_ij - sum_lambda_ji

        local_problem.solve(solver=self._solver)

        self.mu_i: NDArray[np.float64] = local_problem.constraints[0].dual_value  # type: ignore

        gamma_k = self._step_size / ((k + 1) ** self._decay_rate)

        for j in self.node_handle.neighbor_names:
            self.node_handle.send(j, self.mu_i)

        for j in self.node_handle.neighbor_names:
            mu_j = self.node_handle.recv(j)
            lambda_ij = self._lambda_ij_dict.get(j, np.zeros(self.n_ccons))
            self._lambda_ij_dict[j] = lambda_ij - gamma_k * (self.mu_i - mu_j)
