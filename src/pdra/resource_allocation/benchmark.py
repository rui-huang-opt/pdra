import numpy as np
import cvxpy as cp
from typing import Callable, Dict
from numpy.typing import NDArray
from .resource_allocation import Node


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
        solver: str = "OSQP",
        step_size: float = 0.1,
        decay_rate: float = 0.9,
        penalty_factor: float = 1e3,
    ):
        super().__init__(name, f_i, a_i, g_i, max_iter, results_prefix)

        self._rho_i = cp.Variable()
        self._mu_i = np.zeros(self.n_ccons)

        self._lambda_ij_dict: Dict[str, NDArray[np.float64]] = {
            j: np.zeros(self.n_ccons) for j in self._node_handle.neighbor_names
        }
        self._lambda_ji_dict: Dict[str, NDArray[np.float64]] = {
            j: np.zeros(self.n_ccons) for j in self._node_handle.neighbor_names
        }

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
        self._node_handle.send_to_all(self._lambda_ij_dict)
        self._lambda_ji_dict.update(self._node_handle.gather())

        sum_lambda_ij = sum(self._lambda_ij_dict.values())
        sum_lambda_ji = sum(self._lambda_ji_dict.values())
        self._bias_lambda.value = sum_lambda_ij - sum_lambda_ji

        local_problem.solve(solver=self._solver)

        self.mu_i: NDArray[np.float64] = local_problem.constraints[0].dual_value  # type: ignore

        gamma_k = self._step_size / ((k + 1) ** self._decay_rate)

        self._node_handle.broadcast(self.mu_i)
        neighbor_mus = self._node_handle.gather()

        for j, mu_j in neighbor_mus.items():
            self._lambda_ij_dict[j] -= gamma_k * (self.mu_i - mu_j)


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
        solver: str = "OSQP",
        step_size: float = 0.1,
        comm_weight: float = 0.5,
    ):
        super().__init__(name, f_i, a_i, g_i, max_iter, results_prefix)

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

        consensus_error = self._node_handle.laplacian(local_update)
        combined_update = local_update - self._comm_weight * consensus_error

        self._mu_i.value = np.maximum(combined_update, 0)
