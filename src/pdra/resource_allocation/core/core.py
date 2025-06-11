import numpy as np
import cvxpy as cp
from typing import Callable
from numpy.typing import NDArray
from gossip import Gossip
from ..resource_allocation import Node
from ...utils.optimizer import GradientMethod


class CoreNode(Node, key="core"):
    """
    The local optimization problem is defined as:

    min  f_i(x_i)

    s.t. a_i @ x_i + l_i @ y <= b_i,
         g_i(x_i) <= 0 (if exists).

    The value of b_i will be set to 0 if the node don't receive the information of the resource.
    """

    def __init__(
        self,
        communicator: Gossip,
        f_i: Callable[[cp.Variable], cp.Expression],
        a_i: NDArray[np.float64],
        g_i: Callable[[cp.Variable], cp.Expression] | None = None,
        max_iter: int = 1000,
        results_prefix: str = "results",
        solver: str = "OSQP",
        method: str = "AGM",
        step_size: float = 0.1,
    ):
        super().__init__(communicator, f_i, a_i, g_i, max_iter, results_prefix)

        self._y_i: NDArray[np.float64] = np.zeros(self.n_ccons)
        self._c_i: NDArray[np.float64] = np.empty(self.n_ccons)

        self._li_y = cp.Parameter(self.n_ccons)

        self._solver = solver

        self._update_y_i = GradientMethod.create(method, step_size)

        self._results["c_i_series"] = np.zeros((self.n_ccons, self.max_iter))

    def setup_local_problem(self) -> cp.Problem:
        cost = self._f_i(self._x_i)
        constraints: list[cp.Constraint] = [
            self._a_i @ self._x_i + self._li_y <= self._b_i
        ]

        if self._g_i is not None:
            constraints.append(self._g_i(self._x_i) <= 0)

        return cp.Problem(cp.Minimize(cost), constraints)

    def record_results(self, k: int, local_problem: cp.Problem):
        super().record_results(k, local_problem)
        self._results["c_i_series"][:, k] = local_problem.constraints[0].dual_value

    def perform_iteration(self, k: int, local_problem: cp.Problem):
        self._communicator.broadcast(self._y_i)
        y_j_all = self._communicator.gather()

        self._li_y.value = self._y_i * self._communicator.degree - sum(y_j_all)

        local_problem.solve(solver=self._solver)

        self._c_i = local_problem.constraints[0].dual_value  # type: ignore

        self._communicator.broadcast(self._c_i)
        c_j_all = self._communicator.gather()

        li_c = self._c_i * self._communicator.degree - sum(c_j_all)

        self._y_i = self._update_y_i(self._y_i, li_c)
