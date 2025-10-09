import os
import numpy as np
import cvxpy as cp
from abc import ABCMeta, abstractmethod
from typing import Callable, Type, Dict, Any
from numpy.typing import NDArray
from topolink import NodeHandle


class Node(metaclass=ABCMeta):
    _registry: Dict[str, Type["Node"]] = {}

    def __init_subclass__(cls, key: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        _key = key or cls.__name__
        if _key in cls._registry:
            raise ValueError(f"Node type '{_key}' is already registered.")
        cls._registry[_key] = cls

    def __init__(
        self,
        name: str,
        f_i: Callable[[cp.Variable], cp.Expression],
        a_i: NDArray[np.float64],
        g_i: Callable[[cp.Variable], cp.Expression] | None = None,
        max_iter: int = 1000,
        results_prefix: str | None = None,
    ):
        super().__init__()

        self._name = name
        self._node_handle = NodeHandle(self._name)

        self._f_i = f_i
        self._a_i = a_i
        self._g_i = g_i

        self.max_iter = max_iter

        self.n_ccons, self.n_dim = a_i.shape

        self._x_i = cp.Variable(self.n_dim)
        self._b_i = np.zeros(self.n_ccons)

        self._results_prefix = results_prefix

        if self._results_prefix is not None:
            self._results = {
                "f_i_series": np.zeros(self.max_iter),
                "x_i_series": np.zeros((self.n_dim, self.max_iter)),
                "computation_time": np.zeros(self.max_iter),
            }

    @classmethod
    def create(
        cls,
        algorithm: str,
        name: str,
        f_i: Callable[[cp.Variable], cp.Expression],
        a_i: NDArray[np.float64],
        g_i: Callable[[cp.Variable], cp.Expression] | None = None,
        max_iter: int = 1000,
        results_prefix: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> "Node":
        if algorithm not in Node._registry:
            raise ValueError(f"Unknown node type: {algorithm}")
        return Node._registry[algorithm](
            name, f_i, a_i, g_i, max_iter, results_prefix, *args, **kwargs
        )

    def set_resource(self, b_i: NDArray[np.float64]):
        if b_i.ndim != 1 or b_i.shape[0] != self.n_ccons:
            raise ValueError("Invalid shape for resource allocation.")

        self._b_i = b_i

    @abstractmethod
    def setup_local_problem(self) -> cp.Problem: ...

    @property
    def f_i(self) -> float:
        val = self._f_i(self._x_i).value
        if val is None:
            raise ValueError("The problem may not be solved.")
        return float(val)

    @property
    def x_i(self) -> NDArray[np.float64]:
        val = self._x_i.value
        if val is None:
            raise ValueError("The problem may not be solved.")
        return np.asarray(val, dtype=np.float64)

    @abstractmethod
    def perform_iteration(self, k: int, local_problem: cp.Problem): ...

    def record_results(self, k: int, local_problem: cp.Problem):
        self._results["f_i_series"][k] = local_problem.objective.value
        self._results["x_i_series"][:, k] = local_problem.variables()[0].value
        self._results["computation_time"][k] = local_problem.solver_stats.solve_time

    def save_results(self, results_prefix: str):
        os.makedirs(results_prefix, exist_ok=True)
        np.savez(
            os.path.join(results_prefix, f"node_{self._name}.npz"), **self._results
        )

    def run(self):
        local_problem = self.setup_local_problem()

        if self._results_prefix is None:
            for k in range(self.max_iter):
                self.perform_iteration(k, local_problem)
        else:
            for k in range(self.max_iter):
                self.perform_iteration(k, local_problem)
                self.record_results(k, local_problem)

            self.save_results(self._results_prefix)


from .optimizer import GradientMethod


class ProposedNode(Node, key="proposed"):
    """
    The proposed method in our paper,
    that iteratively refines a solution by solving subproblems:

    min  f_i(x_i)

    s.t. a_i @ x_i + l_i @ y <= b_i,
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
        method: str = "AGM",
        step_size: float = 0.1,
    ):
        super().__init__(name, f_i, a_i, g_i, max_iter, results_prefix)

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
        self._li_y.value = self._node_handle.laplacian(self._y_i)

        local_problem.solve(solver=self._solver)

        self._c_i = local_problem.constraints[0].dual_value  # type: ignore

        li_c = self._node_handle.laplacian(self._c_i)

        self._y_i = self._update_y_i(self._y_i, li_c)
