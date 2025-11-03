from abc import ABCMeta, abstractmethod
from typing import Callable, Type, Dict, Any
from numpy import float64, zeros, empty, asarray
from cvxpy import Parameter, Variable, Expression, Problem, Constraint, Minimize
from numpy.typing import NDArray
from topolink import NodeHandle


class ResourceAllocator(metaclass=ABCMeta):
    _registry: Dict[str, Type["ResourceAllocator"]] = {}

    def __init_subclass__(cls, key: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        _key = key or cls.__name__
        if _key in cls._registry:
            raise ValueError(f"Node type '{_key}' is already registered.")
        cls._registry[_key] = cls

    def __init__(
        self,
        name: str,
        f_i: Callable[[Variable], Expression],
        a_i: NDArray[float64],
        g_i: Callable[[Variable], Expression] | None = None,
        max_iter: int = 1000,
        with_history: bool = False,
    ):
        super().__init__()

        self._name = name
        self._node_handle = NodeHandle(self._name)

        self._f_i = f_i
        self._a_i = a_i
        self._g_i = g_i

        self.max_iter = max_iter

        self.n_ccons, self.n_dim = a_i.shape

        self._x_i = Variable(self.n_dim)
        self._b_i = zeros(self.n_ccons)

        if with_history:
            self._history = {
                "f_i_series": zeros(self.max_iter),
                "x_i_series": zeros((self.n_dim, self.max_iter)),
                "computation_time": zeros(self.max_iter),
            }
        else:
            self._history = {}

    @classmethod
    def create(
        cls,
        name: str,
        f_i: Callable[[Variable], Expression],
        a_i: NDArray[float64],
        g_i: Callable[[Variable], Expression] | None = None,
        max_iter: int = 1000,
        with_history: bool = False,
        *args: Any,
        key: str = "proposed",
        **kwargs: Any,
    ) -> "ResourceAllocator":
        if key not in cls._registry:
            raise ValueError(f"Unknown node type: {key}")
        return cls._registry[key](
            name, f_i, a_i, g_i, max_iter, with_history, *args, **kwargs
        )

    @property
    def f_i(self) -> float:
        val = self._f_i(self._x_i).value
        if val is None:
            raise ValueError("The problem may not be solved.")
        return float(val)

    @property
    def x_i(self) -> NDArray[float64]:
        val = self._x_i.value
        if val is None:
            raise ValueError("The problem may not be solved.")
        return asarray(val, dtype=float64)

    @property
    def history(self) -> dict[str, NDArray[float64]]:
        return self._history

    @abstractmethod
    def _setup_local_problem(self) -> Problem: ...

    @abstractmethod
    def _perform_iteration(self, k: int, local_problem: Problem): ...

    def _record_history(self, k: int, local_problem: Problem):
        self._history["f_i_series"][k] = local_problem.objective.value
        self._history["x_i_series"][:, k] = local_problem.variables()[0].value
        self._history["computation_time"][k] = local_problem.solver_stats.solve_time

    def set_resource(self, b_i: NDArray[float64]):
        if b_i.shape != (self.n_ccons,):
            raise ValueError(
                f"The shape of b_i must be ({self.n_ccons},), but got {b_i.shape}."
            )
        self._b_i = b_i

    def run(self):
        local_problem = self._setup_local_problem()

        if not self._history:
            for k in range(self.max_iter):
                self._perform_iteration(k, local_problem)
        else:
            for k in range(self.max_iter):
                self._perform_iteration(k, local_problem)
                self._record_history(k, local_problem)


from .optimizer import GradientMethod


class Proposed(ResourceAllocator, key="proposed"):
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
        f_i: Callable[[Variable], Expression],
        a_i: NDArray[float64],
        g_i: Callable[[Variable], Expression] | None = None,
        max_iter: int = 1000,
        with_history: bool = False,
        solver: str = "OSQP",
        method: str = "AGM",
        step_size: float = 0.1,
    ):
        super().__init__(name, f_i, a_i, g_i, max_iter, with_history)

        self._y_i: NDArray[float64] = zeros(self.n_ccons)
        self._c_i: NDArray[float64] = empty(self.n_ccons)

        self._li_y = Parameter(self.n_ccons)

        self._solver = solver

        self._update_y_i = GradientMethod.create(method, step_size)

        self._history["c_i_series"] = zeros((self.n_ccons, self.max_iter))

    def _setup_local_problem(self) -> Problem:
        cost = self._f_i(self._x_i)
        constraints: list[Constraint] = [
            self._a_i @ self._x_i + self._li_y <= self._b_i
        ]

        if self._g_i is not None:
            constraints.append(self._g_i(self._x_i) <= 0)

        return Problem(Minimize(cost), constraints)

    def _record_history(self, k: int, local_problem: Problem):
        super()._record_history(k, local_problem)
        self._history["c_i_series"][:, k] = local_problem.constraints[0].dual_value

    def _perform_iteration(self, k: int, local_problem: Problem):
        self._li_y.value = self._node_handle.laplacian(self._y_i)

        local_problem.solve(solver=self._solver)

        self._c_i = local_problem.constraints[0].dual_value  # type: ignore

        li_c = self._node_handle.laplacian(self._c_i)

        self._y_i = self._update_y_i(self._y_i, li_c)
