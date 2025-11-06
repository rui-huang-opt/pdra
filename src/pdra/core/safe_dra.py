from typing import Callable
from numpy import float64, zeros, asarray
from cvxpy import Parameter, Variable, Expression, Problem, Constraint, Minimize
from numpy.typing import NDArray
from topolink import NodeHandle
from .optimizer import GradientDescent


class SafeDRA:
    def __init__(
        self,
        name: str,
        f_i: Callable[[Expression], Expression],
        a_i: NDArray[float64],
        g_i: Callable[[Expression], Expression] | None = None,
        b_i: NDArray[float64] | None = None,
        method: str = "nesterov",
        step_size: float = 0.1,
    ):
        super().__init__()

        self._name = name
        self._node_handle = NodeHandle(self._name)

        self._f_i = f_i
        self._a_i = a_i
        self._g_i = g_i

        self._n_ccons, self._n_dim = a_i.shape

        self._x_i = Variable(self._n_dim)
        self._b_i = zeros(self._n_ccons) if b_i is None else b_i

        self._y_i: NDArray[float64] = zeros(self._n_ccons)

        self._li_y = Parameter(self._n_ccons)

        self._local_problem = self._setup_local_problem()
        self._optimizer = GradientDescent.create(step_size, key=method)

    @property
    def f_i(self) -> float:
        value = self._local_problem.value
        if not isinstance(value, float):
            raise ValueError("The local objective value is invalid.")
        return value

    @property
    def x_i(self) -> NDArray[float64]:
        value = self._x_i.value
        if value is None:
            raise ValueError("The problem may not be solved.")
        return asarray(value, dtype=float64)

    @property
    def overhead(self) -> float:
        value = self._local_problem.solver_stats.solve_time
        if value is None:
            raise ValueError("The solver overhead time is undefined.")
        return value

    @property
    def _c_i(self) -> NDArray[float64]:
        return asarray(self._local_problem.constraints[0].dual_value, dtype=float64)

    def _setup_local_problem(self) -> Problem:
        cost = self._f_i(self._x_i)
        constraints: list[Constraint] = [
            self._a_i @ self._x_i + self._li_y <= self._b_i
        ]

        if self._g_i is not None:
            constraints.append(self._g_i(self._x_i) <= 0)

        return Problem(Minimize(cost), constraints)

    def step(self, k: int, solver: str = "OSQP"):
        self._li_y.value = self._node_handle.laplacian(self._y_i)

        self._local_problem.solve(solver=solver)

        li_c = self._node_handle.laplacian(self._c_i)

        self._y_i = self._optimizer.step(self._y_i, li_c, k)
