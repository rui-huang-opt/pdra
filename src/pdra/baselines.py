from typing import Callable, Dict
from numpy import float64, zeros, maximum, asarray
from cvxpy import Variable, Expression, Problem, Constraint, Minimize, Parameter
from numpy.typing import NDArray
from .network import NetworkOps


class RSDD:
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
        ops: NetworkOps,
        f_i: Callable[[Expression], Expression],
        a_i: NDArray[float64],
        g_i: Callable[[Expression], Expression] | None = None,
        b_i: NDArray[float64] | None = None,
        step_size: float = 0.1,
        decay_rate: float = 0.9,
        penalty: float = 1e3,
    ):
        self._ops = ops

        self._f_i = f_i
        self._a_i = a_i
        self._g_i = g_i

        self._n_ccons, self._n_dim = a_i.shape

        self._x_i = Variable(self._n_dim)
        self._b_i = zeros(self._n_ccons) if b_i is None else b_i

        self._rho_i = Variable()
        self._mu_i = zeros(self._n_ccons)

        self._lambda_ij_dict: Dict[str, NDArray[float64]] = {
            j: zeros(self._n_ccons) for j in self._ops.neighbors
        }

        self._lambda_residual = Parameter(self._n_ccons)

        self._step_size = step_size
        self._decay_rate = decay_rate
        self._penalty = penalty

        self._local_problem = self._setup_local_problem()

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

    def get_mu_i(self) -> NDArray[float64]:
        value = self._local_problem.constraints[0].dual_value
        if value is None:
            raise ValueError("The dual variable value is undefined.")
        return asarray(value, dtype=float64)

    def _setup_local_problem(self) -> Problem:
        cost = self._f_i(self._x_i) + self._penalty * self._rho_i
        constraints: list[Constraint] = [
            self._a_i @ self._x_i - self._b_i + self._lambda_residual <= self._rho_i,
            self._rho_i >= 0,
        ]

        if self._g_i is not None:
            constraints.append(self._g_i(self._x_i) <= 0)

        return Problem(Minimize(cost), constraints)

    def step(self, k: int, solver: str = "OSQP"):
        lambda_ji_dict = self._ops.exchange_map(self._lambda_ij_dict)

        sum_lambda_ij = sum(self._lambda_ij_dict.values())
        sum_lambda_ji = sum(lambda_ji_dict.values())
        self._lambda_residual.value = sum_lambda_ij - sum_lambda_ji

        self._local_problem.solve(solver=solver)
        mu_i = self.get_mu_i()

        gamma_k = self._step_size / ((k + 1) ** self._decay_rate)

        mu_js = self._ops.exchange(mu_i)

        for j, mu_j in mu_js.items():
            self._lambda_ij_dict[j] -= gamma_k * (mu_i - mu_j)


class DualDecomposition:
    """
    The local optimization problem is defined as:

    min  f_i(x_i) + mu_i @ (a_i @ x_i - b_i)

    s.t. g_i(x_i) <= 0 (if exists).

    The value of b_i will be set to 0 if the node don't receive the information of the resource.
    """

    def __init__(
        self,
        ops: NetworkOps,
        f_i: Callable[[Expression], Expression],
        a_i: NDArray[float64],
        g_i: Callable[[Expression], Expression] | None = None,
        b_i: NDArray[float64] | None = None,
        step_size: float = 0.1,
    ):
        self._ops = ops

        self._f_i = f_i
        self._a_i = a_i
        self._g_i = g_i

        self._n_ccons, self._n_dim = a_i.shape

        self._x_i = zeros(self._n_dim)
        self._x_i_wave = Variable(self._n_dim)
        self._mu_i = Parameter(self._n_ccons, value=zeros(self._n_ccons))
        self._b_i = zeros(self._n_ccons) if b_i is None else b_i

        self._local_problem = self._setup_local_problem()

        self._step_size = step_size

    @property
    def f_i(self) -> float:
        value: float = self._f_i(self._x_i)  # type: ignore
        return value

    @property
    def x_i(self) -> NDArray[float64]:
        return self._x_i

    @property
    def overhead(self) -> float:
        value = self._local_problem.solver_stats.solve_time
        if value is None:
            raise ValueError("The solver overhead time is undefined.")
        return value

    def _setup_local_problem(self) -> Problem:
        cost = self._f_i(self._x_i_wave) + self._mu_i @ (
            self._a_i @ self._x_i_wave - self._b_i
        )
        constraints: list[Constraint] = (
            [] if self._g_i is None else [self._g_i(self._x_i_wave) <= 0]
        )

        return Problem(Minimize(cost), constraints)

    def step(self, k: int, solver: str = "OSQP"):
        self._local_problem.solve(solver=solver)

        x_i_wave = self._x_i_wave.value
        mu_i = self._mu_i.value

        if x_i_wave is None or mu_i is None:
            raise ValueError("The problem may not be solved.")

        # Primal recovery
        self._x_i = (self._x_i * k + x_i_wave) / (k + 1)

        # Dual update
        subgrad = self._a_i @ x_i_wave - self._b_i
        raw_update = mu_i + self._step_size * subgrad

        combined_update = self._ops.weighted_mix(raw_update)
        self._mu_i.value = maximum(combined_update, 0)
