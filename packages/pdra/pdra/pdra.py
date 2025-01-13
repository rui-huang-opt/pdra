import numpy as np
import cvxpy as cp
from numbers import Real
from typing import List, Union, Callable
from numpy.typing import NDArray
from multiprocessing import Process
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from gossip import Gossip


class GradientBasedMethod(metaclass=ABCMeta):
    def __init__(self, step_size: Real):
        self.step_size = step_size

    @abstractmethod
    def __call__(self, x: NDArray[np.float64], direction: NDArray[np.float64]) -> np.ndarray: ...


# Accelerated gradient method
class AGD(GradientBasedMethod):
    def __init__(self, step_size: Real):
        super().__init__(step_size)

        self.w = None
        self.theta = 1

    def __call__(self, x: NDArray[np.float64], gradient: NDArray[np.float64]):
        old_w = x if self.w is None else self.w
        old_theta = self.theta

        self.w = x - self.step_size * gradient
        self.theta = (1 + np.sqrt(1 + 4 * (self.theta**2))) / 2
        new_x = self.w + ((old_theta - 1) / self.theta) * (self.w - old_w)

        return new_x


# Subgradient method
class SM(GradientBasedMethod):
    def __init__(self, step_size: Real):
        super().__init__(step_size)

        self.decay_factor = 1

    def __call__(self, x: NDArray[np.float64], subgradient: NDArray[np.float64]):
        new_x = x - self.step_size * self.decay_factor * subgradient
        self.decay_factor = np.sqrt(
            1 - 1 / (self.decay_factor**2 + 1)
        )  # step size is gamma / sqrt(k + 1)

        return new_x


@dataclass
class DRAConfiguration:
    iterations: int
    method: GradientBasedMethod
    result_dir: str
    solver: str = cp.OSQP


# Distributed resource allocation
class NodeDRABase(Process, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        config: DRAConfiguration,
        f_i: Callable[[cp.Variable], cp.Expression],
        a_i: NDArray[np.float64],
        communication: Gossip,
        **kwargs,
    ):
        super().__init__()

        self.name = name

        self.iterations = config.iterations
        self.coupling_constraints_num, self.dimension = a_i.shape

        self.x_i = cp.Variable(self.dimension)
        self.y_i = np.zeros(self.coupling_constraints_num)

        # l_i @ y, where l_i is the ith row of laplacian matrix L
        self.li_y = cp.Parameter(self.coupling_constraints_num)

        # The method to update y_i
        self.update_y_i = config.method

        self.communication = communication
        self.result_dir = config.result_dir

        # Only certain node can get the information of the resource, others will set b_i to 0 by default
        b_i: Union[int, NDArray[np.float64]] = kwargs.get("b_i", 0)

        # The solver to the local problem (is set to OSQP by default)
        self.solver = kwargs.get("solver", cp.OSQP)

        # The local optimization problem is modeled as
        #
        # min  f_i(x_i)
        # s.t. a_i @ x_i + l_i @ y - b_i <= 0,
        #      local constraints.
        #
        # The value of b_i will be set to 0 if the node don't receive the information of the resource
        self.prob = cp.Problem(
            cp.Minimize(f_i(self.x_i)),
            [a_i @ self.x_i + self.li_y - b_i <= 0] + self.local_constraints,
        )

        # The iteration series of the required data
        self.f_i_series = np.zeros(config.iterations)
        self.x_i_series = np.zeros((self.dimension, config.iterations))
        self.c_i_series = np.zeros((self.coupling_constraints_num, config.iterations))

    # If there is no local constraint, return []
    @property
    @abstractmethod
    def local_constraints(self) -> List[cp.Constraint]: ...

    def save_result(self):
        np.savez(
            f"{self.result_dir}/{self.name}_result.npz",
            f_i_series=self.f_i_series,
            x_i_series=self.x_i_series,
            c_i_series=self.c_i_series,
        )

    def run(self) -> None:
        for k in range(self.iterations):
            # Exchange the information of y with neighbors
            self.communication.broadcast(self.y_i)
            y_j_all = self.communication.gather()

            # Calculate l_i @ y through summing up y_i - y_j for all j belonging to N_i,
            # where N_i is the set of neighbors
            self.li_y.value = self.y_i * self.communication.degree - sum(y_j_all)

            self.prob.solve(solver=self.solver)

            # Obtain the Lagrange multipliers
            c_i = self.prob.constraints[0].dual_value

            # Exchange the information of c_i with neighbors
            self.communication.broadcast(c_i)
            c_j_all = self.communication.gather()

            # Calculate the (sub)gradient, l_i @ c
            li_c = c_i * self.communication.degree - sum(c_j_all)

            self.y_i = self.update_y_i(self.y_i, li_c)

            # Take the snapshot
            self.f_i_series[k] = self.prob.value
            self.x_i_series[:, k] = self.x_i.value
            self.c_i_series[:, k] = c_i

        self.save_result()
