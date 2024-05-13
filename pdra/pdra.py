import numpy as np
import cvxpy as cp
import dissys
from typing import List
from abc import ABCMeta, abstractmethod
from trunclap import TruncatedLaplace


# Accelerated gradient method
class AGD:
    def __init__(self, step_size: int or float):
        self.step_size = step_size
        self.w = None
        self.theta = 1

    def __call__(self, x: np.ndarray, gradient: np.ndarray):
        old_w = x if self.w is None else self.w
        old_theta = self.theta

        self.w = x - self.step_size * gradient
        self.theta = (1 + np.sqrt(1 + 4 * (self.theta ** 2))) / 2
        new_x = self.w + ((old_theta - 1) / self.theta) * (self.w - old_w)

        return new_x


# Subgradient method
class SM:
    def __init__(self, step_length: int or float):
        self.step_length = step_length
        self.decay_factor = 1

    def __call__(self, x: np.ndarray, subgradient: np.ndarray):
        new_x = x - self.step_length * self.decay_factor * subgradient
        self.decay_factor = np.sqrt(1 - 1 / (self.decay_factor ** 2 + 1))  # step size is gamma / sqrt(k + 1)

        return new_x


# Distributed resource allocation
class NodeDRABase(dissys.Node, metaclass=ABCMeta):
    def __init__(self,
                 iterations: int,
                 gamma: int or float,
                 method: str,
                 a_i: np.ndarray,
                 **kwargs):
        super().__init__()

        self.iterations = iterations

        self.coupling_constraints_num, self.dimension = a_i.shape

        self.x_i = cp.Variable(self.dimension)

        # Local auxiliary variables y_i
        self.y_i = np.zeros(self.coupling_constraints_num)

        # l_i @ y, where l_i is the ith row of laplacian matrix L
        self.li_y = cp.Parameter(self.coupling_constraints_num)

        # The method to update y_i
        self.update_y_i = globals()[method](gamma)

        # Only certain node can get the information of the resource, others will set b_i to 0 by default
        b_i = kwargs.get('b_i', 0)

        # The solver to the local problem (is set to OSQP by default)
        self.solver = kwargs.get('solver', cp.OSQP)

        # The local optimization problem is modeled as
        #
        # min  f_i(x_i)
        # s.t. a_i @ x_i + l_i @ y - b_i <= 0,
        #      local constraints.
        #
        # The value of b_i will be set to 0 if the node don't receive the information of the resource
        self.prob = cp.Problem(cp.Minimize(self.f_i), [a_i @ self.x_i + self.li_y - b_i <= 0] + self.local_constraints)

        # The iteration series of the required data
        self.f_i_series = np.zeros(iterations)
        self.x_i_series = np.zeros((self.dimension, iterations))
        self.c_i_series = np.zeros((self.coupling_constraints_num, iterations))

    @property
    @abstractmethod
    def f_i(self) -> cp.Expression:
        ...

    # If there is no local constraint, return []
    @property
    @abstractmethod
    def local_constraints(self) -> List[cp.Constraint]:
        ...

    def run(self) -> None:
        for k in range(self.iterations):
            # Exchange the information of y with neighbors
            self.broadcast_to_all_neighbors(self.y_i)
            y_j_all = self.receive_from_all_neighbors()

            # Calculate l_i @ y through summing up y_i - y_j for all j belonging to N_i,
            # where N_i is the set of neighbors
            self.li_y.value = self.y_i * self.in_degree - sum(y_j_all)

            self.prob.solve(solver=self.solver)

            # Obtain the Lagrange multipliers
            c_i = self.prob.constraints[0].dual_value

            # Exchange the information of c_i with neighbors
            self.broadcast_to_all_neighbors(c_i)
            c_j_all = self.receive_from_all_neighbors()

            # Calculate the (sub)gradient, l_i @ c
            li_c = c_i * self.in_degree - sum(c_j_all)

            self.y_i = self.update_y_i(self.y_i, li_c)

            # Take the snapshot
            self.f_i_series[k] = self.f_i.value
            self.x_i_series[:, k] = self.x_i.value
            self.c_i_series[:, k] = c_i


class Edge(dissys.Edge):
    ...


def resource_perturbation(epsilon: int or float,
                          delta: int or float,
                          sensitivity: int or float,
                          resource: np.ndarray) -> np.ndarray:
    s = (sensitivity / epsilon) * np.log(resource.size * (np.exp(epsilon) - 1) / delta + 1)
    trunc_lap = TruncatedLaplace(-s, s, 0, sensitivity / epsilon)
    perturbed_resource = resource - s * np.ones(resource.size) + trunc_lap(resource.size)

    return perturbed_resource
