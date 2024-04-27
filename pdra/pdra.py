import numpy as np
import cvxpy as cp
import dissys as ds
from abc import ABCMeta, abstractmethod
from trunclap import TruncatedLaplace
from typing import List


class NodePDRABase(ds.Node, metaclass=ABCMeta):
    def __init__(self,
                 iterations: int,
                 gamma: int or float,
                 a_i: np.ndarray,
                 b_i: np.ndarray or None):
        super().__init__()

        self.iterations = iterations
        self.coupling_constraints_num, self.dimension = a_i.shape

        # Step size for updating y_i
        self.gamma = gamma

        self.x_i = cp.Variable(self.dimension)
        self.a_i = a_i
        self.b_i = np.zeros(self.coupling_constraints_num) if b_i is None else b_i

        # Local auxiliary variables y_i and Lagrange multipliers c_i
        self.y_i = np.zeros(self.coupling_constraints_num)
        self.c_i = np.zeros(self.coupling_constraints_num)

        # l_i @ y and l_i @ c, where l_i is the ith row of laplacian matrix L
        self.li_y = cp.Parameter(self.coupling_constraints_num)
        self.li_c = np.zeros(self.coupling_constraints_num)

        # The local optimization problem is modeled as
        #
        # min  f_i(x_i)
        # s.t. a_i @ x_i + l_i @ y - b_i <= 0,
        #      local constraints.
        #
        self.prob = cp.Problem(cp.Minimize(self.f_i),
                               [self.a_i @ self.x_i + self.li_y - self.b_i <= 0] + self.local_constraints)

        # The solver is set to OSQP by default
        self.solver = cp.OSQP

        # The iterations of the required data
        self.f_i_iter = np.zeros(iterations)
        self.x_i_iter = np.zeros((self.dimension, iterations))
        self.c_i_iter = np.zeros((self.coupling_constraints_num, iterations))

    @property
    @abstractmethod
    def f_i(self) -> cp.Expression:
        ...

    # If there is no local constraint, return []
    @property
    @abstractmethod
    def local_constraints(self) -> List[cp.Constraint]:
        ...

    # Update the auxiliary variable y through:
    # 1. "accelerated gradient method"
    # 2. "subgradient method"
    @abstractmethod
    def update_y(self, k: int) -> None:
        ...

    def run(self) -> None:
        for k in range(self.iterations):
            # Exchange the information of y with neighbors
            self.send_to_neighbors(self.y_i)
            y_j_all = self.recv_from_neighbors()

            # Calculate l_i @ y through summing up y_i - y_j for all j belonging to N_i,
            # where N_i is the set of neighbors
            self.li_y.value = self.y_i * self.in_degree - sum(y_j_all)

            self.prob.solve(solver=self.solver)

            # Update the Lagrange multipliers
            self.c_i = self.prob.constraints[0].dual_value

            # Tape the required data
            self.f_i_iter[k] = self.f_i.value
            self.x_i_iter[:, k] = self.x_i.value
            self.c_i_iter[:, k] = self.c_i

            # Exchange the information of c with neighbors
            self.send_to_neighbors(self.c_i)
            c_j_all = self.recv_from_neighbors()

            # Calculate the (sub)gradient l_i @ c
            self.li_c = self.c_i * self.in_degree - sum(c_j_all)

            self.update_y(k)


# Accelerated gradient method
class NodeAG(NodePDRABase, metaclass=ABCMeta):
    def __init__(self,
                 iterations: int,
                 gamma: int or float,
                 a_i: np.ndarray,
                 b_i: np.ndarray or None):
        # Auxiliary variables in accelerated gradient method
        super().__init__(iterations, gamma, a_i, b_i)
        self.w_i = np.zeros(self.coupling_constraints_num)
        self.theta_i = 1

    def update_y(self, k: int) -> None:
        old_w_i = self.w_i
        old_theta_i = self.theta_i

        self.w_i = self.y_i - self.gamma * self.li_c
        self.theta_i = (1 + np.sqrt(1 + 4 * (self.theta_i ** 2))) / 2
        self.y_i = self.w_i + ((old_theta_i - 1) / self.theta_i) * (self.w_i - old_w_i)


# Subgradient method
class NodeSG(NodePDRABase, metaclass=ABCMeta):
    def __init__(self,
                 iterations: int,
                 gamma: int or float,
                 a_i: np.ndarray,
                 b_i: np.ndarray or None):
        super().__init__(iterations, gamma, a_i, b_i)

    def update_y(self, k: int) -> None:
        self.y_i = self.y_i - (self.gamma / np.sqrt(k + 1)) * self.li_c


class Edge(ds.Edge):
    pass


def resource_perturbation(epsilon: int or float,
                          delta: int or float,
                          sensitivity: int or float,
                          resource: np.ndarray) -> np.ndarray:
    s = (sensitivity / epsilon) * np.log(resource.size * (np.exp(epsilon) - 1) / delta + 1)
    trunc_lap = TruncatedLaplace(-s, s, 0, sensitivity / epsilon)
    perturbed_resource = resource - s * np.ones(resource.size) + trunc_lap(resource.size)

    return perturbed_resource
