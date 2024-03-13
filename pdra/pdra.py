import numpy as np
import cvxpy as cp
import pandas as pd
import dissys as ds
from abc import ABCMeta, abstractmethod
from trunclap import TruncatedLaplace
from typing import List, Dict


class NodePDRABase(ds.Node, metaclass=ABCMeta):
    def __init__(self, iterations, dimension, gamma, f_i, a_i: np.ndarray, b_i):
        super().__init__()
        # Iteration number
        self.iterations = iterations

        # Local decision variables and the array for taping its iterations
        self.x_i = cp.Variable(dimension)
        self.x_i_iter = np.zeros((dimension, self.iterations))

        # Step size for accelerated gradient descent or the step length for subgradient method
        self.gamma = gamma

        # Local objective function and constraint coefficient matrix
        self.f_i = f_i
        self.a_i = a_i

        # Array for taping the iterations of local objective function
        self.f_i_iter = np.zeros(self.iterations)

        # The number of local constraints
        self.cons_num = a_i.shape[0]

        # Only node 1 can receive resource, other nodes set it to 0 by default
        self.b_i = b_i if b_i is not None else np.zeros(self.cons_num)

        # Auxiliary variables y, Lagrange multipliers c and the arrays for taping their iterations
        self.y_i = np.zeros(self.cons_num)
        self.c_i = np.zeros(self.cons_num)
        self.y_i_iter = np.zeros((self.cons_num, self.iterations))
        self.c_i_iter = np.zeros((self.cons_num, self.iterations))

        # l_i @ y, where l_i is the ith row of laplacian matrix L
        # It is set as a Parameter class in cvxpy for it changes during every iteration
        self.li_y = cp.Parameter(self.cons_num)

        # l_i @ c, representing the corresponding (sub)gradient of y
        self.li_c = np.zeros(self.cons_num)

        # The local optimization problem is modeled as
        #
        # min  f_i(x_i)
        # s.t. a_i @ x_i + l_i @ y - b_i <= 0,
        #      local constraints.
        #
        self.prob = cp.Problem(cp.Minimize(self.f_i(self.x_i)),
                               [self.a_i @ self.x_i + self.li_y - self.b_i <= 0] + self.local_constraints)

    # If there is no local constraint, return []
    @property
    @abstractmethod
    def local_constraints(self) -> List[cp.Constraint]:
        ...

    # It is better to use different solver for different types of problem
    @property
    @abstractmethod
    def solver(self) -> str:
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
            self.x_i_iter[:, k] = self.x_i.value
            self.f_i_iter[k] = self.prob.value
            self.y_i_iter[:, k] = self.y_i
            self.c_i_iter[:, k] = self.c_i

            # Exchange the information of c with neighbors
            self.send_to_neighbors(self.c_i)
            c_j_all = self.recv_from_neighbors()

            # Calculate the (sub)gradient l_i @ c
            self.li_c = self.c_i * self.in_degree - sum(c_j_all)

            self.update_y(k)


# Accelerated gradient method
class NodeAG(NodePDRABase, metaclass=ABCMeta):
    def __init__(self, iterations, dimension, gamma, f_i, a_i: np.ndarray, b_i):
        super().__init__(iterations, dimension, gamma, f_i, a_i, b_i)

        # Auxiliary variables in accelerated gradient method
        self.w_i = np.zeros(self.cons_num)
        self.theta_i = 1

    def update_y(self, k: int) -> None:
        old_w_i = self.w_i
        old_theta_i = self.theta_i

        self.w_i = self.y_i - self.gamma * self.li_c
        self.theta_i = (1 + np.sqrt(1 + 4 * (self.theta_i ** 2))) / 2
        self.y_i = self.w_i + ((old_theta_i - 1) / self.theta_i) * (self.w_i - old_w_i)


# Subgradient method
class NodeSG(NodePDRABase, metaclass=ABCMeta):
    def __init__(self, iterations, dimension, gamma, f_i, a_i: np.ndarray, b_i):
        super().__init__(iterations, dimension, gamma, f_i, a_i, b_i)

    def update_y(self, k: int) -> None:
        self.y_i = self.y_i - (self.gamma / np.sqrt(k + 1)) * self.li_c


class Edge(ds.Edge):
    pass


def resource_perturbation(epsilon: int or float,
                          delta: int or float,
                          sensitivity: int or float,
                          resource_dim: int,
                          resource: np.ndarray) -> np.ndarray:
    s = (sensitivity / epsilon) * np.log(resource_dim * (np.exp(epsilon) - 1) / delta + 1)
    trunc_lap = TruncatedLaplace(-s, s, 0, sensitivity / epsilon)
    perturbed_resource = resource - s * np.ones(resource_dim) + trunc_lap(resource_dim)

    return perturbed_resource


def save_data(f_star: int or float,
              nodes: Dict[str, NodePDRABase],
              a_dic: Dict[str, np.ndarray],
              b: np.ndarray,
              file_path: str) -> None:
    # Error between F_iter and F_star
    f_iter = sum([node.f_i_iter for node in nodes.values()])

    err = f_iter - f_star
    df = pd.DataFrame(err)
    df.to_excel(file_path + r'\err.xlsx', index=False)

    # Lagrange multipliers
    for i, node in nodes.items():
        df = pd.DataFrame(node.c_i_iter)
        df.to_excel(file_path + r'\node' + i + r'\c_iter.xlsx', index=False)

    # The iterations of coupling constraints
    cons_iter = sum([a_dic[i] @ node.x_i_iter for i, node in nodes.items()]) - b[:, np.newaxis]

    df = pd.DataFrame(cons_iter)
    df.to_excel(file_path + r'\cons_iter.xlsx', index=False)
