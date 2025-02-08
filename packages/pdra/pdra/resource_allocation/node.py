import numpy as np
import cvxpy as cp
from numbers import Real
from typing import Callable, TypedDict
from numpy.typing import NDArray
from multiprocessing import Process
from gossip import Gossip
from .gradient_method import GradientMethodRegistry


class Configuration(TypedDict):
    """
    Configuration dictionary for the node.

    Arguments
    ----------
    iterations : int
        Number of iterations for the update of the auxiliary variable.
    step_size : Real
        Step size for the update of the auxiliary variable.
    method : str
        Method for the update of the auxiliary variable.
    solver : str
        Solver for the local optimization problem.
    result_path : str
        Path for saving the result.
    """
    
    iterations: int
    step_size: Real
    method: str
    solver: str
    result_path: str


class Node(Process):
    """
    Node class for distributed resource allocation problem.

    Arguments
    ----------
    name : str
        Node name.
    configuration : Configuration
        Configuration dictionary for the node.
    communication : Gossip
        Communication object, which is used for handling the gossip communication.
    f_i : Callable[[cp.Variable], cp.Expression]
        Local objective function.
    a_i : NDArray[np.float64]
        Coupling constraints matrix.
    g_i : Callable[[cp.Variable], cp.Expression], optional
        Local constraints function.
    """

    def __init__(
        self,
        name: str,
        configuration: Configuration,
        communication: Gossip,
        f_i: Callable[[cp.Variable], cp.Expression],  # Local objective
        a_i: NDArray[np.float64],  # Coupling constraints matrix
        g_i: Callable[[cp.Variable], cp.Expression] = None,  # Local constraints
    ):
        super().__init__()

        self.name = name

        self.configuration = configuration

        self.n_ccons, self.n_dim = a_i.shape

        self.x_i = cp.Variable(self.n_dim)
        self.y_i = np.zeros(self.n_ccons)

        # l_i @ y, where l_i is the ith row of laplacian matrix L
        self.li_y = cp.Parameter(self.n_ccons)
        self.update_y_i = GradientMethodRegistry.get(
            configuration["method"], configuration["step_size"]
        )

        self.communication = communication

        # Only certain node can get the information of the resource, others will set b_i to 0 by default
        self._b_i = cp.Parameter(self.n_ccons, value=np.zeros(self.n_ccons))

        self.prob = self.setup_local_problem(f_i, a_i, g_i)

        self.f_i_series = np.zeros(configuration["iterations"])
        self.x_i_series = np.zeros((self.n_dim, configuration["iterations"]))
        self.c_i_series = np.zeros((self.n_ccons, configuration["iterations"]))

    def set_resource(self, b_i: NDArray[np.float64]):
        self._b_i.value = b_i

    def setup_local_problem(
        self,
        f_i: Callable[[cp.Variable], cp.Expression],
        a_i: NDArray[np.float64],
        g_i: Callable[[cp.Variable], cp.Expression],
    ) -> cp.Problem:
        """
        The local optimization problem is modeled as

        min  f_i(x_i)

        s.t. a_i @ x_i + l_i @ y <= b_i,
             g_i(x_i) <= 0 (if exists).

        The value of b_i will be set to 0 if the node don't receive the information of the resource
        """
        cost = f_i(self.x_i)
        constraints = [a_i @ self.x_i + self.li_y - self._b_i <= 0]

        if g_i is not None:
            constraints.append(g_i(self.x_i) <= 0)

        return cp.Problem(cp.Minimize(cost), constraints)

    def save_result(self):
        np.savez(
            f"{self.configuration['result_path']}/node_{self.name}.npz",
            f_i_series=self.f_i_series,
            x_i_series=self.x_i_series,
            c_i_series=self.c_i_series,
        )

    def run(self) -> None:
        for k in range(self.configuration["iterations"]):
            self.communication.broadcast(self.y_i)
            y_j_all = self.communication.gather()

            self.li_y.value = self.y_i * self.communication.degree - sum(y_j_all)

            self.prob.solve(solver=self.configuration["solver"])

            c_i = self.prob.constraints[0].dual_value

            self.communication.broadcast(c_i)
            c_j_all = self.communication.gather()

            li_c = c_i * self.communication.degree - sum(c_j_all)

            self.y_i = self.update_y_i(self.y_i, li_c)

            self.f_i_series[k] = self.prob.value
            self.x_i_series[:, k] = self.x_i.value
            self.c_i_series[:, k] = c_i

        self.save_result()
