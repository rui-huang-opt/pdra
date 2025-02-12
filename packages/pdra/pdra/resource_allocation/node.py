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
    comm_weight : Real, optional
        Weight for the communication term, default to None because it is not used in all cases.
    """

    iterations: int
    step_size: Real
    method: str
    solver: str
    result_path: str
    comm_weight: Real = None


class Results(TypedDict):
    """
    Results dictionary for the node.

    Arguments
    ----------
    f_i_series : NDArray[np.float64]
        Series of local objective function values.
    x_i_series : NDArray[np.float64]
        Series of local decision variables.
    c_i_series : NDArray[np.float64]
        Series of dual variables.
    """

    f_i_series: NDArray[np.float64]
    x_i_series: NDArray[np.float64]
    c_i_series: NDArray[np.float64]


class Node(Process):
    """
    Node class for distributed resource allocation problem.

    Arguments
    ----------
    name : str
        Node name.
    config : Configuration
        Configuration dictionary for the node.
    comm : Gossip
        Communicator for the node, using the gossip protocol.
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
        config: Configuration,
        comm: Gossip,
        f_i: Callable[[cp.Variable], cp.Expression],  # Local objective
        a_i: NDArray[np.float64],  # Coupling constraints matrix
        g_i: Callable[[cp.Variable], cp.Expression] = None,  # Local constraints
    ):
        super().__init__()

        self.name = name

        self.config = config

        self.n_ccons, self.n_dim = a_i.shape

        self.x_i = cp.Variable(self.n_dim)
        self.y_i = np.zeros(self.n_ccons)

        # l_i @ y, where l_i is the ith row of laplacian matrix L
        self.li_y = cp.Parameter(self.n_ccons)
        self.update_y_i = GradientMethodRegistry.get(
            config["method"], config["step_size"]
        )

        self.comm = comm

        # Only certain node can get the information of the resource, others will set b_i to 0 by default
        self._b_i = cp.Parameter(self.n_ccons, value=np.zeros(self.n_ccons))

        self.local_problem = self.setup_local_problem(f_i, a_i, g_i)

    def set_resource(self, b_i: NDArray[np.float64]):
        if b_i.ndim != 1 or b_i.shape[0] != self.n_ccons:
            raise ValueError("Invalid shape for resource allocation.")

        self._b_i.value = b_i

    def setup_local_problem(
        self,
        f_i: Callable[[cp.Variable], cp.Expression],
        a_i: NDArray[np.float64],
        g_i: Callable[[cp.Variable], cp.Expression],
    ) -> cp.Problem:
        """
        The local optimization problem is defined as:

        min  f_i(x_i)

        s.t. a_i @ x_i + l_i @ y <= b_i,
             g_i(x_i) <= 0 (if exists).

        The value of b_i will be set to 0 if the node don't receive the information of the resource.

        Arguments
        ----------
        f_i : Callable[[cp.Variable], cp.Expression]
            Local objective function.
        a_i : NDArray[np.float64]
            Coupling constraints matrix.
        g_i : Callable[[cp.Variable], cp.Expression]
            Local constraints function.

        Returns
        ----------
        cp.Problem
            Local optimization problem for the node.
        """
        cost = f_i(self.x_i)
        constraints = [a_i @ self.x_i + self.li_y - self._b_i <= 0]

        if g_i is not None:
            constraints.append(g_i(self.x_i) <= 0)

        return cp.Problem(cp.Minimize(cost), constraints)

    def initialize_results(self) -> Results:
        return {
            "f_i_series": np.zeros(self.config["iterations"]),
            "x_i_series": np.zeros((self.n_dim, self.config["iterations"])),
            "c_i_series": np.zeros((self.n_ccons, self.config["iterations"])),
        }

    def record_results(self, k: int, results: Results):
        results["f_i_series"][k] = self.local_problem.value
        results["x_i_series"][:, k] = self.x_i.value
        results["c_i_series"][:, k] = self.local_problem.constraints[0].dual_value

    def save_results(self, results: Results):
        result_path = self.config["result_path"]
        np.savez(f"{result_path}/node_{self.name}.npz", **results)

    def update(self):
        self.comm.broadcast(self.y_i)
        y_j_all = self.comm.gather()

        self.li_y.value = self.y_i * self.comm.degree - sum(y_j_all)

        self.local_problem.solve(solver=self.config["solver"])

        c_i = self.local_problem.constraints[0].dual_value

        self.comm.broadcast(c_i)
        c_j_all = self.comm.gather()

        li_c = c_i * self.comm.degree - sum(c_j_all)

        self.y_i = self.update_y_i(self.y_i, li_c)

    def run(self) -> None:
        results = self.initialize_results()

        for k in range(self.config["iterations"]):
            self.update()
            self.record_results(k, results)

        self.save_results(results)
