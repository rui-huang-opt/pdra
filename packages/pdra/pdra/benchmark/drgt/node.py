import numpy as np
import cvxpy as cp
from typing import Callable, TypedDict, List
from numpy.typing import NDArray
from multiprocessing import Process
from gossip import Gossip


class Configuration(TypedDict):
    iterations: int
    gamma: float
    phi: float
    row_weights: List[float]
    col_weights: List[float]
    method: str
    solver: str
    result_path: str


class Node(Process):
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

        self.omega_i = cp.Variable(self.n_dim)
        self.omega_wave_i = cp.Parameter(self.n_ccons, value=np.zeros(self.n_ccons))
        self.s_i = np.zeros(self.n_dim)

        self.comm = comm

        # Only certain node can get the information of the resource, others will set b_i to 0 by default
        self._b_i = cp.Parameter(self.n_ccons, value=np.zeros(self.n_ccons))

        self.prob = self.setup_local_problem(f_i, a_i, g_i)

        self.f_i_series = np.zeros(config["iterations"])
        self.omega_i_series = np.zeros((self.n_dim, config["iterations"]))

    def setup_local_problem(
        self,
        f_i: Callable[[cp.Variable], cp.Expression],
        a_i: NDArray[np.float64],
        g_i: Callable[[cp.Variable], cp.Expression] = None,
    ) -> cp.Problem:
        """
        Setup the local problem for the node.
        The local problem is defined as:

        min f_i(omega_i) - omega_wave_i @ a_i @ omega_i.

        s.t. g_i(omega_i) <= 0.

        The omega_wave_i is the iteration of the dual variable.

        Arguments
        ----------
        f_i : Callable[[cp.Variable], cp.Expression]
            Local objective function.
        a_i : NDArray[np.float64]
            Coupling constraints matrix.
        g_i : Callable[[cp.Variable], cp.Expression], optional
            Local constraints function.

        Returns
        ----------
        cp.Problem
            Local problem for the node.
        """

        objective = f_i(self.omega_i) - self.omega_wave_i @ a_i @ self.omega_i
        constraints = [] if g_i is None else [g_i(self.omega_i) <= 0]

        return cp.Problem(cp.Minimize(objective), constraints)

    def run(self):
        for k in range(self.config["iterations"]):
            self.comm.broadcast(self.s_i)
            all_s_j = self.comm.gather()

            self.comm.broadcast(self.omega_wave_i.value)
            all_omega_wave_j = self.comm.gather()
