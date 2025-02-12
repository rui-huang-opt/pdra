import numpy as np
import cvxpy as cp
from typing import TypedDict, Callable
from numpy.typing import NDArray
from multiprocessing import Process
from gossip import Gossip


class Configuration(TypedDict):
    pass


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
        self.comm = comm

        self.n_dim, self.n_ccons = a_i.shape

        self.x_i_wave = cp.Variable(self.n_dim)
        self.mu_i_wave = cp.Parameter(self.n_ccons, value=np.zeros(self.n_ccons))

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

        min  f_i(x_i) + mu_i @ (a_i @ x_i - b_i)

        s.t. g_i(x_i) <= 0 (if exists).

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

        cost = f_i(self.x_i_wave) + self.mu_i_wave @ (a_i @ self.x_i_wave - self._b_i)
        constraints = [] if g_i is None else [g_i(self.x_i_wave) <= 0]

        return cp.Problem(cp.Minimize(cost), constraints)
    
    def run(self):
        pass
