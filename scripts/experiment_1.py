import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
from pdra import NodeDRABase, DRAConfiguration, AGD
from gossip import create_gossip_network
from plot_options import SCRIPT_TYPE, OPTIONS
import time


class NodeDQP(NodeDRABase):
    def __init__(self, name, configuration: DRAConfiguration, f_i, a_i, comm, **kwargs):
        result_dir = "../data/dqp/result"

        super().__init__(name, configuration, f_i, a_i, comm, **kwargs)

    @property
    def local_constraints(self) -> List[cp.Constraint]:
        return []


if __name__ == "__main__":
    """
    Parameters
    """
    N = 9  # Number of nodes
    T = 2000  # iteration number
    STEP_SIZE = 3

    NODES = [f"{i}" for i in range(1, N + 1)]
    EDGES = [
        ("1", "2"),
        ("2", "3"),
        ("3", "4"),
        ("3", "6"),
        ("3", "7"),
        ("4", "5"),
        ("6", "8"),
        ("7", "9"),
    ]

    NODES_POS = {
        "1": (-2, 1),
        "2": (-1, 0.5),
        "3": (0, 0),
        "4": (-1, -0.5),
        "5": (-2, -1),
        "6": (1, 0.5),
        "7": (1, -0.5),
        "8": (2, 1),
        "9": (2, -1),
    }

    """
    Load the model of the problem
    """
    # di = 4
    # M = 3
    #
    # Q = {}
    # g = {}
    # A = {}
    #
    # for i in NODES:
    #     Q_i_prime = np.random.randn(di, di)
    #     Q[i] = np.eye(di) + Q_i_prime @ Q_i_prime.T
    #     g[i] = np.random.randint(-5, 5, di)
    #     A[i] = np.random.randint(-50, 50, (M, di))
    #
    # np.savez(rf'../data/{experiment}/model/Q.npz', **Q)
    # np.savez(rf'../data/{experiment}/model/g.npz', **g)
    # np.savez(rf'../data/{experiment}/model/A.npz', **A)
    #
    # vec = np.random.uniform(0, 1, 3)
    # pr = vec / vec.sum()
    # D = np.random.choice([1, 2, 3], 1000, p=pr)
    # np.save(rf'../data/experiment/model/D.npy', D)

    Q = np.load(f"../data/dqp/model/Q.npz")
    g = np.load(f"../data/dqp/model/g.npz")
    A = np.load(f"../data/dqp/model/A.npz")

    D = np.load("../data/dqp/model/D.npy")

    proportion_of_1 = D[np.where(D == 1)].size / 1000
    proportion_of_2 = D[np.where(D == 2)].size / 1000
    proportion_of_3 = D[np.where(D == 3)].size / 1000

    b = np.array([proportion_of_1, proportion_of_2, proportion_of_3])

    """
    Centralized optimization
    """
    x = {i: cp.Variable(A[i].shape[1]) for i in NODES}

    cost = cp.sum([x[i] @ Q[i] @ x[i] / 2 + g[i] @ x[i] for i in NODES])
    constraints = [cp.sum([A[i] @ x[i] for i in NODES]) - b <= 0]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)

    F_star = problem.value

    if SCRIPT_TYPE == "main":
        """
        Resource perturbation and distributed resource allocation
        """
        # epsilon = 0.5
        # delta = 0.005
        # Delta = 0.002

        # b_bar = pdra.resource_perturbation(epsilon, delta, Delta, b)

        config = DRAConfiguration(T, AGD(STEP_SIZE), "../data/dqp/result")
        gossip_network = create_gossip_network(NODES, EDGES)

        f = {
            i: lambda x_i, index=i: x_i @ Q[index] @ x_i / 2 + g[index] @ x_i
            for i in NODES
        }
        nodes = {
            i: NodeDQP(i, config, f[i], A[i], gossip_network[i])
            for i in NODES
            if i != "1"
        }
        nodes["1"] = NodeDQP("1", config, f["1"], A["1"], gossip_network["1"], b_i=b)

        begin = time.time()

        for node in nodes.values():
            node.start()

        for Node in nodes.values():
            node.join()

        end = time.time()

        print(f"Time: {end - begin}")

    elif SCRIPT_TYPE == "plot":
        """
        Plot the graph and the result
        """
        graph = nx.Graph()
        graph.add_nodes_from(NODES)
        graph.add_edges_from(EDGES)

        fig_1, ax_1 = plt.subplots(1, 1)
        ax_1.set_aspect(1)
        ax_1.set_ylim([-1.5, 1.5])

        nx.draw(graph, pos=NODES_POS, ax=ax_1, **OPTIONS)

        fig_1.savefig("../figures/dqp/fig_1.png", dpi=300, bbox_inches="tight")

        result = {
            node: np.load(f"../data/dqp/result/{node}_result.npz") for node in NODES
        }

        f_i_series = {node: result[node]["f_i_series"] for node in NODES}
        err_series = sum(f_i_series.values()) - F_star

        fig_2, ax_2 = plt.subplots()

        ax_2.plot(err_series, label="Distributed Quadratic Programming")
        ax_2.set_xlabel("Iteration")
        ax_2.set_ylabel("Error")
        ax_2.legend()

        fig_2.savefig("../figures/dqp/fig_2.png", dpi=300, bbox_inches="tight")
