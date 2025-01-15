import toml
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict
from numpy.typing import NDArray
from functools import partial
from gossip import create_gossip_network
from pdra import Node

if __name__ == "__main__":
    """
    Load the configuration
    """
    config = toml.load("../config.toml")

    """
    Parameters
    """
    EXPERIMENT = "DQP"
    NODES = config[EXPERIMENT]["NODES"]
    EDGES = config[EXPERIMENT]["EDGES"]
    NODES_POS = config[EXPERIMENT]["NODES_POS"]
    NOED_CONFIG = config[EXPERIMENT]["NODE_CONFIG"]

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
    # np.save(rf'../data/{experiment}/model/D.npy', D)

    Q: Dict[str, NDArray[np.float64]] = np.load(f"../data/dqp/model/Q.npz")
    g: Dict[str, NDArray[np.float64]] = np.load(f"../data/dqp/model/g.npz")
    A: Dict[str, NDArray[np.float64]] = np.load(f"../data/dqp/model/A.npz")

    D: NDArray[np.float64] = np.load("../data/dqp/model/D.npy")

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
    problem.solve(solver="OSQP")

    F_star = problem.value

    if config["PLOT_MODE"]:
        """
        Plot the graph and the result
        """
        graph = nx.Graph()
        graph.add_nodes_from(NODES)
        graph.add_edges_from(EDGES)

        fig_1, ax_1 = plt.subplots(1, 1)
        ax_1.set_aspect(1)
        ax_1.set_ylim([-1.5, 1.5])

        nx.draw(graph, pos=NODES_POS, ax=ax_1, **config["GRPAH_PLOT_OPTIONS"])

        fig_1.savefig("../figures/dqp/fig_1.png", dpi=300, bbox_inches="tight")

        result = {i: np.load(f"../data/dqp/result/node_{i}.npz") for i in NODES}

        f_i_series = {node: result[node]["f_i_series"] for node in NODES}
        err_series = sum(f_i_series.values()) - F_star

        fig_2, ax_2 = plt.subplots()

        ax_2.plot(err_series, label="Distributed Quadratic Programming")
        ax_2.set_xlabel("Iteration")
        ax_2.set_ylabel("Error")
        ax_2.legend()

        fig_2.savefig("../figures/dqp/fig_2.png", dpi=300, bbox_inches="tight")

    else:
        """
        Resource perturbation and distributed resource allocation
        """
        # epsilon = 0.5
        # delta = 0.005
        # Delta = 0.002

        # b_bar = pdra.resource_perturbation(epsilon, delta, Delta, b)

        gossip_network = create_gossip_network(NODES, EDGES)

        def f(x: cp.Variable, index: str) -> cp.Expression:
            return x @ Q[index] @ x / 2 + g[index] @ x

        nodes = [
            Node(i, NOED_CONFIG, gossip_network[i], partial(f, index=i), A[i])
            for i in NODES
        ]

        nodes[0].set_resource(b)

        for node in nodes:
            node.start()

        for Node in nodes:
            node.join()
