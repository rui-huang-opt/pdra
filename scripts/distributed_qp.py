import toml
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict
from numpy.typing import NDArray
from functools import partial
from gossip import create_gossip_network
from pdra import Node, TruncatedLaplace

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
    NODE_CONFIG = config[EXPERIMENT]["NODE_CONFIG"]

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

    Q: Dict[str, NDArray[np.float64]] = np.load(f"../data/{EXPERIMENT}/model/Q.npz")
    g: Dict[str, NDArray[np.float64]] = np.load(f"../data/{EXPERIMENT}/model/g.npz")
    A: Dict[str, NDArray[np.float64]] = np.load(f"../data/{EXPERIMENT}/model/A.npz")

    D: NDArray[np.float64] = np.load(f"../data/{EXPERIMENT}/model/D.npy")

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

        fig1, ax1 = plt.subplots(1, 1)
        ax1.set_aspect(1)
        ax1.set_ylim([-1.5, 1.5])

        nx.draw(graph, pos=NODES_POS, ax=ax1, **config["GRPAH_PLOT_OPTIONS"])

        fig1.savefig(f"../figures/{EXPERIMENT}/fig_1.png", dpi=300, bbox_inches="tight")
        fig1.savefig(
            f"../figures/{EXPERIMENT}/fig_1.pdf", format="pdf", bbox_inches="tight"
        )

        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = "\\usepackage{amsmath}"
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Times New Roman"
        plt.rcParams["font.size"] = 15

        results = {
            i: np.load(f"../data/{EXPERIMENT}/results/node_{i}.npz") for i in NODES
        }

        iterations = np.arange(1, NODE_CONFIG["iterations"] + 1)

        f_i_series = {i: results[i]["f_i_series"] for i in NODES}

        fig2, ax2 = plt.subplots()
        err_series = sum(f_i_series.values()) - F_star

        ax2.step(
            iterations, err_series, label=r"$F(\boldsymbol{x}^{(k)})-F(\boldsymbol{x}^*)$"
        )

        ax2.set_xlabel("Iteration number $k$")
        ax2.legend()

        fig2.savefig(
            f"../figures/{EXPERIMENT}/fig_3_a.png", dpi=300, bbox_inches="tight"
        )
        fig2.savefig(
            f"../figures/{EXPERIMENT}/fig_3_a.pdf", format="pdf", bbox_inches="tight"
        )

        colors = {
            "1": "tab:blue",
            "2": "tab:orange",
            "3": "tab:green",
            "4": "tab:red",
            "5": "tab:purple",
            "6": "tab:brown",
            "7": "tab:pink",
            "8": "tab:gray",
            "9": "tab:olive",
        }

        fig3, ax3 = plt.subplots()
        c_series: Dict[str, NDArray[np.float64]] = {
            i: results[i]["c_i_series"] for i in NODES
        }

        for i in NODES:
            for j in range(c_series[i].shape[0]):
                line, = ax3.step(iterations, c_series[i][j], color=colors[i])

            line.set_label(f"node {i}")

        ax3.set_xlabel("Iteration number $k$")
        ax3.legend()

        fig3.savefig(
            f"../figures/{EXPERIMENT}/fig_3_b.png", dpi=300, bbox_inches="tight"
        )
        fig3.savefig(
            f"../figures/{EXPERIMENT}/fig_3_b.pdf", format="pdf", bbox_inches="tight"
        )

        fig4, ax4 = plt.subplots()
        x_series: Dict[str, NDArray[np.float64]] = {
            i: results[i]["x_i_series"] for i in NODES
        }

        constraint_values = sum([A[i] @ x_series[i] for i in NODES]) - b[:, np.newaxis]

        for i in range(constraint_values.shape[0]):
            ax4.step(iterations, constraint_values[i], label=f"constraint {i + 1}")

        ax4.set_xlabel("Iteration number $k$")
        ax4.legend()

        fig4.savefig(
            f"../figures/{EXPERIMENT}/fig_3_c.png", dpi=300, bbox_inches="tight"
        )
        fig4.savefig(
            f"../figures/{EXPERIMENT}/fig_3_c.pdf", format="pdf", bbox_inches="tight"
        )

    else:
        """
        Resource perturbation
        """
        epsilon = 0.5
        delta = 0.005
        Delta = 0.002

        np.random.seed(0)

        s = (Delta / epsilon) * np.log(b.size * (np.exp(epsilon) - 1) / delta + 1)
        truncated_laplace = TruncatedLaplace(-s, s, 0, Delta / epsilon)
        b_bar = b - s * np.ones(b.size) + truncated_laplace(b.size)

        """
        Distributed resource allocation
        """
        gossip_network = create_gossip_network(NODES, EDGES)

        def f(x: cp.Variable, index: str) -> cp.Expression:
            return x @ Q[index] @ x / 2 + g[index] @ x

        nodes = [
            Node(i, NODE_CONFIG, gossip_network[i], partial(f, index=i), A[i])
            for i in NODES
        ]

        nodes[0].set_resource(b_bar)

        for node in nodes:
            node.start()

        for node in nodes:
            node.join()
