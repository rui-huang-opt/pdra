import os
import toml
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict
from numpy.typing import NDArray
from functools import partial
from gossip import create_gossip_network
from pdra import create_node, TruncatedLaplace

if __name__ == "__main__":
    """
    Load the configuration
    """
    configs = toml.load("../configs.toml")["dqp"]

    node_names = configs["node_names"]
    edge_pairs = configs["edge_pairs"]

    algorithm = configs["algorithm"]
    node_params = configs["node_params"]

    """
    Load the model of the problem
    """
    ## Generate the model as described in the paper
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
    # np.savez(configs["data_dir"] + "Q.npz", **Q)
    # np.savez(configs["data_dir"] + "g.npz", **g)
    # np.savez(configs["data_dir"] + "A.npz", **A)
    #
    # vec = np.random.uniform(0, 1, 3)
    # pr = vec / vec.sum()
    # D = np.random.choice([1, 2, 3], 1000, p=pr)
    # np.save(configs["data_path"] + "D.npy", D)

    Q: Dict[str, NDArray[np.float64]] = np.load(configs["data_dir"] + "Q.npz")
    g: Dict[str, NDArray[np.float64]] = np.load(configs["data_dir"] + "g.npz")
    A: Dict[str, NDArray[np.float64]] = np.load(configs["data_dir"] + "A.npz")

    D: NDArray[np.float64] = np.load(configs["data_dir"] + "D.npy")

    proportion_of_1 = D[np.where(D == 1)].size / 1000
    proportion_of_2 = D[np.where(D == 2)].size / 1000
    proportion_of_3 = D[np.where(D == 3)].size / 1000

    b = np.array([proportion_of_1, proportion_of_2, proportion_of_3])

    if configs["run_mode"] == "cen":
        """
        Centralized optimization
        """
        x = {i: cp.Variable(A[i].shape[1]) for i in node_names}

        cost = cp.sum([x[i] @ Q[i] @ x[i] / 2 + g[i] @ x[i] for i in node_names])
        constraints = [cp.sum([A[i] @ x[i] for i in node_names]) - b <= 0]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver="OSQP")

        opt_val = problem.value

        print(f"Centralized optimal value: {opt_val}")

        with open(f"../configs.toml", "w") as f:
            configs["opt_val"] = opt_val
            toml.dump(configs, f)

    elif configs["run_mode"] == "dis":
        """
        Resource perturbation
        """
        epsilon = 0.5
        delta = 0.005
        Delta = 0.002

        np.random.seed(0)

        if algorithm == "core":
            s = (Delta / epsilon) * np.log(b.size * (np.exp(epsilon) - 1) / delta + 1)
            tl = TruncatedLaplace(-s, s, 0, Delta / epsilon)
            perturbation = -s * np.ones(b.size) + tl.sample(b.size)
        elif algorithm == "rsdd":
            perturbation = np.random.laplace(0, Delta / epsilon, b.size)
        else:
            raise ValueError("Invalid algorithm")

        b_bar = b + perturbation

        """
        Distributed resource allocation
        """
        Node = create_node(algorithm)
        gossip_network = create_gossip_network(node_names, edge_pairs)

        def f(x: cp.Variable, index: str) -> cp.Expression:
            return x @ Q[index] @ x / 2 + g[index] @ x

        nodes = [
            Node(gossip_network[i], partial(f, index=i), A[i], **node_params[algorithm])
            for i in node_names
        ]

        nodes[0].set_resource(b_bar)

        for node in nodes:
            node.start()

        for node in nodes:
            node.join()

    elif configs["run_mode"] == "plot":
        fig_dir = configs["fig_dir"]
        os.makedirs(fig_dir, exist_ok=True)

        """
        Plot the graph and the result
        """
        graph = nx.Graph()
        graph.add_nodes_from(node_names)
        graph.add_edges_from(edge_pairs)

        fig1, ax1 = plt.subplots(1, 1)

        ax1.set_aspect(1)
        ax1.set_ylim([-1.5, 1.5])

        nx.draw(graph, ax=ax1, **configs["nx_options"])

        fig1.savefig(configs["fig_dir"] + "fig_1.png", dpi=300, bbox_inches="tight")
        fig1.savefig(
            configs["fig_dir"] + "fig_1.pdf", format="pdf", bbox_inches="tight"
        )

        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = "\\usepackage{amsmath}"
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Times New Roman"
        plt.rcParams["font.size"] = 15

        results = {
            "core": {
                i: np.load(node_params["core"]["results_path"] + f"/node_{i}.npz")
                for i in node_names
            },
            "rsdd": {
                i: np.load(node_params["rsdd"]["results_path"] + f"/node_{i}.npz")
                for i in node_names
            },
        }

        iterations = np.arange(1, node_params["core"]["max_iter"] + 1)

        fig2, ax2 = plt.subplots()

        err_series = {
            "core": sum([results["core"][i]["f_i_series"] for i in node_names])
            - configs["opt_val"],
            "rsdd": sum([results["rsdd"][i]["f_i_series"] for i in node_names])
            - configs["opt_val"],
        }

        ax2.step(
            iterations,
            err_series["core"],
            label="The proposed algorithm",
            color="tab:blue",
            linestyle="-",
        )

        ax2.step(
            iterations,
            err_series["rsdd"],
            label="RSDD",
            color="tab:orange",
            linestyle="--",
        )

        ax2.set_xlabel("Iteration number $k$")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.grid()

        fig2.savefig(configs["fig_dir"] + "fig_3_a.png", dpi=300, bbox_inches="tight")
        fig2.savefig(
            configs["fig_dir"] + "fig_3_a.pdf", format="pdf", bbox_inches="tight"
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
            i: results["core"][i]["c_i_series"] for i in node_names
        }

        for i in node_names:
            for j in range(c_series[i].shape[0]):
                (line,) = ax3.step(iterations, c_series[i][j], color=colors[i])

            line.set_label(f"node {i}")

        ax3.set_xlabel("Iteration number $k$")
        ax3.legend()
        ax3.grid()

        fig3.savefig(configs["fig_dir"] + "fig_3_b.png", dpi=300, bbox_inches="tight")
        fig3.savefig(
            configs["fig_dir"] + "fig_3_b.pdf", format="pdf", bbox_inches="tight"
        )

        fig4, ax4 = plt.subplots()

        constraint_values = {
            "core": sum([A[i] @ results["core"][i]["x_i_series"] for i in node_names])
            - b[:, np.newaxis],
            "rsdd": sum([A[i] @ results["rsdd"][i]["x_i_series"] for i in node_names])
            - b[:, np.newaxis],
        }

        for i in range(constraint_values["core"].shape[0]):
            ax4.step(
                iterations,
                constraint_values["core"][i],
                linestyle="-",
                color=colors[str(i + 1)],
                label=f"The proposed algorithm, constraint {i + 1}",
            )
            ax4.step(
                iterations,
                constraint_values["rsdd"][i],
                linestyle="--",
                color=colors[str(i + 1)],
                label=f"rsdd, constraint {i + 1}",
            )

        ax4.set_xlabel("Iteration number $k$")
        ax4.legend()
        ax4.grid()

        fig4.savefig(
            configs["fig_dir"] + "fig_3_c.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig4.savefig(
            configs["fig_dir"] + "fig_3_c.pdf", format="pdf", bbox_inches="tight"
        )

    else:
        raise ValueError("Invalid run mode")
