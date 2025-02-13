import toml
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict
from numpy.typing import NDArray
from numpy.lib.npyio import NpzFile
from matplotlib.ticker import MultipleLocator
from functools import partial
from gossip import create_gossip_network
from pdra import Node, TruncatedLaplace

if __name__ == "__main__":
    config = toml.load("../config.toml")

    EXPERIMENT = "CP"
    NODES = config[EXPERIMENT]["NODES"]
    EDGES = config[EXPERIMENT]["EDGES"]
    NODES_POS = config[EXPERIMENT]["NODES_POS"]
    NODE_CONFIG = config[EXPERIMENT]["NODE_CONFIG"]

    # Generate the model of the problem
    # di = 2
    # M = 2
    #
    # c_profit = {}
    # A_material = {}
    # x_laboratory = {}
    #
    # for i in Nodes_set:
    #     c_profit[i] = np.random.uniform(1, 2, di)
    #     A_material[i] = np.random.uniform(1, 5, (M, di))
    #     x_laboratory[i] = np.random.uniform(1, 5, di)
    #
    # np.savez(rf'..\data\Collaborative Production\model\c_profit.npz', **c_profit)
    # np.savez(rf'..\data\Collaborative Production\model\A_material.npz', **A_material)
    # np.savez(rf'..\data\Collaborative Production\model\x_laboratory.npz', **x_laboratory)
    #
    # b_material = np.random.uniform(18, 22, M)
    # np.save(rf'..\data\Collaborative Production\model\b_material.npy', b_material)

    c_pro: NpzFile = np.load(f"../data/{EXPERIMENT}/c_pro.npz")
    A_mat: NpzFile = np.load(f"../data/{EXPERIMENT}/A_mat.npz")
    x_lab: NpzFile = np.load(f"../data/{EXPERIMENT}/x_lab.npz")

    b_mat: NDArray[np.float64] = np.load(f"../data/{EXPERIMENT}/b_mat.npy")

    if config["RUN_MODE"] == "CEN":
        """
        Centralized optimization
        """
        x = {i: cp.Variable(A_mat[i].shape[1]) for i in NODES}

        cost = cp.sum([-c_pro[i] @ x[i] for i in NODES])

        lab_constraints = [x[i] - x_lab[i] <= 0 for i in NODES]
        mat_constraints = [cp.sum([A_mat[i] @ x[i] for i in NODES]) - b_mat <= 0]

        constraints = mat_constraints + lab_constraints

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver="GLPK")

        F_star = problem.value

        print(f"Optimal value: {F_star}")

        with open(f"../config.toml", "w") as f:
            config[EXPERIMENT]["OPT_VAL"] = F_star
            toml.dump(config, f)

    elif config["RUN_MODE"] == "DIS":
        """
        Resource perturbation
        """
        epsilon = 0.5
        delta = 0.005
        Delta = 0.1

        s = (Delta / epsilon) * np.log(b_mat.size * (np.exp(epsilon) - 1) / delta + 1)
        tl = TruncatedLaplace(-s, s, 0, Delta / epsilon)
        b_mat_bar = b_mat - s * np.ones(b_mat.size) + tl.sample(b_mat.size)

        """
        Distributed resource allocation
        """

        def f(x_i: cp.Variable, index: str) -> cp.Expression:
            return -c_pro[index] @ x_i

        def g(x_i: cp.Variable, index: str) -> cp.Expression:
            return x_i - x_lab[index]

        gossip_network = create_gossip_network(NODES, EDGES)
        nodes = [
            Node(
                i,
                NODE_CONFIG,
                gossip_network[i],
                partial(f, index=i),
                A_mat[i],
                partial(g, index=i),
            )
            for i in NODES
        ]
        nodes[0].set_resource(b_mat_bar)

        for node in nodes:
            node.start()

        for node in nodes:
            node.join()

    elif config["RUN_MODE"] == "VIS":
        """
        Plot the graph and the result
        """
        graph = nx.Graph()
        graph.add_nodes_from(NODES)
        graph.add_edges_from(EDGES)

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_aspect(1)

        nx.draw(graph, pos=NODES_POS, ax=ax1, **config["GRPAH_PLOT_OPTIONS"])

        fig1.savefig(f"../figures/{EXPERIMENT}/fig_2.png", dpi=300, bbox_inches="tight")
        fig1.savefig(
            f"../figures/{EXPERIMENT}/fig_2.pdf", format="pdf", bbox_inches="tight"
        )

        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = "\\usepackage{amsmath}"
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 15

        iterations = np.arange(1, NODE_CONFIG["iterations"] + 1)

        results = {
            i: np.load(
                config[EXPERIMENT]["NODE_CONFIG"]["result_path"] + f"/node_{i}.npz"
            )
            for i in NODES
        }

        f_i_series = {i: results[i]["f_i_series"] for i in NODES}

        fig2, ax2 = plt.subplots()
        err_series = sum(f_i_series.values()) - config[EXPERIMENT]["OPT_VAL"]

        ax2.step(
            iterations,
            err_series,
            label=r"$P(\boldsymbol{x}^*)-P(\boldsymbol{x}^{(k)})$",
        )

        ax2.set_xlabel("Iteration number $k$")
        ax2.yaxis.set_major_locator(MultipleLocator(10))
        ax2.legend()

        fig2.savefig(
            f"../figures/{EXPERIMENT}/fig_4_a.png", dpi=300, bbox_inches="tight"
        )
        fig2.savefig(
            f"../figures/{EXPERIMENT}/fig_4_a.pdf", format="pdf", bbox_inches="tight"
        )

        fig3, ax3 = plt.subplots()
        ax3ins = ax3.inset_axes((0.2, 0.3, 0.6, 0.6))
        ax3ins.tick_params(axis="both", labelsize=15)

        c_series: Dict[str, NDArray[np.float64]] = {
            i: results[i]["c_i_series"] for i in NODES
        }

        for i in NODES:
            for j in range(c_series[i].shape[0]):
                ax3.step(iterations, c_series[i][j])
                ax3ins.step(iterations[2950:2999], c_series[i][j, 2950:2999])

        ax3.set_ylim(0, 6)
        ax3.set_xlabel("Iteration number $k$")

        fig3.savefig(
            f"../figures/{EXPERIMENT}/fig_4_b.png", dpi=300, bbox_inches="tight"
        )
        fig3.savefig(
            f"../figures/{EXPERIMENT}/fig_4_b.pdf", format="pdf", bbox_inches="tight"
        )

        fig4, ax4 = plt.subplots()
        x_series: Dict[str, NDArray[np.float64]] = {
            i: results[i]["x_i_series"] for i in NODES
        }

        constraint_values: NDArray[np.float64] = (
            sum([A_mat[i] @ x_series[i] for i in NODES]) - b_mat[:, np.newaxis]
        )

        for i in range(constraint_values.shape[0]):
            ax4.step(iterations, constraint_values[i], label=f"material {i + 1}")

        ax4.set_xlabel("Iteration number $k$")
        ax4.legend()

        fig4.savefig(
            f"../figures/{EXPERIMENT}/fig_4_c.png", dpi=300, bbox_inches="tight"
        )
        fig4.savefig(
            f"../figures/{EXPERIMENT}/fig_4_c.pdf", format="pdf", bbox_inches="tight"
        )
