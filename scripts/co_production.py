import toml
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from functools import partial
from gossip import create_gossip_network
from pdra import Node

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

    c_pro = np.load(f"../data/cp/model/c_pro.npz")
    A_mat = np.load(f"../data/cp/model/A_mat.npz")
    x_lab = np.load(f"../data/cp/model/x_lab.npz")

    b_mat = np.load(f"../data/cp/model/b_mat.npy")

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

    if config["PLOT_MODE"]:
        graph = nx.Graph()
        graph.add_nodes_from(NODES)
        graph.add_edges_from(EDGES)

        fig_1, ax_1 = plt.subplots(1, 1, figsize=(10, 5))
        ax_1.set_aspect(1)

        nx.draw(graph, pos=NODES_POS, ax=ax_1, **config["GRPAH_PLOT_OPTIONS"])

        fig_1.savefig("../figures/cp/fig_1.png", dpi=300, bbox_inches="tight")

        result = {
            i: np.load(f"../data/cp/result/node_{i}.npz") for i in NODES
        }

        f_i_series = {i: result[i]["f_i_series"] for i in NODES}
        err_series = sum(f_i_series.values()) - F_star

        fig_2, ax_2 = plt.subplots()

        ax_2.plot(err_series, label="collaborative production")
        ax_2.set_xlabel("Iteration")
        ax_2.set_ylabel("Error")
        ax_2.legend()

        fig_2.savefig("../figures/cp/fig_2.png", dpi=300, bbox_inches="tight")

    else:
        # Resource perturbation
        # epsilon = 0.5
        # delta = 0.005
        # Delta = 0.1

        # b_material_bar = pdra.resource_perturbation(epsilon, delta, Delta, b_material)

        def f(x_i: cp.Variable, index: str) -> cp.Expression:
            return -c_pro[index] @ x_i

        def g(x_i: cp.Variable, index: str) -> cp.Expression:
            return x_i - x_lab[index]

        # Distributed resource allocation
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
        nodes[0].set_resource(b_mat)

        for node in nodes:
            node.start()

        for node in nodes:
            node.join()
