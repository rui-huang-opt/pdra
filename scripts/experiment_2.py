import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
from pdra import NodeDRABase
from gossip import create_gossip_network
from plot_options import SCRIPT_TYPE, OPTIONS


class NodeCP(NodeDRABase):
    def __init__(
        self,
        name,
        c_profit_i,
        x_laboratory_i,
        iterations,
        gamma,
        a_material_i,
        comm,
        **kwargs,
    ):
        self.c_profit_i = c_profit_i
        self.x_laboratory_i = x_laboratory_i

        result_dir = "../data/cp/result"

        # Set the solver to GLPK since it is more suitable for LP
        super().__init__(
            name,
            iterations,
            gamma,
            "SM",
            a_material_i,
            comm,
            result_dir=result_dir,
            solver=cp.GLPK,
            **kwargs,
        )

    @property
    def f_i(self) -> cp.Expression:
        return -self.c_profit_i @ self.x_i

    @property
    def local_constraints(self) -> List[cp.Constraint]:
        return [self.x_i - self.x_laboratory_i <= 0]


if __name__ == "__main__":
    N = 15
    NODES = {f"{i}" for i in range(1, N + 1)}
    EDGES = {
        ("1", "9"),
        ("1", "11"),
        ("2", "3"),
        ("2", "6"),
        ("2", "7"),
        ("2", "10"),
        ("6", "12"),
        ("3", "15"),
        ("4", "8"),
        ("5", "14"),
        ("8", "15"),
        ("11", "13"),
        ("11", "14"),
        ("13", "15"),
    }
    NODES_POS = {
        "1": (-3.0, 1.5),
        "2": (2.0, 1.0),
        "3": (1.0, 0.5),
        "4": (2.0, -1.0),
        "5": (-4.0, 0.0),
        "6": (1.0, 1.5),
        "7": (3.0, 0.5),
        "8": (1.0, -0.5),
        "9": (-4.0, 2.0),
        "10": (3.0, 1.5),
        "11": (-2.0, 1.0),
        "12": (0.0, 2.0),
        "13": (-1.0, 0.5),
        "14": (-3.0, 0.5),
        "15": (0.0, 0.0),
    }

    # Parameters initialization
    T = 3000
    step_size = 10

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
    problem.solve(solver=cp.GLPK)

    F_star = problem.value

    if SCRIPT_TYPE == "main":
        # Resource perturbation
        # epsilon = 0.5
        # delta = 0.005
        # Delta = 0.1

        # b_material_bar = pdra.resource_perturbation(epsilon, delta, Delta, b_material)

        # Distributed resource allocation
        gossip_network = create_gossip_network(NODES, EDGES)
        nodes = {
            i: NodeCP(
                i,
                c_pro[i],
                x_lab[i],
                T,
                step_size,
                A_mat[i],
                gossip_network[i],
            )
            for i in NODES
            if i != "1"
        }
        nodes["1"] = NodeCP(
            "1",
            c_pro["1"],
            x_lab["1"],
            T,
            step_size,
            A_mat["1"],
            gossip_network["1"],
            b_i=b_mat,
        )

        for node in nodes.values():
            node.start()

        for node in nodes.values():
            node.join()

    elif SCRIPT_TYPE == "plot":
        graph = nx.Graph()
        graph.add_nodes_from(NODES)
        graph.add_edges_from(EDGES)

        fig_1, ax_1 = plt.subplots(1, 1, figsize=(10, 5))
        ax_1.set_aspect(1)

        nx.draw(graph, pos=NODES_POS, ax=ax_1, **OPTIONS)

        fig_1.savefig("../figures/cp/fig_1.png", dpi=300, bbox_inches="tight")

        result = {
            node: np.load(f"../data/cp/result/{node}_result.npz") for node in NODES
        }

        f_i_series = {node: result[node]["f_i_series"] for node in NODES}
        err_series = sum(f_i_series.values()) - F_star

        fig_2, ax_2 = plt.subplots()

        ax_2.plot(err_series, label="collaborative production")
        ax_2.set_xlabel("Iteration")
        ax_2.set_ylabel("Error")
        ax_2.legend()

        fig_2.savefig("../figures/cp/fig_2.png", dpi=300, bbox_inches="tight")
