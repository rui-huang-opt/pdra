import os
import numpy as np
import cvxpy as cp
import networkx as nx
import toml
import matplotlib.pyplot as plt
from functools import partial
from typing import Dict
from gossip import create_sync_network
from pdra import Node

if __name__ == "__main__":
    configs = toml.load("configs/distributed_qp.toml")

    node_names = configs["node_names"]
    edge_pairs = configs["edge_pairs"]

    algorithm = configs["algorithm"]
    node_params = configs["node_params"]

    if configs["run_type"] == "test":
        model_data: Dict[str, Dict[str, list]] = configs["model_data"]

        Q = {name: np.array(value) for name, value in model_data["Q"].items()}
        c = {name: np.array(value) for name, value in model_data["c"].items()}
        A = {name: np.array(value) for name, value in model_data["A"].items()}
        b = np.array(model_data["b"])

        def f(x: cp.Variable, index: str) -> cp.Expression:
            return x @ Q[index] @ x / 2 + c[index] @ x

        communicators = create_sync_network(node_names, edge_pairs)

        nodes = [
            Node.create(
                algorithm,
                communicators[i],
                partial(f, index=i),
                A[i],
                **node_params[algorithm],
            )
            for i in node_names
        ]

        nodes[0].set_resource(b)

        for node in nodes:
            node.start()

        for node in nodes:
            node.join()

    elif configs["run_type"] == "plot":
        fig1, ax1 = plt.subplots()

        G = nx.Graph()
        G.add_nodes_from(node_names)
        G.add_edges_from(edge_pairs)

        pos = configs["pos"]

        nx.draw(
            G,
            pos,
            ax=ax1,
            with_labels=True,
            node_size=1000,
            node_color="skyblue",
            font_size=10,
            font_color="black",
            font_weight="bold",
            edge_color="black",
            width=2,
            style="dashed",
        )

        fig1.savefig("figures/distributed_qp/graph.png", dpi=300, bbox_inches="tight")

        results_prefix = node_params[algorithm]["results_prefix"]
        results = {
            i: np.load(os.path.join(results_prefix, f"node_{i}.npz"))
            for i in node_names
        }

        fig2, ax2 = plt.subplots()

        err_series = (
            sum([results[i]["f_i_series"] for i in node_names])
            - configs["optimal_value"]
        )

        ax2.plot(err_series, label="Error")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Error")
        ax2.legend()
        ax2.grid(True)

        fig2.savefig(
            f"figures/distributed_qp/{algorithm}_error.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig3, ax3 = plt.subplots()
        computation_times = {i: results[i]["computation_time"] for i in node_names}
        for i in node_names:
            ax3.plot(computation_times[i], label=f"Node {i}")

        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Computation Time (s)")
        ax3.set_title("Computation Time per Node")
        ax3.grid(True)
        ax3.legend()

        fig3.savefig(
            f"figures/distributed_qp/{algorithm}_computation_time.png",
            dpi=300,
            bbox_inches="tight",
        )
