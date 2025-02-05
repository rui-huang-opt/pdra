import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict
from numpy.typing import NDArray
from multiprocessing import Process
from gossip import create_gossip_network, Gossip


class Node(Process):
    alpha: float = 0.5
    iterations: int = 50
    results_path: str = "results/consensus/"

    def __init__(
        self, name: str, communication: Gossip, initial_state: NDArray[np.float64]
    ):
        super().__init__()

        self.name = name
        self.communication = communication
        self.state = np.tile(initial_state, (self.iterations + 1, 1))

    def run(self):
        for k in range(self.iterations):
            self.communication.broadcast(self.state[k])
            neighbors_state = self.communication.gather()

            consensus_error = self.communication.degree * self.state[k] - sum(
                neighbors_state
            )

            self.state[k + 1] = self.state[k] - self.alpha * consensus_error

        np.save(self.results_path + f"node_{self.name}.npy", self.state)


if __name__ == "__main__":
    nodes = ["1", "2", "3", "4", "5"]
    edges = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

    nodes_state: Dict[str, NDArray[np.float64]] = {
        "1": np.array([10.1, 20.2, 30.3]),
        "2": np.array([52.3, 42.2, 32.1]),
        "3": np.array([25.6, 35.5, 45.4]),
        "4": np.array([17.7, 27.6, 37.5]),
        "5": np.array([20.9, 30.8, 40.7]),
    }

    gossip_network = create_gossip_network(nodes, edges)

    consensus_nodes = [
        Node(name, gossip_network[name], nodes_state[name]) for name in nodes
    ]

    for node in consensus_nodes:
        node.start()

    for node in consensus_nodes:
        node.join()

    figure_path = "figures/consensus/"

    fig1, ax1 = plt.subplots()

    nodes_pos = {
        "1": (0, 0),
        "2": (1, 0),
        "3": (1, 1),
        "4": (0, 1),
        "5": (0.5, 0.5),
    }

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    nx.draw(
        G,
        nodes_pos,
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

    ax1.set_title("Graph G")

    fig1.savefig(figure_path + "graph.png", dpi=300, bbox_inches="tight")

    nodes_color = {
        "1": "red",
        "2": "blue",
        "3": "green",
        "4": "orange",
        "5": "purple",
    }

    fig2, ax2 = plt.subplots()

    states_dict: Dict[str, NDArray[np.float64]] = {
        "1": np.load("results/consensus/node_1.npy"),
        "2": np.load("results/consensus/node_2.npy"),
        "3": np.load("results/consensus/node_3.npy"),
        "4": np.load("results/consensus/node_4.npy"),
        "5": np.load("results/consensus/node_5.npy"),
    }

    for name in nodes:
        for i in range(3):
            (line,) = ax2.plot(
                states_dict[name][:, i],
                color=nodes_color[name],
            )

        line.set_label(f"Node {name}")

    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("State")

    ax2.legend(loc="upper right")

    ax2.set_title("Consensus")

    fig2.savefig(figure_path + "consensus.png", dpi=300, bbox_inches="tight")
