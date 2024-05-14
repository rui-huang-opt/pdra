import threading
import queue
import numpy as np
from typing import List


class Node(threading.Thread):
    def __init__(self):
        super().__init__()

        self.__in_edges: List['DiEdge'] = []
        self.__out_edges: List['DiEdge'] = []

        self.__in_degree: int = 0
        self.__out_degree: int = 0

    @property
    def in_degree(self) -> int:
        return self.__in_degree

    @property
    def out_degree(self) -> int:
        return self.__out_degree

    def append_out_edge(self, edge: 'DiEdge') -> None:
        self.__out_edges.append(edge)
        self.__out_degree += 1

    def append_in_edge(self, edge: 'DiEdge') -> None:
        self.__in_edges.append(edge)
        self.__in_degree += 1

    def remove_out_edge(self, edge: 'DiEdge') -> None:
        self.__out_edges.remove(edge)
        self.__out_degree -= 1

    def remove_in_edge(self, edge: 'DiEdge') -> None:
        self.__in_edges.remove(edge)
        self.__in_degree -= 1

    def send_to_neighbor(self, index: int, data: np.ndarray) -> None:
        self.__out_edges[index].send(data)

    def receive_from_neighbor(self, index: int) -> np.ndarray:
        return self.__in_edges[index].receive()

    def broadcast_to_all_neighbors(self, data: np.ndarray) -> None:
        for edge in self.__out_edges:
            edge.send(data)

    def receive_from_all_neighbors(self) -> List[np.ndarray]:
        return [edge.receive() for edge in self.__in_edges]


class DiEdge(queue.Queue):
    def __init__(self, from_node: 'Node', to_node: 'Node', noise_scale: int or float = None, maxsize: int = 1):
        self.__is_connected = False

        self.__from_node = from_node
        self.__to_node = to_node
        self.connect()

        self.__noise_scale = noise_scale

        super().__init__(maxsize=maxsize)

    @property
    def is_connected(self) -> bool:
        return self.__is_connected

    def connect(self) -> None:
        if not self.__is_connected:
            self.__from_node.append_out_edge(self)
            self.__to_node.append_in_edge(self)
            self.__is_connected = True

    def disconnect(self) -> None:
        if self.__is_connected:
            self.__from_node.remove_out_edge(self)
            self.__to_node.remove_in_edge(self)
            self.__is_connected = False

    def send(self, data: np.ndarray) -> None:
        if self.__noise_scale is None:
            self.put(data)
        else:
            noise = np.random.normal(scale=self.__noise_scale, size=data.size)
            self.put(data + noise)

    def receive(self) -> np.ndarray:
        return self.get()


class Edge:
    def __init__(self, node_1: 'Node', node_2: 'Node', noise_scale: int or float = None, maxsize: int = 1):
        self.__edge_1 = DiEdge(node_1, node_2, noise_scale, maxsize)
        self.__edge_2 = DiEdge(node_2, node_1, noise_scale, maxsize)

    @property
    def is_connected(self) -> bool:
        return self.__edge_1.is_connected and self.__edge_2.is_connected

    def connect(self) -> None:
        self.__edge_1.connect()
        self.__edge_2.connect()

    def disconnect(self) -> None:
        self.__edge_1.disconnect()
        self.__edge_2.disconnect()
