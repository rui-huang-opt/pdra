import threading
import queue
import numpy as np
from typing import List


class Node(threading.Thread):
    def __init__(self):
        super().__init__()

        self.__in_edges: List['Edge'] = []
        self.__out_edges: List['Edge'] = []

        self.__in_degree: int = 0
        self.__out_degree: int = 0

    @property
    def out_degree(self) -> int:
        return self.__out_degree

    @property
    def in_degree(self) -> int:
        return self.__in_degree

    def append_out_edge(self, edge: 'Edge') -> None:
        self.__out_edges.append(edge)
        self.__out_degree += 1

    def append_in_edge(self, edge: 'Edge') -> None:
        self.__in_edges.append(edge)
        self.__in_degree += 1

    def remove_out_edge(self, edge: 'Edge') -> None:
        self.__out_edges.remove(edge)
        self.__out_degree -= 1

    def remove_in_edge(self, edge: 'Edge') -> None:
        self.__in_edges.remove(edge)
        self.__in_degree -= 1

    def send_to_neighbors(self, data: np.ndarray) -> None:
        for edge in self.__out_edges:
            edge.put(data)

    def recv_from_neighbors(self) -> List[np.ndarray]:
        return [edge.get() for edge in self.__in_edges]


class Edge(queue.Queue):
    def __init__(self, from_node: 'Node', to_node: 'Node', maxsize=1):
        self.__is_connected = False

        self.__from_node = from_node
        self.__to_node = to_node
        self.connect()

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
