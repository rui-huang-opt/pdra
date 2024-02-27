import threading
import numpy as np
from typing import List, Dict, Tuple
from .edge import Edge


class NodeException(Exception):
    pass


class Node(threading.Thread):
    def __init__(self, name: str, in_edges: List[Edge], out_edges: List[Edge]):
        super().__init__()

        self.__name = name

        self.__in_degree = len(in_edges)
        self.__out_degree = len(out_edges)

        self.__in_edges = in_edges
        self.__out_edges = out_edges

    @property
    def name(self):
        return self.__name

    @property
    def out_degree(self):
        return self.__out_degree

    @property
    def in_degree(self):
        return self.__in_degree

    @property
    def in_edges(self):
        return self.__in_edges

    @property
    def out_edges(self):
        return self.__out_edges

    def add_edges(self, edges: Edge or List[Edge]) -> None:
        edges_ = [edges] if isinstance(edges, Edge) else edges
        for edge in edges_:
            if edge.ending_point == self.__name:
                self.__in_edges.append(edge)
                self.__in_degree += 1
            elif edge.starting_point == self.__name:
                self.__out_edges.append(edge)
                self.__out_degree += 1
            else:
                raise NodeException(f'Edge {edge.name} can not be attached to node {self.__name}!')

    def delete_edges(self, edges: str or List[str]) -> None:
        edges_ = [edges] if isinstance(edges, Edge) else edges
        for edge in edges_:
            if edge.ending_point == self.__name and edge in self.__in_edges:
                self.__in_edges.remove(edge)
                self.__in_degree -= 1
            elif edge.starting_point == self.__name and edge in self.__out_edges:
                self.__out_edges.remove(edge)
                self.__out_degree -= 1
            else:
                raise NodeException(f'Edge {edge.name} is not attached to node {self.__name}!')

    def send_to_neighbors(self, data: np.ndarray) -> None:
        for edge in self.__out_edges:
            edge.put(data)

    def recv_from_neighbors(self) -> List[np.ndarray]:
        data = []
        for edge in self.__in_edges:
            data.append(edge.get())

        return data


def gen_communication_edges(conn: Dict[str, str]) -> Tuple[Dict[str, List[Edge]], Dict[str, List[Edge]]]:
    edges = []

    for node in conn.keys():
        for out_neighbor in conn[node]:
            edges.append(Edge(node, out_neighbor))

    in_edges = {node: [] for node in conn.keys()}
    out_edges = {node: [] for node in conn.keys()}
    for edge in edges:
        in_edges[edge.ending_point].append(edge)
        out_edges[edge.starting_point].append(edge)

    return in_edges, out_edges
