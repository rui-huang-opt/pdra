from typing import Protocol
from numpy import float64
from numpy.typing import NDArray


class NetworkOps(Protocol):
    """
    Protocol for communication operations in distributed optimization.

    Attributes
    ----------
    name : str
        Name of the node.

    num_neighbors : int
        Number of neighboring nodes.

    neighbor_names : list[str]
        List of names of neighboring nodes.

    Methods
    -------
    send_each(state_by_neighbor: dict[str, NDArray[float64]]) -> None
        Sends different state arrays to each specified neighbor node.

    broadcast(state: NDArray[float64]) -> None
        Broadcasts the given state to all neighbor nodes.

    gather() -> dict[str, NDArray[float64]]
        Gathers data from all neighbors.

    weighted_gather() -> dict[str, NDArray[float64]]
        Gathers data from all neighbors, applying corresponding weights to each received array.

    laplacian(state: NDArray[float64]) -> NDArray[float64]
        Compute the Laplacian of the input state across neighboring nodes.

    weighted_mix(state: NDArray[float64]) -> NDArray[float64]
        Perform a weighted mixing of the input state across neighboring nodes.
    """

    @property
    def name(self) -> str: ...

    @property
    def num_neighbors(self) -> int: ...

    @property
    def neighbor_names(self) -> list[str]: ...

    def send_each(self, state_by_neighbor: dict[str, NDArray[float64]]): ...

    def broadcast(self, state: NDArray[float64]): ...

    def gather(self) -> dict[str, NDArray[float64]]: ...

    def weighted_gather(self) -> dict[str, NDArray[float64]]: ...

    def laplacian(self, state: NDArray[float64]) -> NDArray[float64]: ...

    def weighted_mix(self, state: NDArray[float64]) -> NDArray[float64]: ...
