from typing import Protocol, KeysView
from numpy import float64
from numpy.typing import NDArray


class NetworkOps(Protocol):
    """
    Protocol for communication operations in distributed optimization.

    In our examples, we use 'topolink.NodeHandle' as a concrete implementation of this protocol.
    The source code of 'topolink.NodeHandle' can be found at:
        https://github.com/rui-huang-opt/topolink.

    However, any class that implements these methods and properties can be used as long as it adheres to this protocol.

    Attributes
    ----------
    name : str
        Name of the node.

    num_neighbors : int
        Number of neighbor nodes.

    neighbor_names : KeysView[str]
        Names of all neighbor nodes.

    neighbor_weights : dict[str, float]
        Weights associated with each neighbor node.
        If no weights were specified during graph creation, 1.0 is used for all neighbors.
    """

    @property
    def name(self) -> str: ...

    @property
    def num_neighbors(self) -> int: ...

    @property
    def neighbor_names(self) -> KeysView[str]: ...

    @property
    def neighbor_weights(self) -> dict[str, float]: ...

    def exchange_map(
        self, state_map: dict[str, NDArray[float64]]
    ) -> dict[str, NDArray[float64]]:
        """
        Exchanges the given state map with all neighbor nodes.

        This method broadcasts the state map to all neighbors and then gathers their states.

        Args:
            state_map (dict[str, NDArray[np.float64]]): The state map to exchange with neighbors.

        Returns:
            dict[str, NDArray[np.float64]]: A dictionary mapping neighbor names to their received state maps.
        """
        ...

    def exchange(self, state: NDArray[float64]) -> dict[str, NDArray[float64]]:
        """
        Exchanges the given state with all neighbor nodes.

        This method broadcasts the state to all neighbors and then gathers their states.

        Args:
            state (NDArray[np.float64]): The state array to exchange with neighbors.

        Returns:
            dict[str, NDArray[np.float64]]: A dictionary mapping neighbor names to their received state arrays.
        """
        ...

    def laplacian(self, state: NDArray[float64]) -> NDArray[float64]:
        """
        Computes the Laplacian of the given state vector based on the states of neighboring nodes.

        The Laplacian is calculated as:

            laplacian = state * number_of_neighbors - sum_of_neighbor_states

        Args:
            state (NDArray[float64]): The state vector of the current node.

        Returns:
            NDArray[float64]: The Laplacian vector representing the difference between the current state and the average state of its neighbors.
        """
        ...

    def weighted_mix(self, state: NDArray[float64]) -> NDArray[float64]:
        """
        Performs the weighted mixing operation for distributed optimization using the weight matrix W.

        For a given node i, the mixed state is computed as the i-th row of Wx, where x is the stacked state vector of all nodes.
        If x_i is multi-dimensional, the operation is applied element-wise.
        Specifically:

            mixed_state = W_ii * state + sum_j(W_ij * neighbor_state_j)

        where W_ii is self._weight and W_ij are the weights in self._neighbor_weights.

        Args:
            state (NDArray[np.float64]): The current state vector of node i.

        Returns:
            NDArray[float64]: The mixed state vector corresponding to the i-th row of Wx.
        """
        ...
