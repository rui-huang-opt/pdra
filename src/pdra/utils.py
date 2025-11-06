from typing import TypedDict
from typing_extensions import NotRequired
from numpy import float64
from numpy.typing import NDArray


class IterLoggerRecord(TypedDict):
    """
    Record structure for each iteration log.

    Attributes
    ----------
    f_i : float
        The value of the local objective function at the current iteration.
    x_i : NDArray[float64]
        The current solution vector at the local node.
    overhead : float
        The overhead time for the current iteration.
    """

    f_i: float
    x_i: NDArray[float64]
    overhead: NotRequired[float]


from pandas import DataFrame


class IterLogger:
    def __init__(self):
        self._buffer: list[IterLoggerRecord] = []

    def log(self, f_i: float, x_i: NDArray[float64], overhead: float | None = None):
        record: IterLoggerRecord = {"f_i": f_i, "x_i": x_i}

        if overhead is not None:
            record["overhead"] = overhead

        self._buffer.append(record)

    def to_dataframe(self) -> DataFrame:
        df = DataFrame(self._buffer)
        if "x_i" in df:
            x_cols = DataFrame(df["x_i"].to_list()).add_prefix("x_")
            df = df.drop(columns=["x_i"]).join(x_cols)
        return df
