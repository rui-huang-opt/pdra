from abc import ABCMeta, abstractmethod
from typing import Type, Dict
from numpy import float64, sqrt
from numpy.typing import NDArray


class GradientDescent(metaclass=ABCMeta):
    _registry: Dict[str, Type["GradientDescent"]] = {}

    def __init_subclass__(cls, key: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        _key = key or cls.__name__
        if _key in GradientDescent._registry:
            raise ValueError(f"Gradient method '{_key}' is already registered.")
        GradientDescent._registry[_key] = cls

    def __init__(self, step_size: float):
        self._step_size = step_size

    @classmethod
    def create(cls, step_size: float, key: str) -> "GradientDescent":
        return cls._registry[key](step_size)

    @abstractmethod
    def step(
        self, x: NDArray[float64], direction: NDArray[float64], k: int
    ) -> NDArray[float64]: ...


class Nesterov(GradientDescent, key="nesterov"):
    def __init__(self, step_size: float):
        super().__init__(step_size)

        self._w: NDArray[float64] | None = None
        self._theta: float = 1.0

    def step(self, x: NDArray[float64], gradient: NDArray[float64], k: int):
        old_w = x if self._w is None else self._w
        old_theta = self._theta

        self._w = x - self._step_size * gradient
        self._theta = (1 + sqrt(1 + 4 * (self._theta**2))) / 2
        new_x = self._w + ((old_theta - 1) / self._theta) * (self._w - old_w)

        return new_x


class SubgradientMethod(GradientDescent, key="subgrad"):
    def __init__(self, step_size: float):
        super().__init__(step_size)

    def step(self, x: NDArray[float64], subgradient: NDArray[float64], k: int):
        new_x = x - (self._step_size / sqrt(k + 1)) * subgradient

        return new_x
