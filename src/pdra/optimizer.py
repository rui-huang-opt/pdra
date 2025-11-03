from abc import ABCMeta, abstractmethod
from typing import Type, Dict
from numpy import float64, sqrt
from numpy.typing import NDArray


class GradientMethod(metaclass=ABCMeta):
    _registry: Dict[str, Type["GradientMethod"]] = {}

    def __init_subclass__(cls, key: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        _key = key or cls.__name__
        if _key in GradientMethod._registry:
            raise ValueError(f"Gradient method '{_key}' is already registered.")
        GradientMethod._registry[_key] = cls

    def __init__(self, step_size: float):
        self._step_size = step_size

    @abstractmethod
    def __call__(
        self, x: NDArray[float64], direction: NDArray[float64]
    ) -> NDArray[float64]: ...

    @classmethod
    def create(cls, name: str, step_size: float) -> "GradientMethod":
        return cls._registry[name](step_size)


class AcceleratedGradientMethod(GradientMethod, key="AGM"):
    def __init__(self, step_size: float):
        super().__init__(step_size)

        self._w: NDArray[float64] | None = None
        self._theta: float = 1.0

    def __call__(self, x: NDArray[float64], gradient: NDArray[float64]):
        old_w = x if self._w is None else self._w
        old_theta = self._theta

        self._w = x - self._step_size * gradient
        self._theta = (1 + sqrt(1 + 4 * (self._theta**2))) / 2
        new_x = self._w + ((old_theta - 1) / self._theta) * (self._w - old_w)

        return new_x


class SubgradientMethod(GradientMethod, key="SM"):
    def __init__(self, step_size: float):
        super().__init__(step_size)

        self._decay_factor: float = 1.0

    def __call__(self, x: NDArray[float64], subgradient: NDArray[float64]):
        new_x = x - self._step_size * self._decay_factor * subgradient
        self._decay_factor = sqrt(1 - 1 / (self._decay_factor**2 + 1))

        return new_x
