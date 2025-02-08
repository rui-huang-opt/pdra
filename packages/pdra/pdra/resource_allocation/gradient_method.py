import numpy as np
from numpy.typing import NDArray
from numbers import Real
from abc import ABCMeta, abstractmethod
from typing import Type, Dict


# Gradient-based method
class GradientMethod(metaclass=ABCMeta):
    """
    Abstract class for gradient-based methods, including accelerated gradient method and subgradient method.

    Arguments
    ----------
    step_size : Real
        Step size for the update of the variable.
    """

    def __init__(self, step_size: Real):
        self.step_size = step_size

    @abstractmethod
    def __call__(
        self, x: NDArray[np.float64], direction: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...


class GradientMethodRegistry:
    """
    Registry for gradient-based methods.
    """

    _methods: Dict[str, Type[GradientMethod]] = {}

    @classmethod
    def register(cls, name: str) -> Type[GradientMethod]:
        def decorator(grad_method: Type[GradientMethod]) -> Type[GradientMethod]:
            cls._methods[name] = grad_method
            return grad_method

        return decorator

    @classmethod
    def get(cls, name: str, step_size: Real) -> GradientMethod:
        if name not in cls._methods:
            raise ValueError(f"Unknown gradient method: {name}")

        return cls._methods[name](step_size)


# Accelerated gradient method
@GradientMethodRegistry.register("AGM")
class AcceleratedGradientMethod(GradientMethod):
    """
    Accelerated gradient method.
    """

    def __init__(self, step_size: Real):
        super().__init__(step_size)

        self.w = None
        self.theta = 1

    def __call__(self, x: NDArray[np.float64], gradient: NDArray[np.float64]):
        old_w = x if self.w is None else self.w
        old_theta = self.theta

        self.w = x - self.step_size * gradient
        self.theta = (1 + np.sqrt(1 + 4 * (self.theta**2))) / 2
        new_x = self.w + ((old_theta - 1) / self.theta) * (self.w - old_w)

        return new_x


# Subgradient method
@GradientMethodRegistry.register("SM")
class SubgradientMethod(GradientMethod):
    """
    Subgradient method.
    """
    
    def __init__(self, step_size: Real):
        super().__init__(step_size)

        self.decay_factor = 1

    def __call__(self, x: NDArray[np.float64], subgradient: NDArray[np.float64]):
        new_x = x - self.step_size * self.decay_factor * subgradient
        self.decay_factor = np.sqrt(
            1 - 1 / (self.decay_factor**2 + 1)
        )  # step size is gamma / sqrt(k + 1)

        return new_x
