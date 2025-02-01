import numpy as np
from numpy.typing import NDArray
from numbers import Real
from typing import Union


class TruncatedLaplace(object):
    def __init__(
        self,
        low: Real,
        high: Real,
        location: Real,
        scale: Real,
    ):
        if low > high:
            pass
        if scale <= 0:
            pass

        self._low = low
        self._high = high
        self._location = location
        self._scale = scale

        self._normalization_constant = self.get_normalization_constant()
        self._pdf_max = self.get_max_pdf_value()

    def get_normalization_constant(self) -> Real:
        high_diff = self._high - self._location
        low_diff = self._low - self._location

        high_term = np.sign(high_diff) * (1 - np.exp(-np.abs(high_diff) / self._scale))
        low_term = np.sign(low_diff) * (1 - np.exp(-np.abs(low_diff) / self._scale))

        return (high_term - low_term) / 2

    def get_max_pdf_value(self) -> Real:
        val = max(self._low, min(self._high, self._location))
        pdf_max = self.pdf(val)

        return pdf_max

    def pdf(self, u: Real) -> Real:
        """
        Probability Density Function of the Truncated Laplace Distribution.
        """

        return np.exp(-np.abs(u - self._location) / self._scale) / (
            2 * self._scale * self._normalization_constant
        )

    def get_random_scalar_value(self) -> Real:
        """
        Generate a random number by the acceptance-rejection method.
        """

        while True:
            rand1 = np.random.uniform(self._low, self._high)
            rand2 = np.random.uniform(0, self._pdf_max)

            if self.pdf(rand1) > rand2:
                break

        return rand1

    def __call__(self, dim: int = 1) -> Union[Real, NDArray[np.float64]]:
        if dim == 1:
            res = self.get_random_scalar_value()
        else:
            res = np.array([self.get_random_scalar_value() for _ in range(dim)])

        return res
