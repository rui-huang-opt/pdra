import numpy as np


class TruncatedLaplaceError(Exception):
    pass


class TruncatedLaplace(object):
    def __init__(self, low: int or float, high: int or float, location: int or float, scale: int or float):
        if low > high:
            raise TruncatedLaplaceError('The lower bound of the support cannot be greater than the upper bound!')
        if scale <= 0:
            raise TruncatedLaplaceError('The scale must be positive!')

        self.__low = low
        self.__high = high
        self.__location = location
        self.__scale = scale

        self.__k = self.get_k()
        self.__pdf_max = self.get_pdf_max()

    @property
    def low(self) -> int or float:
        return self.__low

    @low.setter
    def low(self, value: int or float) -> None:
        if value > self.__high:
            raise TruncatedLaplaceError('Can\'t set a lower bound greater than the upper bound')

        self.__low = value
        self.__k = self.get_k()
        self.__pdf_max = self.get_pdf_max()

    @property
    def high(self) -> int or float:
        return self.__high

    @high.setter
    def high(self, value: int or float) -> None:
        if value < self.__low:
            raise TruncatedLaplaceError('Can\'t set a upper bound less than the lower bound')

        self.__high = value
        self.__k = self.get_k()
        self.__pdf_max = self.get_pdf_max()

    @property
    def location(self) -> int or float:
        return self.__location

    @location.setter
    def location(self, value: int or float) -> None:
        self.__location = value
        self.__k = self.get_k()
        self.__pdf_max = self.get_pdf_max()

    @property
    def scale(self) -> int or float:
        return self.__scale

    @scale.setter
    def scale(self, value: int or float) -> None:
        if value <= 0:
            raise TruncatedLaplaceError('The scale must be positive!')

        self.__scale = value
        self.__k = self.get_k()
        self.__pdf_max = self.get_pdf_max()

    # Get the probability of a Laplacian random variable belonging to interval [low, high]
    def get_k(self) -> int or float:
        high_diff = self.__high - self.__location
        low_diff = self.__low - self.__location

        high_term = np.sign(high_diff) * (1 - np.exp(-np.abs(high_diff) / self.__scale))
        low_term = np.sign(low_diff) * (1 - np.exp(-np.abs(low_diff) / self.__scale))

        return (high_term - low_term) / 2

    # Probability Density Function
    def pdf(self, u) -> int or float:
        return np.exp(-np.abs(u - self.__location) / self.__scale) / (2 * self.__scale * self.__k)

    def get_pdf_max(self) -> int or float:
        val = max(self.__low, min(self.__high, self.__location))
        pdf_max = self.pdf(val)

        return pdf_max

    def get_random_scalar_value(self) -> int or float:
        # Acceptance-Rejection Method

        while True:
            rand1 = np.random.uniform(self.__low, self.__high)
            rand2 = np.random.uniform(0, self.__pdf_max)

            if self.pdf(rand1) > rand2:
                break

        return rand1

    def __call__(self, dim=1) -> int or float or np.ndarray:
        if not isinstance(dim, int) and dim >= 0:
            raise TruncatedLaplaceError("The model size must be positive integers.")

        if dim == 1:
            res = self.get_random_scalar_value()
        else:
            res = np.array([self.get_random_scalar_value() for _ in range(dim)])

        return res
