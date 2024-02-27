import numpy as np
import warnings


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
    def low(self):
        return self.__low

    @low.setter
    def low(self, value: int or float):
        if value <= self.__high:
            self.__low = value
            self.__k = self.get_k()
            self.__pdf_max = self.get_pdf_max()
        else:
            warnings.warn('Can\'t set a lower bound greater than the upper bound')

    @property
    def high(self):
        return self.__high

    @high.setter
    def high(self, value: int or float):
        if value >= self.__low:
            self.__high = value
            self.__k = self.get_k()
            self.__pdf_max = self.get_pdf_max()
        else:
            warnings.warn('Can\'t set a upper bound less than the lower bound')

    @property
    def location(self):
        return self.__location

    @location.setter
    def location(self, value: int or float):
        self.__location = value
        self.__k = self.get_k()
        self.__pdf_max = self.get_pdf_max()

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value: int or float):
        if value > 0:
            self.__scale = value
            self.__k = self.get_k()
            self.__pdf_max = self.get_pdf_max()
        else:
            warnings.warn('The scale must be positive!')

    def get_k(self):
        high_diff = self.__high - self.__location
        low_diff = self.__low - self.__location

        high_term = np.sign(high_diff) * (1 - np.exp(-np.abs(high_diff) / self.__scale))
        low_term = np.sign(low_diff) * (1 - np.exp(-np.abs(low_diff) / self.__scale))

        return (high_term - low_term) / 2

    def pdf(self, u):
        return np.exp(-np.abs(u - self.__location) / self.__scale) / (2 * self.__scale * self.__k)

    def get_pdf_max(self):
        val = max(self.__low, min(self.__high, self.__location))
        pdf_max = self.pdf(val)

        return pdf_max

    def get_random_scalar_value(self, var=None):
        while True:
            rand1 = np.random.uniform(self.__low, self.__high)
            rand2 = np.random.uniform(0, self.__pdf_max)

            if self.pdf(rand1) > rand2:
                break

        return rand1

    def __call__(self, dim=1):
        if not isinstance(dim, int) and dim >= 0:
            raise TruncatedLaplaceError("The parameter size must be positive integers.")

        if dim == 1:
            res = self.get_random_scalar_value()
        else:
            res = np.array([self.get_random_scalar_value() for _ in range(dim)])

        return res
