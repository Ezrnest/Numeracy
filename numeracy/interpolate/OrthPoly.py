# Created by lyc at 2020/12/7 19:20
import math
from typing import Iterator

import numpy as np


class OrthPoly:
    """
    描述一类正交多项式

    """

    def __init__(self, name) -> None:
        self.name = name

    def weight(self, x):
        """
        返回正交多项式对应的权函数在 `x` 的取值。

        :param x:
        :return:
        """
        pass

    def weightFunc(self):
        """
        返回正交多项式对应的权函数。

        :return:
        """
        return self.weight

    def normSq(self, n):
        """
        返回标准（未归一化的）n-阶正交多项式的积分范数的平方。

        :param n:
        :return:
        """

        pass

    def generator(self, unit=False) -> Iterator[np.poly1d]:
        """
        返回生成此类多项式的迭代器。

        :return:
        """
        pass


class _Legendre(OrthPoly):

    def __init__(self) -> None:
        super().__init__("Legendre")

    def weight(self, x):
        return 1.0

    def normSq(self, n):
        return 2 / (2 * n + 1)

    def generator(self, unit=False):
        p0 = np.poly1d([1.0])
        x = np.poly1d([1.0, 0.0])
        p1 = x
        # (n+1)P_{n+1} = (2n+1) x P_n - n P_{n-1}
        if unit:
            yield p0 / math.sqrt(self.normSq(0))
            yield p1 / math.sqrt(self.normSq(1))
        else:
            yield p0
            yield p1
        n = 1
        while True:
            p2 = (2 * n + 1) * x * p1 - n * p0
            p2 /= (n + 1)
            if unit:
                yield p2 / math.sqrt(self.normSq(n + 1))
            else:
                yield p2
            p0 = p1
            p1 = p2
            n += 1


class _Tchebychev(OrthPoly):

    def __init__(self) -> None:
        super().__init__("Tchebychev")

    def weight(self, x):
        return 1 / math.sqrt(1 - x ** 2)

    def normSq(self, n):
        if n == 0:
            return math.pi
        else:
            return math.pi / 2

    def generator(self, unit=False):
        p0 = np.poly1d([1.0])
        x = np.poly1d([1.0, 0.0])
        p1 = x
        # (n+1)P_{n+1} = (2n+1) x P_n - n P_{n-1}
        if unit:
            yield p0 / math.sqrt(self.normSq(0))
            yield p1 / math.sqrt(self.normSq(1))
        else:
            yield p0
            yield p1
        n = 1
        while True:
            p2 = 2 * x * p1 - p0
            if unit:
                yield p2 / math.sqrt(self.normSq(n + 1))
            else:
                yield p2
            p0 = p1
            p1 = p2
            n += 1


Legendre = _Legendre()
Tchebychev = _Tchebychev()


def orthPolyInterpolate(f, a, b, orthPoly: OrthPoly, n=5):
    """
    在给定区间上使用正交多项式进行插值，返回插值多项式以及在正交多项式基上的系数。

    :param f:
    :param a:
    :param b:
    :param orthPoly: 正交多项式，要对应给定的区间。
    :param n: 插值阶数
    :return: (插值多项式，基上对应系数)
    """

    import numeracy.calculus.Integration as Int
    coes = np.zeros((n + 1,), np.float)
    result = np.poly1d([0.0])
    genUnit = orthPoly.generator(True)
    rho = orthPoly.weight
    for k in range(n + 1):
        p = next(genUnit)
        g = lambda x: f(x) * p(x) * rho(x)
        c = Int.simpsonMethod(g, a, b)
        coes[k] = c
        result += c * p

    return result, coes


def interLegendre(f, n=5):
    return orthPolyInterpolate(f, -1, 1, Legendre, n)


def interTchebychev(f, n=5):
    # TODO: 使用更好的积分方式代替奇异积分
    return orthPolyInterpolate(f, -0.999, 0.999, Tchebychev, n)


# f = lambda x: 1 / (x + 2)
# print(interTchebychev(f))
