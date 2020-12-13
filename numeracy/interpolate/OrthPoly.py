# Created by lyc at 2020/12/7 19:20
import math

import numpy as np


def weightLegendre(x):
    return 1.0


def normSqLegendre(n):
    return 2 / (2 * n + 1)


def genLegendre(unit=False):
    """
    返回生成 Legendre 多项式的迭代器。

    :return:
    """

    p0 = np.poly1d([1.0])
    x = np.poly1d([1.0, 0.0])
    p1 = x
    # (n+1)P_{n+1} = (2n+1) x P_n - n P_{n-1}
    if unit:
        yield p0 / math.sqrt(normSqLegendre(0))
        yield p1 / math.sqrt(normSqLegendre(1))
    else:
        yield p0
        yield p1
    n = 1
    while True:
        p2 = (2 * n + 1) * x * p1 - n * p0
        p2 /= (n+1)
        if unit:
            yield p2 / math.sqrt(normSqLegendre(n+1))
        else:
            yield p2
        p0 = p1
        p1 = p2
        n += 1


# g = genLegendre()
# for i in range(5):
#     print(next(g))

def weightTchebychev(x):
    return 1 / math.sqrt(1 - x ** 2)


def normSqTchebychev(n):
    if n == 0:
        return math.pi
    else:
        return math.pi / 2


def genTchebychev(unit=False):
    """
    返回生成 Tchebychev 多项式的迭代器。

    :return:
    """

    p0 = np.poly1d([1.0])
    x = np.poly1d([1.0, 0.0])
    p1 = x
    # (n+1)P_{n+1} = (2n+1) x P_n - n P_{n-1}
    if unit:
        yield p0 / math.sqrt(normSqTchebychev(0))
        yield p1 / math.sqrt(normSqTchebychev(1))
    else:
        yield p0
        yield p1
    n = 1
    while True:
        p2 = 2 * x * p1 - p0
        if unit:
            yield p2 / math.sqrt(normSqTchebychev(n+1))
        else:
            yield p2
        p0 = p1
        p1 = p2
        n += 1


def orthPolyInterpolate(f, a, b, rho, genPolyUnit, n=5):
    import numeracy.calculus.Integration as Int
    coes = np.zeros((n + 1,), np.float)
    result = np.poly1d([0.0])
    for k in range(n + 1):
        p = next(genPolyUnit)
        g = lambda x: f(x) * p(x) * rho(x)
        c = Int.simpsonMethod(g, a, b)
        coes[k] = c
        result += c * p

    return result, coes


def interLegendre(f, n=5):
    return orthPolyInterpolate(f, -1, 1, weightLegendre, genLegendre(True), n)


def interTchebychev(f, n=5):
    return orthPolyInterpolate(f, -0.999, 0.999, weightTchebychev, genTchebychev(True), n)


