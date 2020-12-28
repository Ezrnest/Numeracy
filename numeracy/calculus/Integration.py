# Created by lyc at 2020/12/7 19:10
import math

import numpy as np


def simpsonMethod(f, a, b, n=100):
    """
    使用 Simpson 方法求积分。

    :param f:
    :param a:
    :param b:
    :param n:
    :return:
    """
    xs = np.linspace(a, b, n + 1)
    h = (b - a) / n
    result = 0.0
    for k in range(n):
        x1 = xs[k]
        x2 = xs[k + 1]
        m = (x1 + x2) / 2
        result += f(x1) + 4 * f(m) + f(x2)
    return result * h / 6


# LEGENDRE_ROOT_COE = [
#     None,
#     None,
#     ([-1 / math.sqrt(3), 1 / math.sqrt(3)], [1, 1]),
#
# ]

# def


def intGaussLegendre(f, a, b, n=4, m=3):
    """
    使用 Gauss-Legendre 公式进行数值积分。此方法将区间分成 n 份，将每个子区间映射到[-1,1]后，使用
    m 阶 Legendre 多项式零点作为插值节点计算积分。

    :param f:
    :param a:
    :param b:
    :param n:
    :param m:
    :return:
    """
    x0, w = np.polynomial.legendre.leggauss(m)
    # xs = np.linspace(a, b, n + 1)
    h = (b - a) / n / 2
    res = 0.0
    for k in range(n):
        mid = a + (2 * k + 1) * h
        for r in range(m):
            x = mid + h * x0[r]
            y = f(x)
            res += y * w[r]
    return res * h


def intRomberg(f, a, b, eps=1E-7, maxIter=5):
    """
    使用 Romberg 方法进行数值积分。
    此方法使用减半步长加密网格，计算近似积分 T[0][k]，步长为 2^{-k}(b-a)。
    再结合 Richardson 外推方法得到诸阶近似积分 T[r][k]。
    当 |T[r][0] - T[r-1][1]| < eps 时得到结果。

    :param f:
    :param a:
    :param b:
    :param eps:
    :return: 近似积分，(r,T)
    """
    T = [[]]
    h = (b - a)
    T0 = T[0]
    T0.append(h / 2 * (f(a) + f(b)))
    r = 0
    while r < maxIter:
        H = 0.0
        for l in range(2 ** r):
            x = a + (l + 0.5) * h
            H += f(x)
        t = (T0[r] + h * H) / 2
        T0.append(t)

        # compute higher Ts.
        T.append([])
        r += 1
        for i in range(r):
            Ti = T[i]
            k = 4 ** (i + 1)
            t = k * Ti[r - i] - Ti[r - i - 1]
            t /= (k - 1)
            T[i + 1].append(t)
        if abs(T[r][0] - T[r - 1][1]) < eps:
            break
        h /= 2
    return T[r][0], (r,T)
    pass


# f = lambda x: x ** 2
#
# print(simpsonMethod(f, 0, 1))
# f = lambda x: 1 / (x ** 2) * math.sin(2 * math.pi / x)
# print(intGaussLegendre(f, 1, 3, 4, 5))
# print(intRomberg(f, 1, 3,maxIter=10))
