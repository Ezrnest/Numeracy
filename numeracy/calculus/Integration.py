# Created by lyc at 2020/12/7 19:10
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


# f = lambda x: x ** 2
#
# print(simpsonMethod(f, 0, 1))
