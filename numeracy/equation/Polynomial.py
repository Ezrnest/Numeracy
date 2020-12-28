# Created by lyc at 2020/11/8 18:43

import numpy as np


def qinAlgo(p: np.poly1d, a):
    """
    使用秦九韶算法计算多项式值 p(x)，返回计算得到的诸 `s_k`。


    其中 `s[0] = p[n], s[k] = p[n-k]+s[k-1] a`

    :param p: 多项式，`p(x)= p[n]x^n + ... + p[0]`
    :param a: 需要计算的值
    :return: 以列表形式的 `s`。
    """
    n = p.order
    s = p[n]
    s_list = [s]
    for i in range(n):
        s = p[n - i - 1] + s * a
        s_list.append(s)
    return s_list


def pseudoDivide(p, a):
    """
    计算多项式带余除法 `p(x) / (x-a)` 的商式。此方法使用秦九韶算法以保证数值稳定性。

    :param p:
    :param a:
    :return:
    """
    S = qinAlgo(p, a)
    return np.poly1d(S[:-1])



def findRoots(p: np.poly1d, x0, eps=1E-7, maxIter=10):
    """
    计算多项式的全部实根，要求给定的多项式具有实的单根。

    :param p:
    :param x0:
    :return:
    """
    from numeracy.equation import Equation
    xs = []
    # p.roots
    p0 = p
    while p.order > 0:
        p1 = np.polyder(p)
        x1, _ = Equation.newtonMethod(p, p1, x0, eps, maxIter)
        xs.append(x1)
        x0 = x1
        p = pseudoDivide(p, x0)
    p1 = np.polyder(p0)
    for i in range(len(xs)):
        x0 = xs[i]
        x1, _ = Equation.newtonMethod(p0, p1, x0, eps, maxIter)
        xs[i] = x1
    return xs

#
# p = np.poly1d([1, -5, 6])
# # print(p)
# # print(qinAlgo(p, 2))
# print(findRoots(p, 2.2))
