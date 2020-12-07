# Created by lyc at 2020/11/22 14:15

"""
多项式插值

"""

import numpy as np


def mdiff(xs, fs, k):
    """
    计算直至k阶的均差，返回长度为 k+1 的列表，第 i 个元素为 i 阶均差。0 阶均差就是原函数值。

    :param xs:
    :param fs:
    :param k:
    :return:
    """
    n = len(xs)
    results = [np.array(fs)]
    for r in range(1, k + 1):
        prev = results[r - 1]
        d = np.zeros(n - r, prev.dtype)
        for i in range(0, n - r):
            d[i] = (prev[i + 1] - prev[i]) / (xs[i + r] - xs[i])
        results.append(d)
    return results


def interLagrange(f, xs):
    """
    使用 Lagrange 多项式进行插值。

    :param f: 待插值的函数
    :param xs: 分点，要求互不相同
    :return: 插值得到的函数对象（一个多项式）
    """
    import numpy.polynomial.polynomial as poly
    result = poly.polyzero
    n = len(xs)
    roots = np.copy(xs[1:])
    for k in range(n):
        coe = 1.0
        for i in range(n):
            if i == k:
                continue
            d = xs[k] - xs[i]
            coe *= d
        l = poly.polyfromroots(roots)[::-1]
        if k < n - 1:
            roots[k] = xs[k]
        l = l * f(xs[k]) / coe
        result = np.polyadd(result, l)
    return np.poly1d(result)


def interLinearPiece(f, xs):
    """
    使用分段线性插值 `f`。

    :param f: 待插值函数
    :param xs: 分点，要求互不相同
    :return: 插值得到的函数对象
    """

    n = len(xs)
    xs = np.sort(xs)
    deltas = np.diff(xs)
    fs = [f(x) for x in xs]

    def inter(x):
        i = np.searchsorted(xs, x)
        # print(i)
        if i == 0:
            if x == xs[0]:
                return fs[0]
            else:
                return 0
        if i == n:
            return 0
        i -= 1
        d = deltas[i]
        return (x - xs[i]) / d * fs[i + 1] + (xs[i + 1] - x) / d * fs[i]

    return inter


def interSpline3(f, xs, a=.0, b=.0, cond=2):
    """
    使用三次样条进行插值，

    :param f: 待插值函数
    :param xs: 分点
    :param a: 左边界条件，仅当边界条件为一、二阶导数时有意义，为 S''(x_0) 或 S'(x_0)
    :param b: 右边界条件（参考 a）
    :param cond: 边界条件的种类，2代表二阶导数条件，1代表一阶导数，0代表周期边界。
    :return: 插值得到的函数对象
    """
    xs = np.sort(xs)
    fs = [f(x) for x in xs]
    if cond == 2:
        return _interSpline3C2(fs, xs, a, b)
    pass


def hermiteBases(l, m, r):
    d1 = m - l
    d2 = r - m

    def alpha(x):
        # if r is None:
        if l is not None and l <= x <= m:
            return (1 - 2 * (x - m) / d1) * (((x - l) / d1) ** 2)
        if r is not None and m <= x <= r:
            return (1 + 2 * (x - m) / d2) * (((x - r) / d2) ** 2)
        return 0.0

    def beta(x):
        if l is not None and l <= x <= m:
            return (x - m) * (((x - l) / d1) ** 2)
        if r is not None and m <= x <= r:
            return (x - m) * (((x - r) / d2) ** 2)
        return 0.0

    return alpha, beta


def _interSpline3C2(fs, xs, a, b):
    """
    带有  S'' 边界条件的 3次样条。使用三弯矩方程求解。

    :param fs: 函数值
    :param xs:
    :param a:
    :param b:
    :return:
    """

    n = len(fs)
    h = np.diff(xs)  # len = n - 1

    # 先得到三弯矩方程系数：
    lambdas = np.zeros(n - 2, h.dtype)
    for i in range(n - 2):
        lambdas[i] = h[i + 1] / (h[i] + h[i + 1])
    mus = 1 - lambdas
    md = mdiff(xs, fs, 2)
    g = md[2] * 6  # len = n-2
    B = np.copy(g)
    B[0] -= mus[0] * a
    B[n - 3] -= lambdas[n - 3] * b
    import numeracy.linear.Vector as Vector
    diag = Vector.of(np.ones_like(mus) * 2)
    upper = Vector.of(lambdas)
    lower = Vector.of(mus[1:])
    B = Vector.of(B)

    # 求解三弯矩方程得到 M
    from numeracy.linear import DirectMethod
    ms = DirectMethod.solveTriDiag(diag, upper, lower, B)  # len = n-2
    M = [a]
    M.extend(ms)
    M.append(b)

    C = np.zeros_like(h)
    D = np.zeros_like(h)
    for j in range(0, n - 1):
        C[j] = fs[j] / h[j] - M[j] * h[j] / 6
        D[j] = fs[j + 1] / h[j] - M[j + 1] * h[j] / 6

    def inter(x):
        i = np.searchsorted(xs, x)
        # print(i)
        if i == 0:
            if x == xs[0]:
                return fs[0]
            else:
                return 0
        if i == n:
            return 0
        i -= 1
        t1 = (M[i] * ((xs[i + 1] - x) ** 3) + M[i + 1] * ((x - xs[i]) ** 3)) / 6 / h[i]
        t2 = C[i] * (xs[i + 1] - x) + D[i] * (x - xs[i])
        return t1 + t2

    return inter

# f = lambda x: x ** 2 + 3 * x + 3
# xs = np.linspace(-1.0, 1.0, 10)
# S = interSpline3(f, xs)
# print(S(0.))
# Lag = interLagrange(f, xs)
# Lin = interLinearPiece(f, xs)
# print(Lag(0))
# # print(xs)
# print(Lin(0))

