# Created by lyc at 2020/10/25 14:18
import math

import numpy as np

from numeracy.Util import require


def binarySearch(f, low, high, eps=1E-5, maxIter=100):
    """
    使用二分法近似求函数在给定区间上的一个零点。

    :param f:
    :param low:
    :param high:
    :param eps: 精度
    :param maxIter: 最大迭代次数
    :return: 一个元组：(近似值，是否达到所要求的精度)
    """
    require(low < high)
    y1 = f(low)
    y2 = f(high)
    s1 = np.sign(y1)
    s2 = np.sign(y2)
    require(s1 != s2)
    for _ in range(maxIter):
        m = (low + high) / 2
        fm = f(m)
        if np.abs(fm) < eps:
            return m, True
        sm = np.sign(fm)
        if sm == s1:
            low = m
            s1 = sm
        else:
            high = m
    return low, False


def fixPointIter(phi, x0, eps=1E-5, maxIter=100, bound=(None, None)):
    """
    使用不动点迭代法近似计算函数 `phi` 的不动点，迭代方法为 `x_{k+1} = phi(x_k)`。
    当 `|x_{k+1} - x_k| < eps`，或者迭代次数超过上限，或者得到 `x_k` 超出范围时，停止迭代。


    :param phi: 迭代函数
    :param x0: 初值
    :param eps: 精度
    :param maxIter: 最大迭代次数
    :param bound: 允许的范围，(low,high)。
    :return: 一个元组：(近似值，是否达到所要求的精度)
    """
    x = x0
    low, high = bound
    for _ in range(maxIter):
        x1 = phi(x)
        if np.abs(x - x1) < eps:
            return x1, True
        x = x1
        if low is not None and x < low:
            return x, False
        if high is not None and x > high:
            return x, False
    return x, False


def speedUpSteffensen(phi, x0, eps=1E-5, maxIter=100):
    """
    使用 Steffensen 加速方法进行不动点迭代。


    :param phi: 迭代函数
    :param x0: 初值
    :param eps: 精度
    :param maxIter: 最大迭代次数
    :return: 一个元组：(近似值，是否达到所要求的精度)
    """
    x = x0
    for _ in range(maxIter):
        y = phi(x)
        z = phi(y)
        x1 = x - (y - x) ** 2 / (z - 2 * y + x)
        if np.abs(x - x1) < eps:
            return x1, True
        x = x1
    return x, False


def newtonMethod(f, f1, x0, eps, maxIter):
    """
    使用 Newton 迭代法计算 f 的零点。

    :param f: 某个函数
    :param f1: f的导函数
    :param x0: 初值
    :param eps: 精度
    :param maxIter: 最大迭代次数
    :return: 一个元组：(近似值，是否达到所要求的精度)
    """

    x = x0
    for _ in range(maxIter):
        x1 = x - f(x) / f1(x)
        if np.abs(x - x1) < eps:
            return x1, True
        x = x1
    return x, False
