# Created by lyc at 2020/10/16 14:02

"""
包含矩阵类的工具等。

"""
import typing
from numpy.matrixlib import matrix
import numpy as np
from numpy import ndarray


class Matrix:
    """
    基于 ndarray 的矩阵类

    """

    def __init__(self, data: ndarray):
        if data.ndim != 2:
            raise Exception("Dim of the array != 2")
        self.data = data
        (self.row, self.column) = data.shape

    def __add__(self, other):
        if isinstance(other, Matrix):
            return Matrix(np.add(self.data, other.data))

        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Matrix):
            return Matrix(np.subtract(self.data, other.data))

        return NotImplemented

    def __neg__(self):
        return Matrix(np.negative(self.data))

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(np.matmul(self.data, other.data))

        if np.isscalar(other):
            return Matrix(self.data * other)

        return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other):
            return Matrix(self.data * other)
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        return np.equal(self.data, other.data)


def of(row, column, f, dtype=float):
    """
    按照函数 `f` 生成形状为 (row * column) 的矩阵 A = (a_ij), a_ij = f(i,j)

    :param row: 行数
    :param column: 列数
    :param f: 指定矩阵元素的函数，接受行列下标作为参数
    :param dtype: 元素数据类型
    """
    data = ndarray((row, column), dtype)
    for i in range(row):
        for j in range(column):
            data[i, j] = f(i, j)
    return Matrix(data)


def identity(n, dtype=float):
    """
    返回 n 阶的单位阵。

    :param n: 阶数
    :param dtype: 矩阵的数据类型
    """
    return Matrix(np.eye(n, dtype=dtype), )


def zero(row, column, dtype):
    """
    返回 (row * column) 的零矩阵。

    :param row: 行数
    :param column: 列数
    :param dtype: 元素数据类型
    """
    return Matrix(np.zeros((row, column), dtype))


def hilbert(n: int, dtype=float) -> Matrix:
    """
    返回 n 阶的 Hilbert矩阵。

    :param n: 阶数
    :param dtype: 元素数据类型
    """
    data = np.ndarray((n, n), dtype)
    for i in range(n):
        for j in range(n):
            data[i, j] = 1.0 / (i + j + 1)
    return Matrix(data)


def decompLU(m: matrix) -> typing.Tuple[matrix, matrix]:
    pass


def decompCholesky(m: matrix) -> typing.Tuple[matrix, matrix]:
    pass
