# Created by lyc at 2020/10/16 14:54
"""


"""
from typing import Union, Optional

from numpy import ndarray

from numeracy.Util import require
from numeracy.linear.Matrix import Matrix
import numpy as np


class Vector(Matrix):
    def __init__(self, data: ndarray):
        super().__init__(data)
        require(self.row == 1 or self.column == 1)
        if self.column == 1:
            self.length = self.row
            self.isColumn = True
        else:
            self.length = self.column
            self.isColumn = False

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(np.add(self.data, other.data))

        return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(np.subtract(self.data, other.data))

        return super().__sub__(other)

    def __neg__(self):
        return Vector(np.negative(self.data))

    def __getitem__(self, item):
        if isinstance(item, int):
            if self.isColumn:
                return self.data[item, 0]
            else:
                return self.data[0, item]
        return super().__getitem__(item)

    def norm(self, p: Optional[float] = 2.0):
        """
        计算向量的 p-范数，如果 `p is None`，则计算无穷范数。

        :param p: 范数幂次
        """
        a = np.abs(self.data)
        if p is None:
            return np.max(a)
        else:
            return np.sum(a ** p) ** (1.0 / p)

    def innerProduct(self, other):
        return np.sum(self.data * other.data)

    def unitize(self):
        """
        返回归一化后的向量 v / |v|

        """
        return Vector(self.data / self.norm())

    def toColumn(self):
        if self.isColumn:
            return self
        return self.transpose()


def of(arr, isColumn=True, dtype=None) -> Vector:
    array = np.asarray(arr, dtype)
    require(array.ndim == 1)
    length = array.shape[0]
    if isColumn:
        array = array.reshape((length, 1))
    else:
        array = array.reshape((1, length))
    return Vector(array)


def constant(c, length, isColumn=True) -> Vector:
    data = np.array([c] * length)

    return of(data, isColumn)


def zero(length, dtype=float, isColumn=True) -> Vector:
    return of(np.zeros(length, dtype), isColumn)
