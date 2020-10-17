# Created by lyc at 2020/10/16 14:02

"""
包含矩阵类的工具等。

"""
import typing
from numpy.matrixlib import matrix
import numpy as np
from numpy import ndarray
from numeracy.Util import require


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
            return fromArray(np.matmul(self.data, other.data))

        if np.isscalar(other):
            return fromArray(self.data * other)

        return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other):
            return fromArray(self.data * other)
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        return np.equal(self.data, other.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, key, value):
        self.data[key] = value

    def isSquare(self):
        return self.row == self.column

    def det(self):
        require(self.isSquare())
        return np.linalg.det(self.data)

    def getRow(self, r):
        from numeracy.linear.Vector import Vector
        return Vector(self.data[r, :])

    def getCol(self, c):
        from numeracy.linear.Vector import Vector
        return Vector(self.data[:, c])

    def getDiag(self):
        require(self.isSquare())
        A = copy(self)
        n = A.row
        for i in range(n):
            for j in range(n):
                if j != i:
                    A[i][j] = 0

        return A


    def isDiag(self):
        require(self.isSquare())
        n = self.row
        for i in range(n):
            for j in range(n):
                if (i != j) and (self[i][j] != 0):
                    return False

        return True

    def getUpper(self):
        require(self.isSquare())
        A = copy(self)
        n = A.row
        for i in range(n):
            for j in range(i):
                A[i][j] = 0

        return A

    def getLower(self):
        require(self.isSquare())
        A = copy(self)
        n = A.row
        for i in range(n):
            for j in range(i + 1, n):
                A[i][j] = 0

        return A

    def inverse(self):
        require(self.isSquare())
        A = copy(self)
        n = A.row
        I = identity(n)
        for j in range(n):
            maxRow = j
            max = abs(A[j][j])
            for i in range(j + 1, n):
                v = abs(A[i][j])
                if v > max:
                    maxRow = i
                    max = v
            if maxRow != j:
                A.data[[j, maxRow]] = A.data[[maxRow, j]]
                I.data[[j, maxRow]] = I.data[[maxRow, j]]
            c = 1 / A[j][j]
            A.data[j, j:] *= c
            I.data[j, j:] *= c
            for i in range(n) :
                if i == j:
                    continue
                p = -A.data[i][j]
                A.data[i] += p * A.data[j]
                I.data[i] += p * A.data[j]

        return I


    def rowVectors(self):
        return list(self.data)

    def columnVectors(self):
        return [self.getCol(c) for c in range(self.column)]

    def transpose(self):
        return fromArray(self.data.transpose())

    @property
    def T(self):
        return self.transpose()

    def decompCholesky(self):
        """
        计算矩阵 A 的 Cholesky 分解：A = LL^H， 其中 L 为下三角矩阵，L^H 为 L 的共轭转置。

        要求这个矩阵是半正定的。

        :return: 下三角矩阵 L
        """

        A = self.data
        n = self.row
        L = np.zeros((n, n), A.dtype)
        for j in range(n):
            t = A[j, j]
            for k in range(j):
                t = t - L[j, k] * np.conj(L[j, k])
            t = np.sqrt(t)
            L[j, j] = t
            # l_{jj} = sqrt(a_{jj} - sum(0,j-1, l_{jk}^2))
            for i in range(j + 1, n):
                a = A[i, j]
                for k in range(j):
                    a -= L[i, k] * np.conj(L[j, k])
                a /= t
                L[i, j] = a
                #  l_{ij} = (a_{ij} - sum(0,j-1,l_{il}l_{jl}))/l_{jj}
        return Matrix(L)

    def __str__(self) -> str:
        return str(self.data)


def fromF(row, column, f, dtype=float):
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


def fromArray(data: ndarray):
    row, column = data.shape
    if row == 1 or column == 1:
        from numeracy.linear.Vector import Vector
        return Vector(data)
    return Matrix(data)

def diagonal(d: ndarray):
    n = len(d)
    A = zero(n, n, float)
    for i in range(n):
        A[i][i] = d[i]
    return A

def copy(A: Matrix) -> Matrix:
    d = np.copy(A.data)
    return Matrix(d)
