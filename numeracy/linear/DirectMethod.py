# Created by lyc at 2020/10/16 13:56

"""
包含线性方程的直接解法，包括
 1. （选主元）Gauss 消去法
 2. 利用 Cholesky 分解求解
 3. 正则化方法

"""
from numeracy.Util import require
from numeracy.linear import Vector, Matrix

from numeracy.linear.Matrix import Matrix as TMatrix
from numeracy.linear.Vector import Vector as TVector
import numpy as np

def gaussElimination(A : TMatrix, b : TVector, pivot) -> TVector:
    # TODO
    pass


def solveUpper(U: TMatrix, b: TVector) -> TVector:
    require(U.isSquare())

    n = U.row
    x = np.zeros(n, U.data.dtype)
    for i in range(n - 1, -1, -1):
        t = 0
        for k in range(i + 1, n):
            t += U[i, k] * x[k]
        x[i] = (b[i] - t) / U[i, i]
    return Vector.of(x)


def solveLower(L: TVector, b: TVector) -> TVector:
    require(L.isSquare())
    n = L.row
    x = np.zeros(n, L.data.dtype)
    for i in range(n):
        t = 0
        for k in range(i):
            t += L[i, k] * x[k]
        x[i] = (b[i] - t) / L[i, i]
    return Vector.of(x)


def solveCholesky(A: TMatrix, b: TVector) -> TVector:
    """
    使用 Cholesky 分解的方法求解线性方程组 Ax=b，即分解 A = LL^H 并求解 Ly=b 以及 L^H x = b。

    :param A: 正定矩阵
    :param b: 列向量
    """
    L = A.decompCholesky()
    y = solveLower(L, b)
    x = solveUpper(L.T, y)
    return x
