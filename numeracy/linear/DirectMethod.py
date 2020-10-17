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
from math import sqrt


def solveGauss0(M: np.ndarray):
    require(M.ndim == 2)
    n, m = M.shape
    for k in range(n):
        maxIdx = k
        maxVal = np.abs(M[k, k])
        for i in range(k + 1, n):
            v = np.abs(M[i, k])
            if v > maxVal:
                maxIdx = i
                maxVal = v
        if maxIdx != k:
            M[[k, maxIdx]] = M[[maxIdx, k]]
        c = 1.0 / M[k, k]
        M[k, k + 1:] *= c
        for i in range(n):
            if i == k:
                continue
            p = -M[i, k]
            M[i, k + 1:] += p * M[k, k + 1:]
    return M[:, n:]


def solveGauss(A: TMatrix, B: TMatrix) -> TMatrix:
    M = np.concatenate((A.data, B.data), axis=1)
    # print(M)
    # print(M.shape)
    r = solveGauss0(M)
    return Matrix.fromArray(r)


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

def solveRegularization(A: TMatrix, b: TVector) -> TVector:
    require(A.isSquare())
    M = (A.T * A).data
    mu = []
    u = []
    v = []
    potent, vector = np.linalg.eigh(M)
    n = A.row
    for i in range(n):
        mu.append(sqrt(potent[i]))
        u.append(Vector.of(vector[i]))
        v.append(A * u[i] / mu[i])

    x = Vector.zero(n)
    for i in range(n):
        x += b.innerProduct(v[i]) * u[i] / mu[i]

    return x

