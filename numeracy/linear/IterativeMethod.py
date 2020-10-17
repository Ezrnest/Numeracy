# Created by chenzc18 at 2020/10/16

"""
包含线性方程的迭代解法，包括
 1. Jacobi迭代法
 2. Gauss-Seidel迭代法
 3.

"""
from numeracy.Util import require

from numeracy.linear.Matrix import Matrix as TMatrix
from numeracy.linear.Vector import Vector as TVector


def solveJacobi(A: TMatrix, b: TVector, margin = 0.01) -> TVector:
    require(A.isSquare())
    M = A.getDiag()
    N = M - A
    M1 = M.inverse()
    B = M1 * N
    f = M1 * b
    x = b
    count = 0
    while count < 10000 and (b - A * x).norm() >= margin:
        x = B * x + f
        count += 1
    return x

def solveGaussSeidel(A: TMatrix, b: TVector, margin = 0.01) ->TVector:
    require(A.isSquare())
    M = A.getLower()
    N = M - A
    M1 = M.inverse()
    B = M1 * N
    f = M1 * b
    x = b
    count = 0
    while count < 10000 and (b - A * x).norm() >= margin:
        x = B * x + f
        count += 1
    return x






