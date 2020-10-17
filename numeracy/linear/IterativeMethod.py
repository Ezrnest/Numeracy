# Created by chenzc18 at 2020/10/16

"""
包含线性方程的迭代解法，包括
 1. Jacobi迭代法
 2. Gauss-Seidel迭代法
 3. SOR 迭代法
 4. 共轭梯度法
 5. 广义最小化残差方法(GMRES)

"""
from typing import Tuple, List

from numeracy.Util import require

from numeracy.linear.Matrix import Matrix as TMatrix
from numeracy.linear.Vector import Vector as TVector
import numeracy.linear.Matrix as Matrix
import numeracy.linear.Vector as Vector

import numpy as np


def solveJacobi(A: TMatrix, b: TVector, margin=0.001, maxIter = 10000) -> TVector:
    """


    :param A: 可逆矩阵
    :param b: 右端向量
    :param margin: 容许的误差，当残差的模长小于该值时停止迭代，默认=0.001
    :param maxIter: 算法收敛之前允许的最大迭代次数，默认=10000
    """
    require(A.isSquare())
    M = A.getDiag()
    N = M - A
    M1 = M.inverse()
    B = M1 * N
    f = M1 * b
    x = Vector.constant(0, A.row)
    count = 0
    while count < maxIter and (b - A * x).norm() >= margin:
        x = B * x + f
        count += 1
    return x


def solveGaussSeidel(A: TMatrix, b: TVector, margin=0.001, maxIter = 10000) -> TVector:
    """


    :param A: 可逆矩阵
    :param b: 右端向量
    :param margin: 容许的误差，当残差的模长小于该值时停止迭代，默认=0.001
    :param maxIter: 算法收敛之前允许的最大迭代次数，默认=10000
    """
    require(A.isSquare())
    M = A.getLower()
    print(M)
    N = M - A
    M1 = M.inverse()
    print(M1)
    B = M1 * N
    f = M1 * b
    x = Vector.constant(0, A.row)
    count = 0
    while count < maxIter and (b - A * x).norm() >= margin:
        x = B * x + f
        count += 1
    return x

def solveSor(A: TMatrix, b: TVector, w = 1.0, margin=0.001, maxIter = 10000) -> TVector:
    """


    :param A: 可逆矩阵
    :param b: 右端向量
    :param w: 松弛因子，默认=1
    :param margin: 容许的误差，当残差的模长小于该值时停止迭代，默认=0.001
    :param maxIter: 算法收敛之前允许的最大迭代次数，默认=10000
    """
    require(A.isSquare())
    D = A.getDiag()
    L = A.getLower() - D
    U = A.getUpper() - D
    M = D - w * L
    N = (1 - w) * D + w * U
    M1 = M.inverse()
    B = M1 * N
    f = M1 * b
    x = Vector.constant(0, A.row)
    count = 0
    while count < maxIter and (b - A * x).norm() >= margin:
        x = B * x + f
        count += 1
    return x


def conjGrad(A: TMatrix, b: TVector, margin=0.001, maxIter = 10000) -> TVector:
    """


    :param A: 对称正定矩阵
    :param b: 右端向量
    :param margin: 容许的误差，当残差的模长小于该值时停止迭代，默认=0.001
    :param maxIter: 算法收敛之前允许的最大迭代次数，默认=10000
    """
    require(A.isSquare())
    x = Vector.constant(0, A.row)
    r = b
    p = b
    count = 0
    while count < maxIter and (b - A * x).norm() >= margin:
        Ap = A * p
        pAp = Ap.innerProduct(p)
        alpha = r.innerProduct(p) / pAp
        x += alpha * p
        r = b - A * x
        beta = r.innerProduct(Ap) / pAp
        p = r + beta * p
        count += 1

    return x


def orthSimHessExtend(A: TMatrix, m: int, v0: TVector) -> Tuple[TMatrix, List[TVector]]:
    """
    计算与 A 正交相似的 (m+1,m) 阶增广 Hessenberg 矩阵 H1 = (h_{ij}),（如果j < i-1 则h_{ij} = 0） ，
    以及对应的正交单位向量组 (v_0, ... v_m) = V
    使得 V^T A V = H1，

    :param A: 矩阵
    :param m: Hessenberg 矩阵的阶数
    :param v0: 初始向量，要求非零
    :return 矩阵 H1 以及以列表形式给出的 v_0, ..., v_m 向量
    """
    n = A.row
    v0 = v0.unitize()
    V = [v0]
    # AV = [A * v0]
    H = np.zeros((m + 1, n), A.data.dtype)
    for i in range(m):
        # v_i is known
        av = A * V[i]
        # Av_i = h_{0,i}v_0 + h_{1,i}v_1 + ... + h_{i,i} v_i + h_{i+1,i} v_{i+1}
        # (A v_i, v_k) =  h_{k,i}, k <= i
        # h_{i+1,i} v_{i+1} = Av_i - (h_{0,i}v_0 + h_{1,i}v_1 + ... + h_{i,i} v_i)
        r = av.data
        for k in range(i + 1):
            vk = V[k]
            h = av.innerProduct(vk)
            H[k, i] = h
            r -= h * vk.data
        r = TVector(r)
        length = r.norm()
        H[i + 1, i] = length
        v = r / length
        V.append(v)
    return TMatrix(H), V


# def gmresStep(A: TMatrix, r: TVector,
#               m: int = 5) -> TVector:
#     return z


def gmres(A: TMatrix, b: TVector, m=5, eps=1E-5, maxIter=20) -> Tuple[TVector, bool, List[float]]:
    """
    使用迭代的 GMRES 方法求解 Ax=b

    :param A: 矩阵
    :param b: 向量
    :param m: 每次迭代的子空间大小
    :param eps: 当误差小于此数时认为收敛
    :param maxIter: 最大迭代次数
    :return: 元组，包含（最后得到的x,是否收敛，每次的误差（列表））
    """
    require(eps >= 0)
    epss = []
    x = Vector.zero(A.column)
    i = 0
    convergent = False
    while True:
        r = b - A * x
        beta = r.norm()
        epss.append(beta)
        if beta < eps:
            convergent = True
            break
        if i >= maxIter:
            break
        i += 1
        H, vs = orthSimHessExtend(A, m, r)
        H1 = H[:, 0:m]
        e1 = np.zeros(m + 1, A.data.dtype)
        e1[0] = beta
        ym = np.linalg.lstsq(H1.data, e1, rcond=None)[0]
        # TODO 针对 Hessenberg 矩阵实现更快的最小二乘法
        ym = Vector.of(ym)
        V = Matrix.fromColVectors(vs[:-1])
        z = V * ym
        x = x + z

    # r = b - A*x
    return x, convergent, epss
