# Created by lyc at 2020/10/16 14:02

"""
包含矩阵类的工具等。

"""
import math
import typing
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

    @property
    def shape(self):
        return self.row, self.column

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

    def __truediv__(self, other):
        if np.isscalar(other):
            return fromArray(self.data / other)

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
        # return self.data[idx]
        r = self.data[idx]
        if np.isscalar(r):
            return r
        if r.ndim == 2:
            return fromArray(r)
        from numeracy.linear import Vector
        isColumn = not isinstance(idx, int) and isinstance(idx[0], slice)
        return Vector.of(r, isColumn)

    # def subMatrix(self, r0, r1, c0, c1):
    #     return fromArray(self.data[r0:r1, c0:c1])

    def __setitem__(self, key, value):
        self.data[key] = value

    def isSquare(self):
        return self.row == self.column

    def det(self):
        require(self.isSquare())
        return np.linalg.det(self.data)

    def getRow(self, r):
        import numeracy.linear.Vector as Vector
        return Vector.of(self.data[r, :], isColumn=False)

    def getCol(self, c):
        import numeracy.linear.Vector as Vector
        return Vector.of(self.data[:, c])

    def getDiag(self):
        require(self.isSquare())
        A = copy(self)
        n = A.row
        for i in range(n):
            for j in range(n):
                if j != i:
                    A[i, j] = 0

        return A

    def isDiag(self):
        require(self.isSquare())
        n = self.row
        for i in range(n):
            for j in range(n):
                if (i != j) and (self[i, j] != 0):
                    return False

        return True

    def getUpper(self):
        require(self.isSquare())
        A = copy(self)
        n = A.row
        for i in range(n):
            for j in range(i):
                A[i, j] = 0

        return A

    def getLower(self):
        require(self.isSquare())
        A = copy(self)
        n = A.row
        for i in range(n):
            for j in range(i + 1, n):
                A[i, j] = 0

        return A

    def inverse(self):
        require(self.isSquare())
        A = copy(self)
        n = A.row
        I = identity(n)
        for j in range(n):
            maxRow = j
            max = abs(A[j, j])
            for i in range(j + 1, n):
                v = abs(A[i, j])
                if v > max:
                    maxRow = i
                    max = v
            if maxRow != j:
                A.data[[j, maxRow]] = A.data[[maxRow, j]]
                I.data[[j, maxRow]] = I.data[[maxRow, j]]
            c = 1 / A[j, j]
            A.data[j] *= c
            I.data[j] *= c
            for i in range(n):
                if i == j:
                    continue
                p = -A.data[i][j]
                A.data[i] += p * A.data[j]
                I.data[i] += p * I.data[j]

        return I

    def rowVectors(self):
        return [self.getRow(r) for r in range(self.row)]

    def columnVectors(self):
        return [self.getCol(c) for c in range(self.column)]

    def transpose(self):
        return fromArray(self.data.transpose())

    @property
    def T(self):
        return self.transpose()

    @property
    def H(self):
        return Matrix(self.data.conj().T)

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

    def decompQR(self):
        """
        返回此矩阵的 QR 分解。

        :return:
        """
        require(self.isSquare())
        vs = self.columnVectors()
        n = self.row
        R = zero(n, n, self.data.dtype)
        qs = []
        for i in range(n):
            v_i = vs[i]
            R[i, i] = v_i.norm()
            q_i = v_i / R[i, i]
            for j in range(i + 1, n):
                v_j = vs[j]
                R[i, j] = q_i.innerProduct(v_j)
                v_j = v_j - R[i, j] * q_i
                vs[j] = v_j
            qs.append(q_i)
        return fromColVectors(qs), R

    def __str__(self) -> str:
        return str(self.data)

    def sqrt(self):
        """
        返回半正定矩阵 A 的“平方根”，即半正定矩阵 B 使得 B^2 = A

        :return:
        """
        require(self.isSquare())
        u, s, vh = np.linalg.svd(self.data, compute_uv=True, hermitian=True)
        s = s ** 0.5
        return Matrix((u * s) @ vh)

    def abs(self):
        """
        返回矩阵 A 的“绝对值”，即 A^T A 的平方根。要求这个矩阵是方阵。

        :return:
        """
        require(self.isSquare())
        u, s, vh = np.linalg.svd(self.data, compute_uv=True, hermitian=False)
        v = vh.conj().T
        return Matrix((v * s) @ vh)

    def norm(self, p=2):
        return np.linalg.norm(self.data, p)


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


def fromColVectors(vs):
    arrs = []
    for v in vs:
        require(v.isColumn)
        arrs.append(v.data)
    data = np.concatenate(arrs, axis=1)
    return Matrix(data)


def diagonal(d: ndarray):
    n = len(d)
    A = zero(n, n, float)
    for i in range(n):
        A[i, i] = d[i]
    return A


def triDiagonal(diag, sub, n, dtype=float):
    """
    返回阶数为 `n` 的对称三对角矩阵。

    :param diag: 主对角元
    :param sub: 次对角线元
    :param n: 阶数
    :return:
    """

    A = zero(n, n, dtype)
    for i in range(n):
        A[i, i] = diag
        if i > 0:
            A[i, i - 1] = A[i - 1, i] = sub
    return A


def copy(A: Matrix) -> Matrix:
    d = np.copy(A.data)
    return Matrix(d)


def givensTrans(A: Matrix, k, l):
    """
    使用 Givens 变换将实对称矩阵`A` 的元素 `A[k,l]` 变为 0，此函数会改变这个矩阵。

    :param A:
    :param k:
    :param l:
    :return:
    """
    a_kk = A[k, k]
    a_ll = A[l, l]
    a_kl = A[k, l]
    t1 = (a_kk - a_ll) / 2
    t2 = a_kl
    θ = np.arctan2(t2, t1) / 2
    c = np.cos(θ)
    s = np.sin(θ)
    c2 = c ** 2
    s2 = s ** 2
    x = a_kl * np.sin(2 * θ)
    A[k, k] = a_kk * c2 + a_ll * s2 + x
    A[l, l] = a_kk * s2 + a_ll * c2 - x
    A[l, k] = A[k, l] = 0

    for i in range(A.row):
        if i == k or i == l:
            continue
        a_ik = A[i, k]
        a_il = A[i, l]
        A[i, k] = A[k, i] = a_ik * c + a_il * s
        A[i, l] = A[l, i] = -a_ik * s + a_il * c

    pass


def eigenValuesJacobi(A: Matrix, eps=1E-3, maxIter=10):
    """
    使用 Jacobi 方法求解实对称矩阵 `A` 的所有特征值。

    :param A: 实对称矩阵
    :param eps: 近似精度
    :param maxIter: 最大迭代次数
    :return: (A 的特征值列表, 其它数据)，其他数据=(迭代次数，每次迭代的变换次数列表，)
    """
    require(A.isSquare())

    A = copy(A)
    n = A.row

    def sumOfSquareOfNonDiag(A: Matrix):
        s = 0
        for i in range(A.row):
            for j in range(A.column):
                if i == j:
                    continue
                x = A[i, j]
                s += x * np.conj(x)
        return s

    N = sumOfSquareOfNonDiag(A)
    δ = (N ** 0.5) / n
    m = 0
    transCounts = []
    # ns = [N]
    while δ > eps and m < maxIter:
        m += 1
        s = 0
        # mi, mj = 0, 0
        # for i in range(n):
        #     for j in range(0, i):
        #         a = A[i, j]
        #         if np.abs(a) > maximal:
        #             maximal = np.abs(a)
        #             mi, mj = i, j
        # # print(maximal,mi,mj)
        # if maximal < eps:
        #     break
        # givensTrans(A, mi, mj)
        transCount = 0
        for i in range(n):
            for j in range(0, i):
                a = np.abs(A[i, j])
                s += a ** 2
                # maximal = max(np.abs(a),maximal)
                if a > δ:
                    transCount += 1
                    givensTrans(A, i, j)
        transCounts.append(transCount)
        # ns.append()
        δ /= n

    return [A[i, i] for i in range(n)], (m, transCounts,)


def eigenValuesQR(A: Matrix, eps=1E-3, maxIter=50):
    """
    使用 QR 方法计算实矩阵 `A` 的全体特征值，要求 `A` 的特征值全为单根。

    :param A: 实对称矩阵
    :param eps: 近似精度
    :param maxIter: 最大迭代次数
    :return: (A 的特征值列表, 其它数据)，其他数据=(迭代次数，每次迭代的变换次数列表，)
    """
    require(A.isSquare())
    from numeracy.linear import IterativeMethod
    H1, vs = IterativeMethod.orthSimHessExtend(A, A.row, A.getCol(0))
    H = H1[:-1, :]
    A = H
    diffs = []
    k = 0
    while k < maxIter:
        k += 1
        Q, R = A.decompQR()
        A_1 = R * Q

        def diagDiff():
            d = 0.0
            for i in range(A.row):
                d = np.max(d, np.abs(A[i, i] - A_1[i, i]))
            return d

        d = diagDiff()
        diffs.append(d)
        small = (d < eps)
        A = A_1
        if small:
            break
    return [A[i, i] for i in range(A.row)], (k, diffs)
