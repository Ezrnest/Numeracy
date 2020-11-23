# Created by lyc at 2020/11/9 14:29
import math

from numeracy.linear import Matrix
import numpy as np
n = 10
A = Matrix.triDiagonal(2, -1, n)
# print(A)
# res = Matrix.eigenValuesJacobi(A,eps=1E-12,maxIter=100)
# print(sorted(res))
# #
res,_ = Matrix.eigenValuesQR(A,maxIter=100)
print(res)
# print(sorted(res))
# print(sorted(np.linalg.eigvalsh(A.data)))
# Q,R = A.decompQR()
# Q1,R1 = np.linalg.qr(A.data)
# print(R)
# print(R1)

ev = []
for k in range(1,n+1):
    ev.append(2 - 2 * math.cos(k * math.pi / (n+1)))
print(ev)
