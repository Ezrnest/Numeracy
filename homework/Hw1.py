# Created by lyc at 2020/10/16 16:34
from numeracy.linear import Matrix, Vector, DirectMethod

n = 10
H = Matrix.hilbert(n)
x = Vector.constant(1.0, n)
b = H * x
print(b)


def solveWithCholesky():
    x1 = DirectMethod.solveCholesky(H, b)
    dx = x1 - x
    print(dx.norm())


solveWithCholesky()
