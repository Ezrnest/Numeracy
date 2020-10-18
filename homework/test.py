import numpy as np
from numeracy.linear import Matrix, Vector, IterativeMethod, DirectMethod

n = 15
H = Matrix.hilbert(n)
x0 = Vector.constant(1.0, n)
b = H * x0

#x = DirectMethod.solveGauss(H, b)
#x = IterativeMethod.solveJacobi(H, b)
#x = DirectMethod.solveCholesky(H, b)
x = DirectMethod.solveRegularization(H, b)
#x = IterativeMethod.conjGrad(H, b, maxIter=100)
#x, m, n = IterativeMethod.gmres(H, b)
print(x)
print((x-x0).norm())