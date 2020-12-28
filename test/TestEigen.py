# Created by lyc at 2020/11/9 14:29
import math
from numeracy.linear import Matrix
import numpy as np
import unittest


class TestMatrix(unittest.TestCase):
    def testEigen(self):
        n = 5
        A = Matrix.triDiagonal(2, -1, n)
        res1, _ = Matrix.eigenValuesJacobi(A, eps=1E-12, maxIter=100)
        res2, _ = Matrix.eigenValuesQR(A, maxIter=100)
        res1 = sorted(res1)
        res2 = sorted(res2)
        ev = []
        for k in range(1, n + 1):
            ev.append(2 - 2 * math.cos(k * math.pi / (n + 1)))
        for (r, r1) in zip(ev, res1):
            self.assertAlmostEqual(r, r1, delta=0.1)
        for (r, r1) in zip(ev, res2):
            self.assertAlmostEqual(r, r1, delta=0.1)

