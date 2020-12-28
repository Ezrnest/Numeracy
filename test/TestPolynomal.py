# Created by lyc at 2020/12/28 15:13
import unittest
import numpy as np

from numeracy.equation.Polynomial import pseudoDivide


class TestPolynomial(unittest.TestCase):

    def testDivide(self):
        a = np.poly1d([1,-3])
        p = np.poly1d([1, -5, 6])
        # self.ass
        self.assertEqual(pseudoDivide(p, 2),a)
