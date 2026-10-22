"""矩阵计算器测试套件"""

import sys
import os
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.matrix import Matrix
from src import operations as ops


class TestMatrixCreation(unittest.TestCase):
    def test_basic_creation(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m.data[0][0], Fraction(1))
        self.assertEqual(m.data[1][1], Fraction(4))

    def test_identity(self):
        m = Matrix.identity(3)
        for i in range(3):
            for j in range(3):
                expected = Fraction(1) if i == j else Fraction(0)
                self.assertEqual(m.data[i][j], expected)

    def test_zeros(self):
        m = Matrix.zeros(2, 3)
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 3)
        for i in range(2):
            for j in range(3):
                self.assertEqual(m.data[i][j], Fraction(0))

    def test_unequal_rows_raises(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3]])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            Matrix([])


class TestArithmetic(unittest.TestCase):
    def setUp(self):
        self.a = Matrix([[1, 2], [3, 4]])
        self.b = Matrix([[5, 6], [7, 8]])

    def test_add(self):
        result = ops.matrix_add(self.a, self.b)
        self.assertEqual(result, Matrix([[6, 8], [10, 12]]))

    def test_sub(self):
        result = ops.matrix_sub(self.a, self.b)
        self.assertEqual(result, Matrix([[-4, -4], [-4, -4]]))

    def test_mul(self):
        result = ops.matrix_mul(self.a, self.b)
        self.assertEqual(result, Matrix([[19, 22], [43, 50]]))

    def test_scalar_mul(self):
        result = ops.scalar_mul(2, self.a)
        self.assertEqual(result, Matrix([[2, 4], [6, 8]]))

    def test_transpose(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        result = ops.transpose(m)
        self.assertEqual(result, Matrix([[1, 4], [2, 5], [3, 6]]))

    def test_dimension_mismatch_add(self):
        c = Matrix([[1, 2, 3]])
        with self.assertRaises(ValueError):
            ops.matrix_add(self.a, c)

    def test_dimension_mismatch_mul(self):
        c = Matrix([[1, 2, 3]])
        with self.assertRaises(ValueError):
            ops.matrix_mul(self.a, c)


class TestDeterminant(unittest.TestCase):
    def test_1x1(self):
        self.assertEqual(ops.determinant(Matrix([[5]])), Fraction(5))

    def test_2x2(self):
        self.assertEqual(ops.determinant(Matrix([[1, 2], [3, 4]])), Fraction(-2))

    def test_3x3(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        self.assertEqual(ops.determinant(m), Fraction(27))

    def test_identity_det(self):
        for n in range(1, 5):
            self.assertEqual(ops.determinant(Matrix.identity(n)), Fraction(1))

    def test_singular_det(self):
        m = Matrix([[1, 2], [2, 4]])
        self.assertEqual(ops.determinant(m), Fraction(0))

    def test_non_square_raises(self):
        with self.assertRaises(ValueError):
            ops.determinant(Matrix([[1, 2, 3], [4, 5, 6]]))


class TestInverse(unittest.TestCase):
    def test_2x2_inverse(self):
        m = Matrix([[1, 2], [3, 4]])
        inv = ops.inverse(m)
        product = ops.matrix_mul(m, inv)
        self.assertEqual(product, Matrix.identity(2))

    def test_3x3_inverse(self):
        m = Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        inv = ops.inverse(m)
        product = ops.matrix_mul(m, inv)
        self.assertEqual(product, Matrix.identity(3))

    def test_singular_raises(self):
        m = Matrix([[1, 2], [2, 4]])
        with self.assertRaises(ValueError):
            ops.inverse(m)


class TestRank(unittest.TestCase):
    def test_full_rank(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(ops.rank(m), 2)

    def test_rank_deficient(self):
        m = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        self.assertEqual(ops.rank(m), 1)

    def test_identity_rank(self):
        self.assertEqual(ops.rank(Matrix.identity(4)), 4)


class TestSolve(unittest.TestCase):
    def test_simple_system(self):
        # x + 2y = 5, 3x + 4y = 11 → x=1, y=2
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5], [11]])
        x = ops.solve(a, b)
        self.assertEqual(x.data[0][0], Fraction(1))
        self.assertEqual(x.data[1][0], Fraction(2))

    def test_3x3_system(self):
        a = Matrix([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
        b = Matrix([[6], [-4], [27]])
        x = ops.solve(a, b)
        # 验证 Ax = b
        result = ops.matrix_mul(a, x)
        self.assertEqual(result, b)


class TestLU(unittest.TestCase):
    def test_lu_decomposition(self):
        m = Matrix([[2, 1], [4, 3]])
        L, U = ops.lu_decompose(m)
        product = ops.matrix_mul(L, U)
        self.assertEqual(product, m)

    def test_lu_3x3(self):
        m = Matrix([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
        L, U = ops.lu_decompose(m)
        product = ops.matrix_mul(L, U)
        self.assertEqual(product, m)


class TestRREF(unittest.TestCase):
    def test_identity_rref(self):
        m = Matrix.identity(3)
        self.assertEqual(ops.reduced_row_echelon(m), Matrix.identity(3))

    def test_simple_rref(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        rref = ops.reduced_row_echelon(m)
        # 第一行应为 [1, 0, -1]
        self.assertEqual(rref.data[0][0], Fraction(1))
        self.assertEqual(rref.data[0][1], Fraction(0))


class TestCofactor(unittest.TestCase):
    def test_cofactor_2x2(self):
        m = Matrix([[1, 2], [3, 4]])
        c = ops.cofactor_matrix(m)
        self.assertEqual(c, Matrix([[4, -3], [-2, 1]]))

    def test_adjugate(self):
        m = Matrix([[1, 2], [3, 4]])
        adj = ops.adjugate(m)
        self.assertEqual(adj, Matrix([[4, -2], [-3, 1]]))


if __name__ == '__main__':
    unittest.main()
