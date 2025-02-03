import numpy
import pickle
from nutils import matrix, sparse, testing, warnings


class construction(testing.TestCase):

    def setUp(self):
        super().setUp()
        class TestBackend:
            def assemble(values, rowptr, colidx, ncols):
                self.assertIsInstance(values, numpy.ndarray)
                self.assertEqual(values.ndim, 1)
                self.assertIsInstance(rowptr, numpy.ndarray)
                self.assertEqual(rowptr.ndim, 1)
                self.assertIn(rowptr.dtype.kind, 'iu')
                self.assertIsInstance(colidx, numpy.ndarray)
                self.assertEqual(colidx.ndim, 1)
                self.assertIn(colidx.dtype.kind, 'iu')
                self.assertEqual(len(colidx), len(values))
                self.assertIsInstance(ncols, int)
                return values, rowptr, colidx, ncols
        self.enter_context(matrix.backend(TestBackend))

    def test_assemble_csr(self):
        orig_values = [10., 20., 30.]
        orig_rowptr = [0, 2, 3]
        orig_colidx = [0, 2, 1]
        orig_ncols = 3
        values, rowptr, colidx, ncols = matrix.assemble_csr(orig_values, orig_rowptr, orig_colidx, orig_ncols)
        self.assertEqual(values.tolist(), orig_values)
        self.assertEqual(rowptr.tolist(), orig_rowptr)
        self.assertEqual(colidx.tolist(), orig_colidx)
        self.assertEqual(ncols, orig_ncols)

    def test_assemble_coo(self):
        orig_values = [10., 20., 30.]
        orig_colidx = [0, 2, 1]
        orig_ncols = 3
        values, rowptr, colidx, ncols = matrix.assemble_coo(orig_values, [0, 0, 1], 2, orig_colidx, orig_ncols)
        self.assertEqual(values.tolist(), orig_values)
        self.assertEqual(rowptr.tolist(), [0, 2, 3])
        self.assertEqual(colidx.tolist(), orig_colidx)
        self.assertEqual(ncols, orig_ncols)

    def test_fromsparse(self):
        data = sparse.prune(sparse.fromarray(numpy.eye(2, 3, 1)), inplace=True)
        values, rowptr, colidx, ncols = matrix.fromsparse(data)
        self.assertEqual(values.tolist(), [1., 1.])
        self.assertEqual(rowptr.tolist(), [0, 1, 2])
        self.assertEqual(colidx.tolist(), [1, 2])
        self.assertEqual(ncols, 3)

    def test_empty(self):
        values, rowptr, colidx, ncols = matrix.empty((3, 2))
        self.assertEqual(values.tolist(), [])
        self.assertEqual(rowptr.tolist(), [0, 0, 0, 0])
        self.assertEqual(colidx.tolist(), [])
        self.assertEqual(ncols, 2)

    def test_diag(self):
        values, rowptr, colidx, ncols = matrix.diag([10., 20., 30.])
        self.assertEqual(values.tolist(), [10., 20., 30.])
        self.assertEqual(rowptr.tolist(), [0, 1, 2, 3])
        self.assertEqual(colidx.tolist(), [0, 1, 2])
        self.assertEqual(ncols, 3)

    def test_eye(self):
        values, rowptr, colidx, ncols = matrix.eye(3)
        self.assertEqual(values.tolist(), [1., 1., 1.])
        self.assertEqual(rowptr.tolist(), [0, 1, 2, 3])
        self.assertEqual(colidx.tolist(), [0, 1, 2])
        self.assertEqual(ncols, 3)

    def test_assemble_block_csr(self):
        A00 = [10., 20., 30.], [0, 2, 3], [0, 1, 0], 2 # 2x2
        A01 = [], [0, 0, 0], [], 2 # 2x2
        A10 = [40., 60.], [0, 1, 2], [0, 0], 1 # 2x1
        A11 = [50.], [0, 1, 1], [2], 3 # 2x3
        values, rowptr, colidx, ncols = matrix.assemble_block_csr([[A00, A01], [A10, A11]])
        self.assertEqual(values.tolist(), [10., 20., 30., 40., 50., 60.])
        self.assertEqual(rowptr.tolist(), [0, 2, 3, 5, 6])
        self.assertEqual(colidx.tolist(), [0, 1, 0, 0, 3, 0])
        self.assertEqual(ncols, 4)


@testing.parametrize
class backend(testing.TestCase):

    n = 100
    complex = False

    def setUp(self):
        super().setUp()
        if self.backend == 'scipy':
            self.require_module('scipy')
        elif self.backend == 'mkl':
            self.require_library('mkl_rt')
        self.enter_context(matrix.backend(self.backend))
        self.offdiag = -1+.5j if self.complex else -1
        self.exact = 2 * numpy.eye(self.n) + self.offdiag * numpy.eye(self.n, self.n, 1) + self.offdiag * numpy.eye(self.n, self.n, -1)
        data = sparse.prune(sparse.fromarray(self.exact), inplace=True)
        assert len(data) == self.n*3-2
        self.matrix = matrix.fromsparse(data, inplace=True)

    def test_size(self):
        self.assertEqual(self.matrix.size, self.n**2)

    def test_export_dense(self):
        array = self.matrix.export('dense')
        self.assertEqual(array.shape, (self.n, self.n))
        numpy.testing.assert_equal(actual=array, desired=self.exact)

    def test_export_coo(self):
        data, (row, col) = self.matrix.export('coo')
        numpy.testing.assert_equal(row[0::3], numpy.arange(self.n))
        numpy.testing.assert_equal(col[0::3], numpy.arange(self.n))
        numpy.testing.assert_equal(data[0::3], 2)
        numpy.testing.assert_equal(row[1::3], numpy.arange(self.n-1))
        numpy.testing.assert_equal(col[1::3], numpy.arange(1, self.n))
        numpy.testing.assert_equal(data[1::3], self.offdiag)
        numpy.testing.assert_equal(row[2::3], numpy.arange(1, self.n))
        numpy.testing.assert_equal(col[2::3], numpy.arange(self.n-1))
        numpy.testing.assert_equal(data[2::3], self.offdiag)

    def test_export_csr(self):
        data, indices, indptr = self.matrix.export('csr')
        self.assertEqual(indptr[0], 0)
        self.assertEqual(indptr[-1], len(data))
        numpy.testing.assert_equal(data[0::3], 2)
        numpy.testing.assert_equal(data[1::3], self.offdiag)
        numpy.testing.assert_equal(data[2::3], self.offdiag)
        numpy.testing.assert_equal(indices[0::3], numpy.arange(self.n))
        numpy.testing.assert_equal(indices[1::3], numpy.arange(1, self.n))
        numpy.testing.assert_equal(indices[2::3], numpy.arange(self.n-1))
        numpy.testing.assert_equal(indptr[1:-1], numpy.arange(2, 3*(self.n-1), 3))

    def test_neg(self):
        neg = -self.matrix
        numpy.testing.assert_equal(actual=neg.export('dense'), desired=-self.exact)

    def test_mul(self):
        mul = self.matrix * 1.5
        numpy.testing.assert_equal(actual=mul.export('dense'), desired=self.exact * 1.5)
        with self.assertRaises(TypeError):
            self.matrix * 'foo'

    def test_matvec(self):
        x = numpy.arange(self.n)
        b = numpy.zeros(self.n)
        b[0] = -1
        b[-1] = self.n
        if self.complex:
            b = b + x * 1j
            b[0] += .5j
            b[-1] -= .5j * self.n
        numpy.testing.assert_equal(actual=self.matrix @ x, desired=b)

    def test_matmat(self):
        X = numpy.arange(self.n*2).reshape(-1, 2)
        B = numpy.zeros((self.n, 2))
        B[0] = -2, -1
        B[-1] = 2*self.n, 2*(self.n+.5)
        if self.complex:
            B = B + numpy.arange(self.n*2).reshape(-1, 2) * 1j
            B[0] += 1j, .5j
            B[-1] -= 1j * self.n, 1j * (self.n+.5)
        numpy.testing.assert_equal(actual=self.matrix @ X, desired=B)
        with self.assertRaises(TypeError):
            self.matrix @ 'foo'
        with self.assertRaises(matrix.MatrixError):
            self.matrix @ numpy.arange(self.n+1)

    def test_rmul(self):
        rmul = 1.5 * self.matrix
        numpy.testing.assert_equal(actual=rmul.export('dense'), desired=self.exact * 1.5)
        with self.assertRaises(TypeError):
            'foo' / self.matrix

    def test_div(self):
        div = self.matrix / 1.5
        numpy.testing.assert_equal(actual=div.export('dense'), desired=self.exact / 1.5)
        with self.assertRaises(TypeError):
            self.matrix / 'foo'

    def test_add(self):
        j = self.n//2
        v = 10.
        other = matrix.assemble_coo(numpy.full(self.n, v), numpy.arange(self.n), self.n, numpy.full(self.n, j), self.n)
        add = self.matrix + other
        numpy.testing.assert_equal(actual=add.export('dense'), desired=self.exact + numpy.eye(self.n)[j]*v)
        with self.assertRaises(TypeError):
            self.matrix + 'foo'
        with self.assertRaises(matrix.MatrixError):
            self.matrix + matrix.eye(self.n+1)

    def test_sub(self):
        j = self.n//2
        v = 10.
        other = matrix.assemble_coo(numpy.full(self.n, v), numpy.arange(self.n), self.n, numpy.full(self.n, j), self.n)
        sub = self.matrix - other
        numpy.testing.assert_equal(actual=sub.export('dense'), desired=self.exact - numpy.eye(self.n)[j]*v)
        with self.assertRaises(TypeError):
            self.matrix - 'foo'
        with self.assertRaises(matrix.MatrixError):
            self.matrix - matrix.eye(self.n+1)

    def test_transpose(self):
        asym = matrix.assemble_coo(numpy.arange(1, 7), [0, 0, 0, 1, 1, 2], 3, [0, 1, 2, 1, 2, 2], 3)
        exact = numpy.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=float)
        transpose = asym.T
        numpy.testing.assert_equal(actual=transpose.export('dense'), desired=exact.T)

    def test_rowsupp(self):
        sparse = matrix.assemble_coo([1e-10, 0, 1, 1], [0, 0, 2, 2], 3, [0, 1, 1, 2], 3)
        self.assertEqual(tuple(sparse.rowsupp(tol=1e-5)), (False, False, True))
        self.assertEqual(tuple(sparse.rowsupp(tol=0)), (True, False, True))

    def test_solve(self):
        rhs = numpy.arange(self.matrix.shape[0])
        for args in self.solve_args:
            for lhs0 in None, numpy.arange(rhs.size)/rhs.size:
                with self.subTest('{},lhs0={}'.format(args.get('solver', 'direct'), 'none' if lhs0 is None else 'single')):
                    lhs = self.matrix.solve(rhs, lhs0=lhs0, **args)
                    res = numpy.linalg.norm(self.matrix @ lhs - rhs)
                    self.assertLess(res, args.get('atol', 1e-10))

    def test_multisolve(self):
        rhs = numpy.arange(self.matrix.shape[0]*2).reshape(-1, 2)
        for name, lhs0 in ('none', None), ('single', numpy.arange(self.matrix.shape[1])), ('multi', numpy.arange(rhs.size).reshape(rhs.shape)):
            with self.subTest('lhs0={}'.format(name)):
                lhs = self.matrix.solve(rhs, lhs0=lhs0)
                res = numpy.linalg.norm(self.matrix @ lhs - rhs, axis=0)
                self.assertLess(numpy.max(res), 1e-9)

    def test_singular(self):
        singularmatrix = matrix.diag(numpy.arange(self.n)-self.n//2)
        rhs = numpy.ones(self.n)
        for args in self.solve_args:
            with self.subTest(args.get('solver', 'direct')), self.assertRaises(matrix.MatrixError):
                lhs = singularmatrix.solve(rhs, **dict(args, atol=.1))

    def test_solve_repeated(self):
        rhs = numpy.arange(self.matrix.shape[0])
        for args in self.solve_args:
            with self.subTest(args.get('solver', 'direct')):
                for i in range(3):
                    lhs = self.matrix.solve(rhs, **args)
                    res = numpy.linalg.norm(self.matrix @ lhs - rhs)
                    self.assertLess(res, args.get('atol', 1e-10))

    def test_constraints(self):
        cons = numpy.empty(self.matrix.shape[0])
        cons[:] = numpy.nan
        cons[0] = 10
        cons[-1] = 20
        for args in self.solve_args:
            with self.subTest(args.get('solver', 'direct')):
                lhs = self.matrix.solve(constrain=cons, **args)
                self.assertEqual(lhs[0], cons[0])
                self.assertEqual(lhs[-1], cons[-1])
                res = numpy.linalg.norm((self.matrix @ lhs)[1:-1])
                self.assertLess(res, args.get('atol', 1e-10))

    def test_submatrix(self):
        rows = self.n//2 + numpy.array([0, 1])
        cols = self.n//2 + numpy.array([-1, 0, 2])
        array = self.matrix.submatrix(rows, cols).export('dense')
        self.assertEqual(array.shape, (2, 3))
        numpy.testing.assert_equal(actual=array, desired=self.exact[numpy.ix_(rows, cols)])

    def test_submatrix_specialcases(self):
        mat = matrix.assemble_coo([1, 2, 3, 4], [0, 0, 2, 2], 3, [0, 2, 0, 2], 3)
        self.assertAllEqual(mat.export('dense'), [[1, 0, 2], [0, 0, 0], [3, 0, 4]])
        self.assertAllEqual(mat.submatrix([0, 2], [0, 1, 2]).export('dense'), [[1, 0, 2], [3, 0, 4]])
        self.assertAllEqual(mat.submatrix([0, 1, 2], [0, 2]).export('dense'), [[1, 2], [0, 0], [3, 4]])
        self.assertAllEqual(mat.submatrix([0, 2], [0, 2]).export('dense'), [[1, 2], [3, 4]])
        self.assertAllEqual(mat.submatrix([1], [1]).export('dense'), [[0]])

    def test_pickle(self):
        s = pickle.dumps(self.matrix)
        mat = pickle.loads(s)
        self.assertIsInstance(mat, type(self.matrix))
        numpy.testing.assert_equal(mat.export('dense'), self.exact)
        with self.subTest('cross-pickle'), matrix.backend('Numpy'):
            mat = pickle.loads(s)
            from nutils.matrix._numpy import NumpyMatrix
            self.assertIsInstance(mat, NumpyMatrix)
            numpy.testing.assert_equal(mat.export('dense'), self.exact)

    def test_diagonal(self):
        self.assertAllEqual(self.matrix.diagonal(), numpy.diag(self.exact))


backend('numpy',
        backend='numpy',
        solve_args=[{},
                    dict(solver='direct', atol=1e-8),
                    dict(atol=1e-5, precon='diag')])

backend('numpy:complex',
        backend='numpy',
        complex=True,
        solve_args=[{},
                    dict(solver='direct', atol=1e-8)])

backend('scipy',
        backend='scipy',
        solve_args=[{},
                    dict(solver='direct', atol=1e-8),
                    dict(atol=1e-5, precon='diag', truncate=5),
                    dict(solver='gmres', atol=1e-5, restart=100, precon='spilu0'),
                    dict(solver='gmres', atol=1e-5, precon='splu'),
                    dict(solver='cg', atol=1e-5, precon='diag')] + [
            dict(solver=s, atol=1e-5) for s in ('bicg', 'bicgstab', 'cg', 'cgs', 'lgmres')])

backend('scipy:complex',
        backend='scipy',
        complex=True,
        solve_args=[{},
                    dict(solver='direct', atol=1e-8)])

backend('mkl',
        backend='mkl',
        solve_args=[{},
                    dict(solver='direct', atol=1e-8),
                    dict(solver='direct', symmetric=True, atol=1e-8),
                    dict(atol=1e-5, precon='diag', truncate=5),
                    dict(solver='fgmres', atol=1e-8),
                    dict(solver='fgmres', atol=1e-8, precon='diag')])

backend('mkl:complex',
        backend='mkl',
        complex=True,
        solve_args=[{},
                    dict(solver='direct', atol=1e-8),
                    dict(solver='direct', symmetric=True, atol=1e-8)])
