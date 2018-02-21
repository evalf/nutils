import numpy
from nutils import matrix
from . import *

@parametrize
class solver(TestCase):

  ifsupported = parametrize.skip_if(lambda backend, args: not hasattr(matrix, backend), reason='not supported')
  n = 100

  def setUp(self):
    super().setUp()
    self._backend = matrix.backend(self.backend)
    self._backend.__enter__()
    r = numpy.arange(self.n)
    index = numpy.concatenate([[r, r], [r[:-1], r[1:]], [r[1:], r[:-1]]], axis=1)
    data = numpy.hstack([2.] * self.n + [-1.] * (2*self.n-2))
    self.matrix = matrix.assemble(data, index, shape=(self.n, self.n))
    self.exact = 2 * numpy.eye(self.n) - numpy.eye(self.n, self.n, -1) - numpy.eye(self.n, self.n, +1)
    self.tol = self.args.get('atol', 1e-10)

  def tearDown(self):
    self._backend.__exit__(None, None, None)

  @ifsupported
  def test_scalar(self):
    s = matrix.assemble(numpy.array([1.,2.]), index=numpy.empty((0,2), dtype=int), shape=())
    self.assertEqual(s, 3.)

  @ifsupported
  def test_vector(self):
    v = matrix.assemble(numpy.array([1.,2.,3.]), index=numpy.array([[0,2,0]]), shape=(3,))
    self.assertEqual(tuple(v), (4.,0.,2.))

  @ifsupported
  def test_size(self):
    self.assertEqual(self.matrix.size, self.n**2)

  @ifsupported
  def test_export_dense(self):
    array = self.matrix.export('dense')
    self.assertEqual(array.shape, (self.n, self.n))
    numpy.testing.assert_equal(actual=array, desired=self.exact)

  @ifsupported
  def test_export_coo(self):
    data, (row, col) = self.matrix.export('coo')
    numpy.testing.assert_equal(row[0::3], numpy.arange(self.n))
    numpy.testing.assert_equal(col[0::3], numpy.arange(self.n))
    numpy.testing.assert_equal(data[0::3], 2)
    numpy.testing.assert_equal(row[1::3], numpy.arange(self.n-1))
    numpy.testing.assert_equal(col[1::3], numpy.arange(1, self.n))
    numpy.testing.assert_equal(data[1::3], -1)
    numpy.testing.assert_equal(row[2::3], numpy.arange(1, self.n))
    numpy.testing.assert_equal(col[2::3], numpy.arange(self.n-1))
    numpy.testing.assert_equal(data[2::3], -1)

  @ifsupported
  def test_export_csr(self):
    data, indices, indptr = self.matrix.export('csr')
    self.assertEqual(indptr[0], 0)
    self.assertEqual(indptr[-1], len(data))
    numpy.testing.assert_equal(data[0::3], 2)
    numpy.testing.assert_equal(data[1::3], -1)
    numpy.testing.assert_equal(data[2::3], -1)
    numpy.testing.assert_equal(indices[0::3], numpy.arange(self.n))
    numpy.testing.assert_equal(indices[1::3], numpy.arange(1, self.n))
    numpy.testing.assert_equal(indices[2::3], numpy.arange(self.n-1))
    numpy.testing.assert_equal(indptr[1:-1], numpy.arange(2, 3*(self.n-1), 3))

  @ifsupported
  def test_neg(self):
    neg = -self.matrix
    numpy.testing.assert_equal(actual=neg.export('dense'), desired=-self.exact)

  @ifsupported
  def test_mul(self):
    mul = self.matrix * 1.5
    numpy.testing.assert_equal(actual=mul.export('dense'), desired=self.exact * 1.5)

  @ifsupported
  def test_rmul(self):
    rmul = 1.5 * self.matrix
    numpy.testing.assert_equal(actual=rmul.export('dense'), desired=self.exact * 1.5)

  @ifsupported
  def test_div(self):
    div = self.matrix / 1.5
    numpy.testing.assert_equal(actual=div.export('dense'), desired=self.exact / 1.5)

  @ifsupported
  def test_add(self):
    j = self.n//2
    v = 10.
    other = matrix.assemble(numpy.array([v]*self.n), numpy.array([numpy.arange(self.n),[j]*self.n]), shape=(self.n, self.n))
    add = self.matrix + other
    numpy.testing.assert_equal(actual=add.export('dense'), desired=self.exact + numpy.eye(self.n)[j]*v)

  @ifsupported
  def test_sub(self):
    j = self.n//2
    v = 10.
    other = matrix.assemble(numpy.array([v]*self.n), numpy.array([numpy.arange(self.n),[j]*self.n]), shape=(self.n, self.n))
    sub = self.matrix - other
    numpy.testing.assert_equal(actual=sub.export('dense'), desired=self.exact - numpy.eye(self.n)[j]*v)

  @ifsupported
  def test_transpose(self):
    asym = matrix.assemble(numpy.array([1,2,3,4], dtype=float), numpy.array([[0,0,1,1],[0,1,1,2]]), shape=(2,3))
    exact = numpy.array([[1,2,0],[0,3,4]], dtype=float)
    transpose = asym.T
    numpy.testing.assert_equal(actual=transpose.export('dense'), desired=exact.T)

  @ifsupported
  def test_rowsupp(self):
    sparse = matrix.assemble(numpy.array([1e-10,0,1,1]), numpy.array([[0,0,2,2],[0,1,1,2]]), shape=(3,3))
    self.assertEqual(tuple(sparse.rowsupp(tol=1e-5)), (False,False,True))
    self.assertEqual(tuple(sparse.rowsupp(tol=0)), (True,False,True))

  @ifsupported
  def test_solve(self):
    rhs = numpy.arange(self.matrix.shape[0])
    lhs = self.matrix.solve(rhs, **self.args)
    res = numpy.linalg.norm(self.matrix.matvec(lhs) - rhs)
    self.assertLess(res, self.tol)

  @ifsupported
  def test_solve_repeated(self):
    rhs = numpy.arange(self.matrix.shape[0])
    for i in range(3):
      lhs = self.matrix.solve(rhs, **self.args)
      res = numpy.linalg.norm(self.matrix.matvec(lhs) - rhs)
      self.assertLess(res, self.tol)

  @ifsupported
  def test_constraints(self):
    cons = numpy.empty(self.matrix.shape[0])
    cons[:] = numpy.nan
    cons[0] = 10
    cons[-1] = 20
    lhs = self.matrix.solve(constrain=cons, **self.args)
    self.assertEqual(lhs[0], cons[0])
    self.assertEqual(lhs[-1], cons[-1])
    cons[1:-1] = 0
    res = numpy.linalg.norm(self.matrix.matvec(lhs)[1:-1])
    self.assertLess(res, self.tol)

solver(backend='Numpy', args=dict())
solver(backend='Scipy', args=dict())
solver(backend='Scipy', args=dict(atol=1e-5, solver='gmres', restart=100, precon='spilu'))
solver(backend='Scipy', args=dict(atol=1e-5, solver='cg', precon='diag'))
solver(backend='Scipy', args=dict(atol=1e-5, solver='lgmres'))
solver(backend='MKL', args=dict())
