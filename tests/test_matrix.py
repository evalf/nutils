import numpy
from nutils import matrix
from . import *

@parametrize
class solver(TestCase):

  ifsupported = parametrize.skip_if(lambda backend, args: not hasattr(matrix, backend), reason='not supported')

  def setUp(self):
    super().setUp()
    self._backend = matrix.backend(self.backend)
    self._backend.__enter__()
    n = 100
    r = numpy.arange(n)
    index = numpy.concatenate([[r, r], [r[:-1], r[1:]], [r[1:], r[:-1]]], axis=1)
    data = numpy.hstack([2.] * n + [-1.] * (2*n-2))
    self.matrix = matrix.assemble(data, index, shape=(n,n))
    self.tol = self.args.get('atol', 1e-10)

  def tearDown(self):
    self._backend.__exit__(None, None, None)

  @ifsupported
  def test_solve(self):
    rhs = numpy.arange(self.matrix.shape[0])
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
solver(backend='Scipy', args=dict(atol=1e-5, solver='gmres', restart=100))
solver(backend='Scipy', args=dict(atol=1e-5, solver='cg'))
solver(backend='Scipy', args=dict(atol=1e-5, solver='lgmres'))
solver(backend='MKL', args=dict())
