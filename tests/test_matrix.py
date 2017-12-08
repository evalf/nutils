import numpy, scipy.sparse
from nutils import matrix
from . import *

@parametrize
class solver(TestCase):

  def setUp(self):
    super().setUp()
    n = 100
    self.matrix = matrix.ScipyMatrix(scipy.sparse.spdiags(numpy.repeat([[-1],[2],[-1]], n, axis=1), [-1,0,1], n, n).tocsr())
    self.tol = self.args.get('tol', 1e-12)

  def test_solve(self):
    rhs = numpy.arange(self.matrix.shape[0])
    lhs = self.matrix.solve(rhs, **self.args)
    res = numpy.linalg.norm(self.matrix.matvec(lhs) - rhs) / numpy.linalg.norm(rhs)
    self.assertLess(res, self.tol)

  def test_constraints(self):
    cons = numpy.empty(self.matrix.shape[0])
    cons[:] = numpy.nan
    cons[0] = 10
    cons[-1] = 20
    lhs = self.matrix.solve(constrain=cons, **self.args)
    self.assertEqual(lhs[0], cons[0])
    self.assertEqual(lhs[-1], cons[-1])
    cons[1:-1] = 0
    res = numpy.linalg.norm(self.matrix.matvec(lhs)[1:-1]) / numpy.linalg.norm(self.matrix.matvec(cons)[1:-1])
    self.assertLess(res, self.tol)

solver(args=dict())
solver(args=dict(tol=1e-5, restart=100))
solver(args=dict(tol=1e-5, symmetric=True))
solver(args=dict(tol=1e-5, solver='lgmres'))
