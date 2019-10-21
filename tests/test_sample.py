from nutils import *
import random, itertools, functools
from nutils.testing import *

class rectilinear(TestCase):

  def setUp(self):
    self.domain, self.geom = mesh.rectilinear([2,1])
    self.bezier2 = self.domain.sample('bezier', 2)
    self.bezier3 = self.domain.sample('bezier', 3)
    self.gauss2 = self.domain.sample('gauss', 2)

  def test_integrate(self):
    area = self.gauss2.integrate(1)
    self.assertLess(abs(area-2), 1e-15)

  def test_integral(self):
    area = self.gauss2.integral(function.asarray(1)).eval()
    self.assertLess(abs(area-2), 1e-15)

  def test_eval(self):
    x = self.bezier3.eval(self.geom)
    self.assertEqual(x.shape, (self.bezier3.npoints,)+self.geom.shape)

  def test_tri(self):
    self.assertEqual(len(self.bezier2.tri), 4)
    self.assertEqual(len(self.bezier3.tri), 16)

  def test_hull(self):
    self.assertEqual(len(self.bezier2.hull), 8)
    self.assertEqual(len(self.bezier3.hull), 16)

  def test_subset(self):
    subset1 = self.bezier2.subset(numpy.eye(8)[0])
    subset2 = self.bezier2.subset(numpy.eye(8)[1])
    self.assertEqual(subset1.npoints, 4)
    self.assertEqual(subset2.npoints, 4)
    self.assertEqual(subset1, subset2)

  def test_asfunction(self):
    func = self.geom[0]**2 - self.geom[1]**2
    values = self.gauss2.eval(func)
    sampled = self.gauss2.asfunction(values)
    with self.assertRaises(function.EvaluationError):
      self.bezier2.eval(sampled)
    self.assertAllEqual(self.gauss2.eval(sampled), values)
    arg = function.Argument('dofs', [2,3])
    self.assertTrue(function.iszero(function.derivative(sampled, arg)))

class integral(TestCase):

  def setUp(self):
    self.ns = function.Namespace()
    self.topo, self.ns.x = mesh.rectilinear([5])
    self.ns.basis = self.topo.basis('std', degree=1)
    self.ns.v = 'basis_n ?lhs_n'
    self.lhs = numpy.sin(numpy.arange(len(self.ns.basis)))

  def test_eval(self):
    self.assertAllAlmostEqual(
      self.topo.integrate('basis_n d:x' @ self.ns, degree=2),
      self.topo.integral('basis_n d:x' @ self.ns, degree=2).eval(),
      places=15)

  def test_args(self):
    self.assertAlmostEqual(
      self.topo.integrate('v d:x' @ self.ns, degree=2, arguments=dict(lhs=self.lhs)),
      self.topo.integral('v d:x' @ self.ns, degree=2).eval(lhs=self.lhs),
      places=15)

  def test_derivative(self):
    self.assertAllAlmostEqual(
      self.topo.integrate('2 basis_n v d:x' @ self.ns, degree=2, arguments=dict(lhs=self.lhs)),
      self.topo.integral('v^2 d:x' @ self.ns, degree=2).derivative('lhs').eval(lhs=self.lhs),
      places=15)

  def test_transpose(self):
    self.assertAllAlmostEqual(
      self.topo.integrate(self.ns.eval_nm('basis_n (basis_m + 1_m) d:x'), degree=2).export('dense').T,
      self.topo.integral(self.ns.eval_nm('basis_n (basis_m + 1_m) d:x'), degree=2).T.eval().export('dense'),
      places=15)

  def test_empty(self):
    shape = 2, 3
    empty = sample.Integral({}, shape=shape)
    array = empty.eval().export('dense')
    self.assertEqual(array.shape, shape)
    self.assertAllEqual(array.flat, 0)
