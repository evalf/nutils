from nutils import solver, mesh, function, cache, types
from . import *
import numpy, contextlib, tempfile

@contextlib.contextmanager
def tmpcache():
  with tempfile.TemporaryDirectory() as tmpdir:
    with cache.enable(tmpdir):
      yield

def _test_recursion_cache(testcase, solver_iter):
  read = lambda n: tuple(item for i, item in zip(range(n), solver_iter()))
  reference = read(5)
  for lengths in [1,2,3], [1,3,2], [0,3,5]:
    with tmpcache():
      for i, length in enumerate(lengths):
        with testcase.subTest(lengths=lengths, step=i):
          testcase.assertEqual(read(length), reference[:length])


class laplace(TestCase):

  def setUp(self):
    super().setUp()
    domain, geom = mesh.rectilinear([8,8])
    basis = domain.basis('std', degree=1)
    self.cons = domain.boundary['left'].project(0, onto=basis, geometry=geom, ischeme='gauss2')
    dofs = function.Argument('dofs', [len(basis)])
    u = basis.dot(dofs)
    self.residual = domain.integral((basis.grad(geom) * u.grad(geom)).sum(-1), geometry=geom, degree=2) \
                  + domain.boundary['top'].integral(basis, geometry=geom, degree=2)

  def test_res(self):
    for name in 'direct', 'newton':
      with self.subTest(name):
        if name == 'direct':
          lhs = solver.solve_linear('dofs', residual=self.residual, constrain=self.cons)
        else:
          lhs = solver.newton('dofs', residual=self.residual, constrain=self.cons).solve(tol=1e-10, maxiter=0)
        res = self.residual.eval(arguments=dict(dofs=lhs))
        resnorm = numpy.linalg.norm(res[~self.cons.where])
        self.assertLess(resnorm, 1e-13)


class navierstokes(TestCase):

  def setUp(self):
    super().setUp()
    domain, geom = mesh.rectilinear([numpy.linspace(0,1,9)] * 2)
    ubasis, pbasis = function.chain([
      domain.basis('std', degree=2).vector(2),
      domain.basis('std', degree=1),
    ])
    dofs = function.Argument('dofs', [len(ubasis)])
    u = ubasis.dot(dofs)
    p = pbasis.dot(dofs)
    viscosity = 1
    self.inertia = domain.integral((ubasis * u).sum(-1), geometry=geom, degree=5)
    stokesres = domain.integral(viscosity * (ubasis.grad(geom) * (u.grad(geom)+u.grad(geom).T)).sum([-1,-2]) - ubasis.div(geom) * p + pbasis * u.div(geom), geometry=geom, degree=5)
    self.residual = stokesres + domain.integral((ubasis * (u.grad(geom) * u).sum(-1) * u).sum(-1), geometry=geom, degree=5)
    self.cons = domain.boundary['top,bottom'].project([0,0], onto=ubasis, geometry=geom, ischeme='gauss2') \
              | domain.boundary['left'].project([geom[1]*(1-geom[1]),0], onto=ubasis, geometry=geom, ischeme='gauss2')
    self.lhs0 = solver.solve_linear('dofs', residual=stokesres, constrain=self.cons)
    self.tol = 1e-10

  def assert_resnorm(self, lhs):
    res = self.residual.eval(arguments=dict(dofs=lhs))
    resnorm = numpy.linalg.norm(res[~self.cons.where])
    self.assertLess(resnorm, self.tol)

  def test_direct(self):
    with self.assertRaises(solver.ModelError):
      self.assert_resnorm(solver.solve_linear('dofs', residual=self.residual, constrain=self.cons))

  def test_newton(self):
    self.assert_resnorm(solver.newton('dofs', residual=self.residual, lhs0=self.lhs0, constrain=self.cons).solve(tol=self.tol, maxiter=2))

  def test_newton_iter(self):
    _test_recursion_cache(self, lambda: ((types.frozenarray(lhs), info.resnorm) for lhs, info in solver.newton('dofs', residual=self.residual, lhs0=self.lhs0, constrain=self.cons, nrelax=1)))

  def test_pseudotime(self):
    self.assert_resnorm(solver.pseudotime('dofs', residual=self.residual, lhs0=self.lhs0, constrain=self.cons, inertia=self.inertia, timestep=1).solve(tol=self.tol, maxiter=3))

  def test_pseudotime_iter(self):
    _test_recursion_cache(self, lambda: ((types.frozenarray(lhs), info.resnorm) for lhs, info in solver.pseudotime('dofs', residual=self.residual, lhs0=self.lhs0, constrain=self.cons, inertia=self.inertia, timestep=1)))


class optimize(TestCase):

  def setUp(self):
    super().setUp()
    self.ns = function.Namespace()
    self.domain, self.ns.geom = mesh.rectilinear([2,2])
    self.ns.ubasis = self.domain.basis('std', degree=1)

  def test_linear(self):
    ns = self.ns
    ns.u = 'ubasis_n ?dofs_n'
    err = self.domain.boundary['bottom'].integral(ns.eval_('(u - 1)^2'), geometry=ns.geom, degree=2)
    cons = solver.optimize('dofs', err, droptol=1e-15)
    isnan = numpy.isnan(cons)
    self.assertTrue(numpy.equal(isnan, [0,1,1,0,1,1,0,1,1]).all())
    numpy.testing.assert_almost_equal(cons[~isnan], 1, decimal=15)

  def test_nonlinear(self):
    ns = self.ns
    ns.u = 'ubasis_n ?dofs_n'
    ns.fu = 'u + .25 u^3'
    err = self.domain.boundary['bottom'].integral(ns.eval_('(fu - 1.25)^2'), geometry=ns.geom, degree=6)
    cons = solver.optimize('dofs', err, droptol=1e-15, newtontol=1e-15)
    isnan = numpy.isnan(cons)
    self.assertTrue(numpy.equal(isnan, [0,1,1,0,1,1,0,1,1]).all())
    numpy.testing.assert_almost_equal(cons[~isnan], 1, decimal=15)

  def test_nonlinear_multipleroots(self):
    ns = self.ns
    ns.u = 'ubasis_n ?dofs_n'
    ns.gu = 'u + u^2'
    err = self.domain.boundary['bottom'].integral(ns.eval_('(gu - .75)^2'), geometry=ns.geom, degree=2)
    cons = solver.optimize('dofs', err, droptol=1e-15, lhs0=numpy.ones(len(ns.ubasis)), newtontol=1e-10)
    isnan = numpy.isnan(cons)
    self.assertTrue(numpy.equal(isnan, [0,1,1,0,1,1,0,1,1]).all())
    numpy.testing.assert_almost_equal(cons[~isnan], .5, decimal=15)
