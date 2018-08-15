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
    resnorm = numpy.linalg.norm(res[numpy.isnan(self.cons)])
    self.assertLess(resnorm, self.tol)

  def test_direct(self):
    with self.assertRaises(solver.ModelError):
      self.assert_resnorm(solver.solve_linear('dofs', residual=self.residual, constrain=self.cons))

  def test_newton(self):
    self.assert_resnorm(solver.newton('dofs', residual=self.residual, lhs0=self.lhs0, constrain=self.cons).solve(tol=self.tol, maxiter=2))

  def test_newton_iter(self):
    _test_recursion_cache(self, lambda: ((types.frozenarray(lhs), info.resnorm) for lhs, info in solver.newton('dofs', residual=self.residual, constrain=self.cons)))

  def test_pseudotime(self):
    self.assert_resnorm(solver.pseudotime('dofs', residual=self.residual, lhs0=self.lhs0, constrain=self.cons, inertia=self.inertia, timestep=1).solve(tol=self.tol, maxiter=3))

  def test_pseudotime_iter(self):
    _test_recursion_cache(self, lambda: ((types.frozenarray(lhs), info.resnorm) for lhs, info in solver.pseudotime('dofs', residual=self.residual, lhs0=self.lhs0, constrain=self.cons, inertia=self.inertia, timestep=1)))


class finitestrain(TestCase):

  def setUp(self):
    super().setUp()
    domain, geom = mesh.rectilinear([numpy.linspace(0,1,9)] * 2)
    ubasis = domain.basis('std', degree=2).vector(2)
    u = ubasis.dot(function.Argument('dofs', [len(ubasis)]))
    Geom = geom * [1.1, 1] + u
    self.cons = solver.minimize('dofs', domain.boundary['left,right'].integral((u**2).sum(0), degree=4), droptol=1e-15).solve()
    self.boolcons = ~numpy.isnan(self.cons)
    strain = .5 * (function.outer(Geom.grad(geom), axis=1).sum(0) - function.eye(2))
    self.energy = domain.integral((strain**2).sum([0,1]) + 20*(function.determinant(Geom.grad(geom))-1)**2, geometry=geom, degree=6)
    self.residual = self.energy.derivative('dofs')
    self.tol = 1e-10

  def assert_resnorm(self, lhs):
    res = self.residual.eval(arguments=dict(dofs=lhs))
    resnorm = numpy.linalg.norm(res[~self.boolcons])
    self.assertLess(resnorm, self.tol)

  def test_direct(self):
    with self.assertRaises(solver.ModelError):
      self.assert_resnorm(solver.solve_linear('dofs', residual=self.residual, constrain=self.cons))

  def test_newton(self):
    self.assert_resnorm(solver.newton('dofs', residual=self.residual, constrain=self.cons).solve(tol=self.tol, maxiter=7))

  def test_newton_boolcons(self):
    self.assert_resnorm(solver.newton('dofs', residual=self.residual, constrain=self.boolcons).solve(tol=self.tol, maxiter=7))

  def test_newton_iter(self):
    _test_recursion_cache(self, lambda: ((types.frozenarray(lhs), info.resnorm) for lhs, info in solver.newton('dofs', residual=self.residual, constrain=self.cons)))

  def test_minimize(self):
    self.assert_resnorm(solver.minimize('dofs', energy=self.energy, constrain=self.cons).solve(tol=self.tol, maxiter=8))

  def test_minimize_boolcons(self):
    self.assert_resnorm(solver.minimize('dofs', energy=self.energy, constrain=self.boolcons).solve(tol=self.tol, maxiter=8))

  def test_minimize_iter(self):
    _test_recursion_cache(self, lambda: ((types.frozenarray(lhs), info.resnorm) for lhs, info in solver.minimize('dofs', energy=self.energy, constrain=self.cons)))

  def test_nonlinear_diagonalshift(self):
    nelems = 10
    domain, geom = mesh.rectilinear([nelems,1])
    geom *= [2*numpy.pi/nelems, 1]
    ubasis = domain.basis('spline', degree=2).vector(2)
    u = ubasis.dot(function.Argument('dofs', [len(ubasis)]))
    Geom = [.5 * geom[0], geom[1] + function.cos(geom[0])] + u # compress by 50% and buckle
    cons = solver.minimize('dofs', domain.boundary['left,right'].integral((u**2).sum(0), degree=4), droptol=1e-15).solve()
    strain = .5 * (function.outer(Geom.grad(geom), axis=1).sum(0) - function.eye(2))
    energy = domain.integral((strain**2).sum([0,1]) + 150*(function.determinant(Geom.grad(geom))-1)**2, geometry=geom, degree=6)
    nshift = 0
    for iiter, (lhs, info) in enumerate(solver.minimize('dofs', energy, constrain=cons)):
      self.assertLess(iiter, 38)
      if info.shift:
        nshift += 1
      if info.resnorm < self.tol:
        break
    self.assertEqual(nshift, 9)


@parametrize
class optimize(TestCase):

  def setUp(self):
    super().setUp()
    self.ns = function.Namespace()
    self.domain, self.ns.geom = mesh.rectilinear([2,2])
    self.ns.ubasis = self.domain.basis('std', degree=1)
    self.ns.u = 'ubasis_n ?dofs_n'
    self.optimize = solver.optimize if not self.minimize else lambda *args, newtontol=0, **kwargs: solver.minimize(*args, **kwargs).solve(newtontol)

  def test_linear(self):
    err = self.domain.boundary['bottom'].integral('(u - 1)^2' @ self.ns, degree=2)
    cons = self.optimize('dofs', err, droptol=1e-15)
    numpy.testing.assert_almost_equal(cons, numpy.take([1,numpy.nan], [0,1,1,0,1,1,0,1,1]), decimal=15)

  def test_nonlinear(self):
    err = self.domain.boundary['bottom'].integral('(u + .25 u^3 - 1.25)^2' @ self.ns, geometry=self.ns.geom, degree=6)
    cons = self.optimize('dofs', err, droptol=1e-15, newtontol=1e-15)
    numpy.testing.assert_almost_equal(cons, numpy.take([1,numpy.nan], [0,1,1,0,1,1,0,1,1]), decimal=15)

  def test_nonlinear_multipleroots(self):
    err = self.domain.boundary['bottom'].integral('(u + u^2 - .75)^2' @ self.ns, degree=2)
    cons = self.optimize('dofs', err, droptol=1e-15, lhs0=numpy.ones(len(self.ns.ubasis)), newtontol=1e-10)
    numpy.testing.assert_almost_equal(cons, numpy.take([.5,numpy.nan], [0,1,1,0,1,1,0,1,1]), decimal=15)

  def test_nanres(self):
    err = self.domain.integral('(sqrt(1 - u) - .5)^2' @ self.ns, degree=2)
    dofs = self.optimize('dofs', err, newtontol=1e-10)
    numpy.testing.assert_almost_equal(dofs, .75)

optimize(minimize=False)
optimize(minimize=True)
