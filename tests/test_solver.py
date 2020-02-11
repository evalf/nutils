from nutils import solver, mesh, function, cache, types, numeric, warnings
from nutils.testing import *
import numpy, contextlib, tempfile, itertools, logging

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
    self.residual = domain.integral((basis.grad(geom) * u.grad(geom)).sum(-1)*function.J(geom), degree=2) \
                  + domain.boundary['top'].integral(basis*function.J(geom), degree=2)

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
    viscosity = 1e-3
    self.inertia = domain.integral((ubasis * u).sum(-1)*function.J(geom), degree=5)
    stokesres = domain.integral((viscosity * (ubasis.grad(geom) * (u.grad(geom)+u.grad(geom).T)).sum([-1,-2]) - ubasis.div(geom) * p + pbasis * u.div(geom))*function.J(geom), degree=5)
    self.residual = stokesres + domain.integral((ubasis * (u.grad(geom) * u).sum(-1) * u).sum(-1)*function.J(geom), degree=5)
    self.cons = domain.boundary['top,bottom'].project([0,0], onto=ubasis, geometry=geom, ischeme='gauss4') \
              | domain.boundary['left'].project([geom[1]*(1-geom[1]),0], onto=ubasis, geometry=geom, ischeme='gauss4')
    self.lhs0 = solver.solve_linear('dofs', residual=stokesres, constrain=self.cons)
    self.tol = 1e-10

  def assert_resnorm(self, lhs):
    res = self.residual.eval(arguments=dict(dofs=lhs))
    resnorm = numpy.linalg.norm(res[numpy.isnan(self.cons)])
    self.assertLess(resnorm, self.tol)

  def test_direct(self):
    with self.assertRaises(solver.SolverError):
      self.assert_resnorm(solver.solve_linear('dofs', residual=self.residual, constrain=self.cons))

  def test_newton(self):
    self.assert_resnorm(solver.newton('dofs', residual=self.residual, lhs0=self.lhs0, constrain=self.cons).solve(tol=self.tol, maxiter=2))

  def test_newton_medianbased(self):
    self.assert_resnorm(solver.newton('dofs', residual=self.residual, lhs0=self.lhs0, constrain=self.cons, linesearch=solver.MedianBased()).solve(tol=self.tol, maxiter=2))

  def test_newton_relax0(self):
    self.assert_resnorm(solver.newton('dofs', residual=self.residual, lhs0=self.lhs0, constrain=self.cons, relax0=.1).solve(tol=self.tol, maxiter=5))

  def test_newton_tolnotreached(self):
    with self.assertLogs('nutils', logging.WARNING) as cm:
      self.assert_resnorm(solver.newton('dofs', residual=self.residual, lhs0=self.lhs0, constrain=self.cons, linrtol=1e-99).solve(tol=self.tol, maxiter=2))
    for msg in cm.output:
      self.assertIn('solver failed to reach tolerance', msg)

  def test_newton_iter(self):
    _test_recursion_cache(self, lambda: ((types.frozenarray(lhs), info.resnorm) for lhs, info in solver.newton('dofs', residual=self.residual, constrain=self.cons)))

  def test_pseudotime(self):
    self.assert_resnorm(solver.pseudotime('dofs', residual=self.residual, lhs0=self.lhs0, constrain=self.cons, inertia=self.inertia, timestep=1).solve(tol=self.tol, maxiter=12))

  def test_pseudotime_iter(self):
    _test_recursion_cache(self, lambda: ((types.frozenarray(lhs), info.resnorm) for lhs, info in solver.pseudotime('dofs', residual=self.residual, lhs0=self.lhs0, constrain=self.cons, inertia=self.inertia, timestep=1)))


class finitestrain(TestCase):

  def setUp(self):
    super().setUp()
    domain, geom = mesh.rectilinear([numpy.linspace(0,1,9)] * 2)
    ubasis = domain.basis('std', degree=2).vector(2)
    u = ubasis.dot(function.Argument('dofs', [len(ubasis)]))
    Geom = geom * [1.1, 1] + u
    self.cons = solver.optimize('dofs', domain.boundary['left,right'].integral((u**2).sum(0), degree=4), droptol=1e-15)
    self.boolcons = ~numpy.isnan(self.cons)
    strain = .5 * (function.outer(Geom.grad(geom), axis=1).sum(0) - function.eye(2))
    self.energy = domain.integral(((strain**2).sum([0,1]) + 20*(function.determinant(Geom.grad(geom))-1)**2)*function.J(geom), degree=6)
    self.residual = self.energy.derivative('dofs')
    self.tol = 1e-10

  def assert_resnorm(self, lhs):
    res = self.residual.eval(arguments=dict(dofs=lhs))
    resnorm = numpy.linalg.norm(res[~self.boolcons])
    self.assertLess(resnorm, self.tol)

  def test_direct(self):
    with self.assertRaises(solver.SolverError):
      self.assert_resnorm(solver.solve_linear('dofs', residual=self.residual, constrain=self.cons))

  def test_newton(self):
    self.assert_resnorm(solver.newton('dofs', residual=self.residual, constrain=self.cons).solve(tol=self.tol, maxiter=7))

  def test_newton_boolcons(self):
    self.assert_resnorm(solver.newton('dofs', residual=self.residual, constrain=self.boolcons).solve(tol=self.tol, maxiter=7))

  def test_newton_iter(self):
    _test_recursion_cache(self, lambda: ((types.frozenarray(lhs), info.resnorm) for lhs, info in solver.newton('dofs', residual=self.residual, constrain=self.cons)))

  def test_minimize(self):
    self.assert_resnorm(solver.minimize('dofs', energy=self.energy, constrain=self.cons).solve(tol=self.tol, maxiter=12))

  def test_minimize_boolcons(self):
    self.assert_resnorm(solver.minimize('dofs', energy=self.energy, constrain=self.boolcons).solve(tol=self.tol, maxiter=12))

  def test_minimize_iter(self):
    _test_recursion_cache(self, lambda: ((types.frozenarray(lhs), info.resnorm) for lhs, info in solver.minimize('dofs', energy=self.energy, constrain=self.cons)))


class optimize(TestCase):

  def setUp(self):
    super().setUp()
    self.ns = function.Namespace()
    self.domain, self.ns.geom = mesh.rectilinear([2,2])
    self.ns.ubasis = self.domain.basis('std', degree=1)
    self.ns.u = 'ubasis_n ?dofs_n'

  def test_linear(self):
    err = self.domain.boundary['bottom'].integral('(u - 1)^2' @ self.ns, degree=2)
    cons = solver.optimize('dofs', err, droptol=1e-15)
    numpy.testing.assert_almost_equal(cons, numpy.take([1,numpy.nan], [0,1,1,0,1,1,0,1,1]), decimal=15)

  def test_nonlinear(self):
    err = self.domain.boundary['bottom'].integral('(u + .25 u^3 - 1.25)^2 d:geom' @ self.ns, degree=6)
    cons = solver.optimize('dofs', err, droptol=1e-15, tol=1e-15)
    numpy.testing.assert_almost_equal(cons, numpy.take([1,numpy.nan], [0,1,1,0,1,1,0,1,1]), decimal=15)

  def test_nonlinear_multipleroots(self):
    err = self.domain.boundary['bottom'].integral('(u + u^2 - .75)^2' @ self.ns, degree=2)
    cons = solver.optimize('dofs', err, droptol=1e-15, lhs0=numpy.ones(len(self.ns.ubasis)), tol=1e-10)
    numpy.testing.assert_almost_equal(cons, numpy.take([.5,numpy.nan], [0,1,1,0,1,1,0,1,1]), decimal=15)

  def test_nanres(self):
    err = self.domain.integral('(sqrt(1 - u) - .5)^2' @ self.ns, degree=2)
    dofs = solver.optimize('dofs', err, tol=1e-10)
    numpy.testing.assert_almost_equal(dofs, .75)


class burgers(TestCase):

  def setUp(self):
    ns = function.Namespace()
    domain, ns.x = mesh.rectilinear([10], periodic=(0,))
    ns.basis = domain.basis('discont', degree=1)
    ns.u = 'basis_n ?dofs_n'
    ns.f = '.5 u^2'
    self.residual = domain.integral('-basis_n,0 f d:x' @ ns, degree=2)
    self.residual += domain.interfaces.integral('-[basis_n] n_0 ({f} - .5 [u] n_0) d:x' @ ns, degree=4)
    self.inertia = domain.integral('basis_n u d:x' @ ns, degree=5)
    self.lhs0 = numpy.sin(numpy.arange(len(ns.basis))) # "random" initial vector

  def test_iters(self):
    it = iter(solver.impliciteuler('dofs', residual=self.residual, inertia=self.inertia, lhs0=self.lhs0, timestep=100)) # involves 2-level timestep scaling
    assert numpy.equal(next(it), self.lhs0).all()
    self.assertAlmostEqual64(next(it), 'eNpzNBA1NjHuNHQ3FDsTfCbAuNz4nUGZgeyZiDOZxlONmQwU9W3OFJ/pNQAADZIOPA==')

  def test_resume(self):
    _test_recursion_cache(self, lambda: map(types.frozenarray, solver.impliciteuler('dofs', residual=self.residual, inertia=self.inertia, lhs0=self.lhs0, timestep=1)))

  def test_resume_withscaling(self):
    _test_recursion_cache(self, lambda: map(types.frozenarray, solver.impliciteuler('dofs', residual=self.residual, inertia=self.inertia, lhs0=self.lhs0, timestep=100)))


class theta_time(TestCase):

  def check(self, method, theta):
    ns = function.Namespace()
    topo, ns.x = mesh.rectilinear([1])
    ns.u_n = '?u_n + <0>_n'
    inertia = topo.integral('?u_n d:x' @ ns, degree=0)
    residual = topo.integral('-<1>_n sin(?t) d:x' @ ns, degree=0)
    timestep = 0.1
    udesired = numpy.array([0.])
    uactualiter = iter(method(target='u', residual=residual, inertia=inertia, timestep=timestep, lhs0=udesired, timetarget='t'))
    for i in range(5):
      with self.subTest(i=i):
        uactual = next(uactualiter)
        self.assertAllAlmostEqual(uactual, udesired)
        udesired += timestep*(theta*numpy.sin((i+1)*timestep)+(1-theta)*numpy.sin(i*timestep))

  def test_impliciteuler(self):
    self.check(solver.impliciteuler, theta=1)

  def test_cranknicolson(self):
    self.check(solver.cranknicolson, theta=0.5)
