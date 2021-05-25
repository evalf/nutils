from nutils import solver, mesh, function, cache, types, numeric, warnings, sample, sparse
from nutils.testing import *
import numpy, contextlib, tempfile, itertools, logging, functools

@contextlib.contextmanager
def tmpcache():
  with tempfile.TemporaryDirectory() as tmpdir:
    with cache.enable(tmpdir):
      yield

def _test_recursion_cache(testcase, solver_iter):
  edit = lambda v: v.tolist() if numeric.isarray(v) else tuple(map(edit, v)) if isinstance(v, tuple) else v # convert arrays to lists
  read = lambda n: tuple(edit(item) for i, item in zip(range(n), solver_iter()))
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


@parametrize
class navierstokes(TestCase):

  viscosity = 1e-3

  def setUp(self):
    super().setUp()
    domain, geom = mesh.rectilinear([numpy.linspace(0,1,9)] * 2)
    gauss = domain.sample('gauss', 5)
    uin = geom[1] * (1-geom[1])
    dx = function.J(geom, 2)
    ubasis = domain.basis('std', degree=2)
    pbasis = domain.basis('std', degree=1)
    if self.single:
      ubasis, pbasis = function.chain([ubasis.vector(2), pbasis])
      dofs = function.Argument('dofs', [len(ubasis)])
      u = ubasis.dot(dofs)
      p = pbasis.dot(dofs)
      dofs = 'dofs'
      ures = gauss.integral((self.viscosity * (ubasis.grad(geom) * (u.grad(geom) + u.grad(geom).T)).sum([-1,-2]) - ubasis.div(geom) * p) * dx)
      dres = gauss.integral((ubasis * (u.grad(geom) * u).sum(-1)).sum(-1) * dx)
    else:
      u = (ubasis[:,numpy.newaxis] * function.Argument('dofs', [len(ubasis), 2])).sum(0)
      p = (pbasis * function.Argument('pdofs', [len(pbasis)])).sum(0)
      dofs = 'dofs', 'pdofs'
      ures = gauss.integral((self.viscosity * (ubasis[:,numpy.newaxis].grad(geom) * (u.grad(geom) + u.grad(geom).T)).sum(-1) - ubasis.grad(geom) * p) * dx)
      dres = gauss.integral(ubasis[:,numpy.newaxis] * (u.grad(geom) * u).sum(-1) * dx)
    pres = gauss.integral((pbasis * u.div(geom)) * dx)
    cons = solver.optimize('dofs', domain.boundary['top,bottom'].integral((u**2).sum(), degree=4), droptol=1e-10)
    cons = solver.optimize('dofs', domain.boundary['left'].integral((u[0]-uin)**2 + u[1]**2, degree=4), droptol=1e-10, constrain=cons)
    self.cons = cons if self.single else {'dofs': cons}
    stokes = solver.solve_linear(dofs, residual=ures + pres if self.single else [ures, pres], constrain=self.cons)
    self.arguments = dict(dofs=stokes) if self.single else stokes
    self.residual = ures + dres + pres if self.single else [ures + dres, pres]
    inertia = gauss.integral(.5 * (u**2).sum(-1) * dx).derivative('dofs')
    self.inertia = inertia if self.single else [inertia, None]
    self.tol = 1e-10
    self.dofs = dofs
    self.frozen = types.frozenarray if self.single else solver.argdict

  def assert_resnorm(self, lhs):
    res = self.residual.eval(arguments=dict(dofs=lhs))[numpy.isnan(self.cons)] if self.single \
      else numpy.concatenate([sparse.toarray(sparse.take(r, [numpy.isnan(self.cons[d])]) if d in self.cons else r)
        for d, r in zip(self.dofs, sample.eval_integrals_sparse(*self.residual, **lhs))])
    resnorm = numpy.linalg.norm(res)
    self.assertLess(resnorm, self.tol)

  def test_direct(self):
    with self.assertRaises(solver.SolverError):
      solver.solve_linear(self.dofs, residual=self.residual, constrain=self.cons)

  def test_newton(self):
    self.assert_resnorm(solver.newton(self.dofs, residual=self.residual, arguments=self.arguments, constrain=self.cons).solve(tol=self.tol, maxiter=2))

  def test_newton_medianbased(self):
    self.assert_resnorm(solver.newton(self.dofs, residual=self.residual, arguments=self.arguments, constrain=self.cons, linesearch=solver.MedianBased()).solve(tol=self.tol, maxiter=2))

  def test_newton_relax0(self):
    self.assert_resnorm(solver.newton(self.dofs, residual=self.residual, arguments=self.arguments, constrain=self.cons, relax0=.1).solve(tol=self.tol, maxiter=5))

  def test_newton_tolnotreached(self):
    with self.assertLogs('nutils', logging.WARNING) as cm:
      self.assert_resnorm(solver.newton(self.dofs, residual=self.residual, arguments=self.arguments, constrain=self.cons, linrtol=1e-99).solve(tol=self.tol, maxiter=2))
    for msg in cm.output:
      self.assertIn('solver failed to reach tolerance', msg)

  def test_newton_iter(self):
    _test_recursion_cache(self, lambda: ((self.frozen(lhs), info.resnorm) for lhs, info in solver.newton(self.dofs, residual=self.residual, constrain=self.cons)))

  def test_pseudotime(self):
    self.assert_resnorm(solver.pseudotime(self.dofs, residual=self.residual, arguments=self.arguments, constrain=self.cons, inertia=self.inertia, timestep=1).solve(tol=self.tol, maxiter=12))

  def test_pseudotime_iter(self):
    _test_recursion_cache(self, lambda: ((self.frozen(lhs), info.resnorm) for lhs, info in solver.pseudotime(self.dofs, residual=self.residual, arguments=self.arguments, constrain=self.cons, inertia=self.inertia, timestep=1)))

navierstokes(single=False)
navierstokes(single=True)


@parametrize
class finitestrain(TestCase):

  def setUp(self):
    super().setUp()
    domain, geom = mesh.rectilinear([numpy.linspace(0,1,9)] * 2)
    ubasis = domain.basis('std', degree=2)
    if self.vector:
      u = ubasis.vector(2).dot(function.Argument('dofs', [len(ubasis)*2]))
    else:
      u = (ubasis[:,numpy.newaxis] * function.Argument('dofs', [len(ubasis), 2])).sum(0)
    Geom = geom * [1.1, 1] + u
    self.cons = solver.optimize('dofs', domain.boundary['left,right'].integral((u**2).sum(0), degree=4), droptol=1e-15)
    self.boolcons = ~numpy.isnan(self.cons)
    strain = .5 * (function.outer(Geom.grad(geom), axis=1).sum(0) - function.eye(2))
    self.energy = domain.integral(((strain**2).sum([0,1]) + 20*(function.determinant(Geom.grad(geom))-1)**2)*function.J(geom), degree=6)
    self.residual = self.energy.derivative('dofs')
    self.tol = 1e-10

  def assert_resnorm(self, lhs):
    res, = sample.eval_integrals_sparse(self.residual, dofs=lhs)
    resnorm = numpy.linalg.norm(sparse.toarray(res)[~self.boolcons])
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

finitestrain(vector=False)
finitestrain(vector=True)


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

  def test_unknowntarget(self):
    err = self.domain.integral('(sqrt(1 - u) - .5)^2' @ self.ns, degree=2)
    with self.assertRaises(ValueError):
      dofs = solver.optimize(['dofs', 'other'], err, tol=1e-10)
    dofs = solver.optimize(['dofs', 'other'], err, tol=1e-10, droptol=1e-10)
    self.assertEqual(list(dofs), ['dofs'])
    dofs = solver.optimize(['other'], err, tol=1e-10, droptol=1e-10)
    self.assertEqual(dofs, {})
    with self.assertRaises(KeyError):
      dofs = solver.optimize('other', err, tol=1e-10, droptol=1e-10)


class burgers(TestCase):

  def setUp(self):
    super().setUp()
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
    with self.assertWarns(warnings.NutilsDeprecationWarning):
      self.check(solver.cranknicolson, theta=0.5)


class rungekutta_time(TestCase):

  def check(self, method, f):
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
        udesired += timestep * f(i, timestep)

  def test_cranknicolson(self):
    self.check(solver.gausslegendre2, lambda i, timestep: numpy.sin((i+.5)*timestep))


class rungekutta_res(TestCase):

  def check(self, method, f):
    ns = function.Namespace()
    topo, ns.x = mesh.rectilinear([1])
    ns.u_n = '?u_n + <0>_n'
    inertia = residual = topo.integral('u_n d:x' @ ns, degree=0)
    timestep = 0.1
    udesired = numpy.array([1.])
    uactualiter = iter(method(target='u', residual=residual, inertia=inertia, timestep=timestep, lhs0=udesired, timetarget='t'))
    for i in range(5):
      with self.subTest(i=i):
        uactual = next(uactualiter)
        self.assertAllAlmostEqual(uactual, udesired)
        udesired += timestep * f(udesired, timestep)

  def test_cranknicolson(self):
    self.check(solver.gausslegendre2, lambda u, timestep: -u / (1+timestep/2))
