from nutils import solver, mesh, function, cache, types, evaluable, sparse
from nutils.expression_v2 import Namespace
from nutils.testing import TestCase, parametrize
import numpy
import contextlib
import tempfile
import logging


@contextlib.contextmanager
def tmpcache():
    with tempfile.TemporaryDirectory() as tmpdir:
        with cache.enable(tmpdir):
            yield


def _edit(v):
    return (numpy.array(v, copy=True) if isinstance(v, (numpy.ndarray, numpy.number, float, int))
        else tuple(map(_edit, v)) if isinstance(v, tuple)
        else {k: _edit(v) for (k, v) in v.items()} if isinstance(v, dict)
        else {k: _edit(v) for (k, v) in v.__dict__.items()} if isinstance(v, types.attributes)
        else v) # convert arrays to lists so we can use assertEqual


def _assert_almost_equal(testcase, a, b):
    testcase.assertIs(type(a), type(b))
    if isinstance(a, (tuple, list)):
        testcase.assertEqual(len(a), len(b))
        for ia, ib in zip(a, b):
            _assert_almost_equal(testcase, ia, ib)
    elif isinstance(a, dict) and all(isinstance(key, str) for key in a.keys()):
        testcase.assertEqual(frozenset(a.keys()), frozenset(b.keys()))
        for key in a.keys():
            _assert_almost_equal(testcase, a[key], b[key])
    elif isinstance(a, numpy.ndarray):
        testcase.assertAllAlmostEqual(a, b)
    else:
        testcase.fail(f'unsupported type: {type(a)}')


def _test_recursion_cache(testcase, solver_iter):
    read = lambda n: tuple(_edit(item) for i, item in zip(range(n), solver_iter()))
    reference = read(5)
    for lengths in [1, 2, 3], [1, 3, 2], [0, 3, 5]:
        with tmpcache():
            for i, length in enumerate(lengths):
                with testcase.subTest(lengths=lengths, step=i):
                    if length == 0: # treated as special case because assertLogs would fail
                        read(0)
                        continue
                    with testcase.assertLogs('nutils', 'DEBUG') as cm:
                        v = read(length)
                    _assert_almost_equal(testcase, v, reference[:length])
                    testcase.assertRegex('\n'.join(cm.output), '\\[cache\\.Recursion [0-9a-f]{40}\\] start iterating')
                    testcase.assertRegex('\n'.join(cm.output), '\\[cache\\.Recursion [0-9a-f]{40}\\.0000\\] load' if i and max(lengths[:i]) > 0
                                                          else '\\[cache\\.Recursion [0-9a-f]{40}\\.0000\\] cache exhausted')


def _test_solve_cache(testcase, solver_gen):
    _test_recursion_cache(testcase, solver_gen)
    with testcase.subTest('solve'), tmpcache():
        v1 = _edit(solver_gen().solve(1e-5))
        with testcase.assertLogs('nutils', 'DEBUG') as cm:
            v2, info = _edit(solver_gen().solve_withinfo(1e-5))
        _assert_almost_equal(testcase, v1, v2)
        testcase.assertRegex('\n'.join(cm.output), '\\[cache\\.function [0-9a-f]{40}\\] load')
        with testcase.assertLogs('nutils', 'DEBUG') as cm:
            solver_gen().solve(1e-6)
        testcase.assertRegex('\n'.join(cm.output), '\\[cache\\.function [0-9a-f]{40}\\] failed to load')


class laplace(TestCase):

    def setUp(self):
        super().setUp()
        domain, geom = mesh.rectilinear([8, 8])
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
                    lhs = solver.newton('dofs', residual=self.residual, constrain=self.cons).solve(tol=1e-10, maxiter=1)
                res = self.residual.eval(arguments=dict(dofs=lhs))
                resnorm = numpy.linalg.norm(res[~self.cons.where])
                self.assertLess(resnorm, 1e-13)


class laplace_field(TestCase):

    def setUp(self):
        super().setUp()
        domain, geom = mesh.rectilinear([8, 8])
        basis = domain.basis('std', degree=1)
        u = function.dotarg('u', basis)
        v = function.dotarg('v', basis)
        sqr = domain.boundary['left'].integral(u**2, degree=2)
        self.cons = solver.optimize('u,', sqr)
        self.residual = domain.integral((v.grad(geom) @ u.grad(geom))*function.J(geom), degree=2) \
            + domain.boundary['top'].integral(v*function.J(geom), degree=2)

    def test_res(self):
        args = solver.solve_linear('u:v', residual=self.residual, constrain=self.cons)
        res = self.residual.derivative('v').eval(**args)
        resnorm = numpy.linalg.norm(res[numpy.isnan(self.cons['u'])])
        self.assertLess(resnorm, 1e-13)


@parametrize
class navierstokes(TestCase):

    viscosity = 1e-3

    def setUp(self):
        super().setUp()
        domain, geom = mesh.rectilinear([numpy.linspace(0, 1, 9)] * 2)
        gauss = domain.sample('gauss', 5)
        uin = geom[1] * (1-geom[1])
        dx = function.J(geom)
        ubasis = domain.basis('std', degree=2)
        pbasis = domain.basis('std', degree=1)
        if self.single:
            ubasis, pbasis = function.chain([ubasis.vector(2), pbasis])
            dofs = function.Argument('dofs', [len(ubasis)])
            u = dofs @ ubasis
            p = dofs @ pbasis
            dofs = 'dofs'
            ures = gauss.integral((self.viscosity * (ubasis.grad(geom) * (u.grad(geom) + u.grad(geom).T)).sum([-1, -2]) - ubasis.div(geom) * p) * dx)
            dres = gauss.integral((ubasis * (u.grad(geom) * u).sum(-1)).sum(-1) * dx)
        else:
            u = (ubasis[:, numpy.newaxis] * function.Argument('dofs', [len(ubasis), 2])).sum(0)
            p = (pbasis * function.Argument('pdofs', [len(pbasis)])).sum(0)
            dofs = 'dofs', 'pdofs'
            ures = gauss.integral((self.viscosity * (ubasis[:, numpy.newaxis].grad(geom) * (u.grad(geom) + u.grad(geom).T)).sum(-1) - ubasis.grad(geom) * p) * dx)
            dres = gauss.integral(ubasis[:, numpy.newaxis] * (u.grad(geom) * u).sum(-1) * dx)
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

    def assert_resnorm(self, lhs):
        res = self.residual.eval(arguments=dict(dofs=lhs))[numpy.isnan(self.cons)] if self.single \
            else numpy.concatenate([sparse.toarray(sparse.take(r, [numpy.isnan(self.cons[d])]) if d in self.cons else r)
                                    for d, r in zip(self.dofs, evaluable.eval_sparse(self.residual, **lhs))])
        resnorm = numpy.linalg.norm(res)
        self.assertLess(resnorm, self.tol)

    def test_direct(self):
        with self.assertRaises(solver.SolverError):
            solver.solve_linear(self.dofs, residual=self.residual, constrain=self.cons)

    def test_newton(self):
        self.assert_resnorm(solver.newton(self.dofs, residual=self.residual, arguments=self.arguments, constrain=self.cons).solve(tol=self.tol, maxiter=3))

    def test_newton_vanilla(self):
        self.assert_resnorm(solver.newton(self.dofs, residual=self.residual, arguments=self.arguments, constrain=self.cons, linesearch=None).solve(tol=self.tol, maxiter=3))

    def test_newton_medianbased(self):
        self.assert_resnorm(solver.newton(self.dofs, residual=self.residual, arguments=self.arguments, constrain=self.cons, linesearch=solver.MedianBased()).solve(tol=self.tol, maxiter=3))

    def test_newton_relax0(self):
        self.assert_resnorm(solver.newton(self.dofs, residual=self.residual, arguments=self.arguments, constrain=self.cons, relax0=.1).solve(tol=self.tol, maxiter=6))

    def test_newton_tolnotreached(self):
        with self.assertLogs('nutils', logging.WARNING) as cm:
            self.assert_resnorm(solver.newton(self.dofs, residual=self.residual, arguments=self.arguments, constrain=self.cons, linrtol=1e-99).solve(tol=self.tol, maxiter=3))
        for msg in cm.output:
            self.assertIn('solver failed to reach tolerance', msg)

    def test_newton_cache(self):
        _test_solve_cache(self, lambda: solver.newton(self.dofs, residual=self.residual, constrain=self.cons))

    def test_pseudotime(self):
        self.assert_resnorm(solver.pseudotime(self.dofs, residual=self.residual, arguments=self.arguments, constrain=self.cons, inertia=self.inertia, timestep=1).solve(tol=self.tol, maxiter=12))

    def test_pseudotime_cache(self):
        _test_solve_cache(self, lambda: solver.pseudotime(self.dofs, residual=self.residual, arguments=self.arguments, constrain=self.cons, inertia=self.inertia, timestep=1))


navierstokes(single=False)
navierstokes(single=True)


@parametrize
class finitestrain(TestCase):

    def setUp(self):
        super().setUp()
        domain, geom = mesh.rectilinear([numpy.linspace(0, 1, 9)] * 2)
        ubasis = domain.basis('std', degree=2)
        if self.vector:
            u = function.dotarg('dofs', ubasis.vector(2))
        else:
            u = function.dotarg('dofs', ubasis, shape=(2,))
        Geom = geom * [1.1, 1] + u
        self.cons = solver.optimize('dofs', domain.boundary['left,right'].integral((u**2).sum(0), degree=4), droptol=1e-15)
        self.boolcons = ~numpy.isnan(self.cons)
        strain = .5 * (function.outer(Geom.grad(geom), axis=1).sum(0) - function.eye(2))
        self.energy = domain.integral(((strain**2).sum([0, 1]) + 20*(numpy.linalg.det(Geom.grad(geom))-1)**2)*function.J(geom), degree=6)
        self.residual = self.energy.derivative('dofs')
        self.tol = 1e-10

    def assert_resnorm(self, lhs):
        res = evaluable.eval_sparse(self.residual, dofs=lhs)
        resnorm = numpy.linalg.norm(sparse.toarray(res)[~self.boolcons])
        self.assertLess(resnorm, self.tol)

    def test_direct(self):
        with self.assertRaises(solver.SolverError):
            self.assert_resnorm(solver.solve_linear('dofs', residual=self.residual, constrain=self.cons))

    def test_newton(self):
        self.assert_resnorm(solver.newton('dofs', residual=self.residual, constrain=self.cons).solve(tol=self.tol, maxiter=7))

    def test_newton_vanilla(self):
        self.assert_resnorm(solver.newton('dofs', residual=self.residual, constrain=self.cons, linesearch=None).solve(tol=self.tol, maxiter=7))

    def test_newton_boolcons(self):
        self.assert_resnorm(solver.newton('dofs', residual=self.residual, constrain=self.boolcons).solve(tol=self.tol, maxiter=7))

    def test_newton_cache(self):
        _test_solve_cache(self, lambda: solver.newton('dofs', residual=self.residual, constrain=self.cons))

    def test_minimize(self):
        self.assert_resnorm(solver.minimize('dofs', energy=self.energy, constrain=self.cons).solve(tol=self.tol, maxiter=13))

    def test_minimize_boolcons(self):
        self.assert_resnorm(solver.minimize('dofs', energy=self.energy, constrain=self.boolcons).solve(tol=self.tol, maxiter=13))

    def test_minimize_cache(self):
        _test_solve_cache(self, lambda: solver.minimize('dofs', energy=self.energy, constrain=self.cons))


finitestrain(vector=False)
finitestrain(vector=True)


class optimize(TestCase):

    def setUp(self):
        super().setUp()
        self.domain, geom = mesh.rectilinear([2, 2])
        self.ubasis = self.domain.basis('std', degree=1)
        self.ns = Namespace()
        self.ns.u = function.dotarg('dofs', self.ubasis)

    def test_linear(self):
        err = self.domain.boundary['bottom'].integral('(u - 1)^2' @ self.ns, degree=2)
        cons = solver.optimize('dofs', err, droptol=1e-15)
        numpy.testing.assert_almost_equal(cons, numpy.take([1, numpy.nan], [0, 1, 1, 0, 1, 1, 0, 1, 1]), decimal=15)

    def test_nonlinear(self):
        err = self.domain.boundary['bottom'].integral('(u + .25 u^3 - 1.25)^2' @ self.ns, degree=6)
        cons = solver.optimize('dofs', err, droptol=1e-15, tol=1e-15)
        numpy.testing.assert_almost_equal(cons, numpy.take([1, numpy.nan], [0, 1, 1, 0, 1, 1, 0, 1, 1]), decimal=15)

    def test_nonlinear_multipleroots(self):
        err = self.domain.boundary['bottom'].integral('(u + u^2 - .75)^2' @ self.ns, degree=2)
        cons = solver.optimize('dofs', err, droptol=1e-15, lhs0=numpy.ones(len(self.ubasis)), tol=1e-10)
        numpy.testing.assert_almost_equal(cons, numpy.take([.5, numpy.nan], [0, 1, 1, 0, 1, 1, 0, 1, 1]), decimal=15)

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
        domain, geom = mesh.rectilinear([10], periodic=(0,))
        basis = domain.basis('discont', degree=1)
        ns = Namespace()
        ns.x = geom
        ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
        ns.add_field(('u', 'v'), basis)
        ns.f = '.5 u^2'
        self.residual = domain.integral('-∇_0(v) f dV' @ ns, degree=2)
        self.residual += domain.interfaces.integral('-[v] n_0 ({f} - .5 [u] n_0) dS' @ ns, degree=4)
        self.inertia = domain.integral('v u dV' @ ns, degree=5)
        self.lhs0 = numpy.sin(numpy.arange(len(basis)))  # "random" initial vector

    def test_iters(self):
        it = iter(solver.impliciteuler('u:v', residual=self.residual, inertia=self.inertia, arguments=dict(u=self.lhs0), timestep=100))  # involves 2-level timestep scaling
        assert numpy.equal(next(it)['u'], self.lhs0).all()
        self.assertAlmostEqual64(next(it)['u'], 'eNpzNBA1NjHuNHQ3FDsTfCbAuNz4nUGZgeyZiDOZxlONmQwU9W3OFJ/pNQAADZIOPA==')

    def test_resume(self):
        _test_recursion_cache(self, lambda: solver.impliciteuler('u:v', residual=self.residual, inertia=self.inertia, arguments=dict(u=self.lhs0), timestep=1))

    def test_resume_withscaling(self):
        _test_recursion_cache(self, lambda: solver.impliciteuler('u:v', residual=self.residual, inertia=self.inertia, arguments=dict(u=self.lhs0), timestep=100))


class theta_time(TestCase):

    def check(self, method, theta):
        ns = Namespace()
        topo, ns.x = mesh.rectilinear([1])
        ns.define_for('x', jacobians=('dV',))
        ns.u = function.Argument('u', shape=(1,))
        ns.t = function.Argument('t', shape=())
        ns.e = function.ones([1])
        inertia = topo.integral('u_n dV' @ ns, degree=0)
        residual = topo.integral('-e_n sin(t) dV' @ ns, degree=0)
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


class SparsePoly(TestCase):
    '''
    f = A + B x + C y + D x2 + E x y + F y2 + G x3 + H x2 y + I x y2 + J y3
    dfdx = B + 2 D x + E y + 3 G x2 + 2 H x y + I y2
    d2fdxy = E + 2 H x + 2 I y

    poly = 1: A, x: B, y: C, x2: D, xy: E, y2: F, x3: G, x2y: H, xy2: I, y3: J
    '''

    def test_eval_x(self):
        x = function.dotarg('x')
        poly = solver.SparsePoly.factor(10 + 2 * x - x**2 + 5 * x**3, maxdegree=3)
        for x in numpy.array([0, 1, 2, -3]):
            self.assertEqual(poly.eval(dict(x=x)),
                10 + 2 * x - x**2 + 5 * x**3)

    def test_eval_xy(self):
        x = function.dotarg('x')
        y = function.dotarg('y')
        poly = solver.SparsePoly.factor(10 + 2 * x * y**2 - x**2 * y**3 + 5 * x**3 * y, maxdegree=5)
        for x in numpy.array([0, 1, 2, -3]):
            for y in numpy.array([0, 1, 2, -3]):
                self.assertEqual(poly.eval(dict(x=x, y=y)),
                    10 + 2 * x * y**2 - x**2 * y**3 + 5 * x**3 * y)

    def test_deriv_x(self):
        x = function.dotarg('x')
        poly = solver.SparsePoly.factor(10 + 2 * x - x**2 + 5 * x**3, maxdegree=3)
        for x in numpy.array([0, 1, 2, -3]):
            self.assertEqual(poly.derivative('x').eval(dict(x=x)),
                2 - 2 * x + 15 * x**2)

    def test_deriv_xx(self):
        x = function.dotarg('x')
        poly = solver.SparsePoly.factor(10 + 2 * x - x**2 + 5 * x**3, maxdegree=3)
        for x in numpy.array([0, 1, 2, -3]):
            self.assertEqual(poly.derivative('x').derivative('x').eval(dict(x=x)),
                -2 + 30 * x)

    def test_deriv_xy(self):
        x = function.dotarg('x')
        y = function.dotarg('y')
        poly = solver.SparsePoly.factor(x * y**2, maxdegree=5)
        for x in numpy.array([0, 1, 2, -3]):
            for y in numpy.array([0, 1, 2, -3]):
                self.assertEqual(poly.derivative('x').derivative('y').eval(dict(x=x, y=y)),
                    2 * y)

    def test_simple(self):
        x = function.dotarg('x')
        testx = function.dotarg('testx')
        poly = solver.SparsePoly.factor(testx * (10 + 2 * x), maxdegree=2)
        args = poly.solve('x:testx') # 2 * x == -10
        self.assertEqual(args['x'], -5)

    def test_coupled(self):
        basis = numpy.array([1.])
        x = function.dotarg('x', basis)
        testx = function.dotarg('testx', basis)
        y = function.dotarg('y', basis)
        testy = function.dotarg('testy', basis)
        poly = solver.SparsePoly.factor(testx * (10 + 2 * x + y) + testy * (x + y), maxdegree=2)
        args = poly.solve('x:testx,y:testy') # 2 * x + y == -10, x + y = 0
        self.assertEqual(args['x'].tolist(), [-10])
        self.assertEqual(args['y'].tolist(), [10])

    def test_args(self):
        x = function.dotarg('x')
        testx = function.dotarg('testx')
        y = function.dotarg('y')
        z = function.dotarg('z')
        poly = solver.SparsePoly.factor(testx * (10 + 2 * x - y - z), maxdegree=2)
        args = poly.solve('x:testx', dict(y=numpy.array(2), z=numpy.array(4))) # 2 * x == y + z - 10
        self.assertEqual(args['x'], -2)

    def test_higher_order(self):
        x = function.dotarg('x')
        testx = function.dotarg('testx')
        y = function.dotarg('y')
        z = function.dotarg('z')
        poly = solver.SparsePoly.factor(testx * (10 + 2 * x - 2 * y**2 + z**3 + y * z - y**2 * z**2), maxdegree=5)
        args = poly.solve('x:testx', dict(y=numpy.array(3), z=numpy.array(2))) # 2 * x == 2 y**2 - z**3 - y * z + y**2 * z**2 - 10
        self.assertEqual(args['x'], 15)

    def test_nonlinear(self):
        x = function.dotarg('x')
        testx = function.dotarg('testx')
        poly = solver.SparsePoly.factor(testx * (2 * x + x**2 - 8), maxdegree=3)
        args = poly.solve('x:testx', tol=1e-5) # 2 * x + x**2 == 8
        self.assertLess(abs(args['x'] - 2), 1e-5)

    def test_system(self):
        topo, x = mesh.rectilinear([3, 3])
        basis = topo.basis('std', 1)
        u = function.dotarg('u', basis)
        v = function.dotarg('v', basis)
        res = topo.integral((function.grad(v, x) @ function.grad(u, x) + v) * function.J(x), degree=2)
        cons = solver.optimize('u,', topo.boundary['right'].integral(u**2, degree=2), droptol=1e-10)
        lhs_solver = solver.solve_linear('u:v', res, constrain=cons)
        poly = solver.SparsePoly.factor(res, maxdegree=2)
        lhs_poly = poly.solve('u:v', constrain=cons)
        self.assertLess(numpy.linalg.norm(lhs_poly['u'] - lhs_solver['u']), 1e-14)
