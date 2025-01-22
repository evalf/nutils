from nutils import mesh, function, export, testing
from nutils.solver import System
from nutils.expression_v2 import Namespace
import treelog as log
import numpy
import itertools


def main(nelems: int = 40,
         btype: str = 'discont',
         degree: int = 1,
         timescale: float = .5,
         newtontol: float = 1e-5,
         endtime: float = .5):

    '''Burgers' equation

    Solves Burgers' equation on a 1D periodic domain, starting from a centered
    Gaussian and convecting in the positive direction of the first coordinate.

    Parameters
    ----------
    nelems
        Number of elements along a single dimension.
    btype
        Type of basis function (discont/legendre).
    degree
        Polynomial degree for discontinuous basis functions.
    timescale
        Fraction of timestep and element size: timestep=timescale/nelems.
    newtontol
        Newton tolerance.
    endtime
        Stopping time.
    '''

    domain, geom = mesh.line(numpy.linspace(-.5, .5, nelems+1), periodic=True)
    timestep = timescale / nelems

    ns = Namespace()
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.u = domain.field('u', btype=btype, degree=degree)
    ns.du = ns.u - function.replace_arguments(ns.u, 'u:u0')
    ns.v = domain.field('v', btype=btype, degree=degree)
    ns.t = function.field('t')
    ns.dt = ns.t - function.field('t0')
    ns.f = '.5 u^2'
    ns.C = 1
    ns.uinit = 'exp(-25 x^2)'

    res = domain.integral('(v du / dt - ∇(v) f) dV' @ ns, degree=degree*2)
    res -= domain.interfaces.integral('[v] n ({f} - .5 C [u] n) dS' @ ns, degree=degree*2)

    sqr = domain.integral('(u - uinit)^2 dV' @ ns, degree=max(degree*2, 5))
    args = System(sqr, trial='u').solve()
    args['t'] = 0.

    system = System(res, trial='u', test='v')

    bezier = domain.sample('bezier', 7)
    with log.iter.plain('timestep', itertools.count()) as steps:
        for _ in steps:
            log.info('time:', round(args['t'], 10))
            x, u = bezier.eval(['x', 'u'] @ ns, **args)
            export.triplot('solution.png', x[:,numpy.newaxis], u, tri=bezier.tri, hull=bezier.hull, clim=(0, 1))
            if args['t'] >= endtime:
                break
            args = system.step(timestep=timestep, arguments=args, timearg='t', suffix='0', tol=newtontol)

    return args


class test(testing.TestCase):

    def test_1d_p0(self):
        args = main(nelems=10, timescale=.1, degree=0, endtime=.01)
        self.assertAlmostEqual64(args['u'], '''
            eNrz1ttqGGOiZSZlrmbuZdZgcsEwUg8AOqwFug==''')

    def test_1d_p1(self):
        args = main(nelems=10, timescale=.1, degree=1, endtime=.01)
        self.assertAlmostEqual64(args['u'], '''
            eNrbocann6u3yqjTyMLUwfSw2TWzKPNM8+9mH8wyTMNNZxptMirW49ffpwYAI6cOVA==''')

    def test_1d_p2(self):
        args = main(nelems=10, timescale=.1, degree=2, endtime=.01)
        self.assertAlmostEqual64(args['u'], '''
            eNrr0c7SrtWfrD/d4JHRE6Ofxj6mnqaKZofNDpjZmQeYB5pHmL8we23mb5ZvWmjKY/LV6KPRFIMZ+o36
            8dp92gCxZxZG''')

    def test_1d_p1_legendre(self):
        args = main(nelems=10, timescale=.1, btype='legendre', degree=1, endtime=.01)
        self.assertAlmostEqual64(args['u'], '''
            eNrbpbtGt9VQyNDfxMdYzczERNZczdjYnOdsoNmc01kmE870Gj49t0c36BIAAhsO1g==''')

    def test_1d_p2_legendre(self):
        args = main(nelems=10, timescale=.1, btype='legendre', degree=2, endtime=.01)
        self.assertAlmostEqual64(args['u'], '''
            eNoBPADD/8ot2y2/K4UxITFFLk00RTNNLyY2KzTTKx43QjOOzzM3Ss0pz1A2qsvhKGk0jsyXL48xzc5j
            LswtIdLIK5SlF78=''')


if __name__ == '__main__':
    from nutils import cli
    cli.run(main)


# example:tags=Burgers' equation,discontinuous Galerkin
