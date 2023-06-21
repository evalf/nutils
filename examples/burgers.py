# Burgers' equation
#
# In this script we solve Burgers' equation on a 1D periodic domain, starting
# from a centered Gaussian and convecting in the positive direction of the
# first coordinate.

from nutils import mesh, function, solver, export, cli, testing
from nutils.expression_v2 import Namespace
import treelog as log
import numpy
import itertools


def main(nelems: int, btype: str, degree: int, timescale: float, newtontol: float, endtime: float):
    '''
    Burgers equation on a 1D or 2D periodic domain.

    .. arguments::

       nelems [40]
         Number of elements along a single dimension.
       btype [discont]
         Type of basis function (discont/legendre).
       degree [1]
         Polynomial degree for discontinuous basis functions.
       timescale [.5]
         Fraction of timestep and element size: timestep=timescale/nelems.
       newtontol [1e-5]
         Newton tolerance.
       endtime [.5]
         Stopping time.
    '''

    domain, geom = mesh.line(numpy.linspace(-.5, .5, nelems+1), periodic=True)
    timestep = timescale / nelems

    ns = Namespace()
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.add_field(('u', 'u0', 'v'), domain.basis(btype, degree=degree))
    ns.dt = timestep
    ns.dudt = '(u - u0) / dt'
    ns.f = '.5 u^2'
    ns.C = 1
    ns.uinit = 'exp(-25 x^2)'

    res = domain.integral('(v dudt - ∇(v) f) dV' @ ns, degree=degree*2)
    res -= domain.interfaces.integral('[v] n ({f} - .5 C [u] n) dS' @ ns, degree=degree*2)

    sqr = domain.integral('(u - uinit)^2 dV' @ ns, degree=max(degree*2, 5))
    args = solver.optimize('u,', sqr)

    bezier = domain.sample('bezier', 7)
    with log.iter.plain('timestep', itertools.count(step=timestep)) as times:
        for t in times:
            log.info('time:', round(t, 10))
            x, u = bezier.eval(['x', 'u'] @ ns, **args)
            export.triplot('solution.png', x[:,numpy.newaxis], u, tri=bezier.tri, hull=bezier.hull, clim=(0, 1))
            if t >= endtime:
                break
            args['u0'] = args['u']
            args = solver.newton('u:v', res, arguments=args).solve(newtontol)

    return args

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to simulate until 0.5 seconds run :sh:`python3 burgers.py
# endtime=0.5`.


if __name__ == '__main__':
    cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.


class test(testing.TestCase):

    def test_1d_p0(self):
        args = main(nelems=10, timescale=.1, btype='discont', degree=0, endtime=.01, newtontol=1e-5)
        self.assertAlmostEqual64(args['u'], '''
            eNrz1ttqGGOiZSZlrmbuZdZgcsEwUg8AOqwFug==''')

    def test_1d_p1(self):
        args = main(nelems=10, timescale=.1, btype='discont', degree=1, endtime=.01, newtontol=1e-5)
        self.assertAlmostEqual64(args['u'], '''
            eNrbocann6u3yqjTyMLUwfSw2TWzKPNM8+9mH8wyTMNNZxptMirW49ffpwYAI6cOVA==''')

    def test_1d_p2(self):
        args = main(nelems=10, timescale=.1, btype='discont', degree=2, endtime=.01, newtontol=1e-5)
        self.assertAlmostEqual64(args['u'], '''
            eNrr0c7SrtWfrD/d4JHRE6Ofxj6mnqaKZofNDpjZmQeYB5pHmL8we23mb5ZvWmjKY/LV6KPRFIMZ+o36
            8dp92gCxZxZG''')

    def test_1d_p1_legendre(self):
        args = main(nelems=10, timescale=.1, btype='legendre', degree=1, endtime=.01, newtontol=1e-5)
        self.assertAlmostEqual64(args['u'], '''
            eNrbpbtGt9VQyNDfxMdYzczERNZczdjYnOdsoNmc01kmE870Gj49t0c36BIAAhsO1g==''')

    def test_1d_p2_legendre(self):
        args = main(nelems=10, timescale=.1, btype='legendre', degree=2, endtime=.01, newtontol=1e-5)
        self.assertAlmostEqual64(args['u'], '''
            eNoBPADD/8ot2y2/K4UxITFFLk00RTNNLyY2KzTTKx43QjOOzzM3Ss0pz1A2qsvhKGk0jsyXL48xzc5j
            LswtIdLIK5SlF78=''')

# example:tags=Burgers' equation,discontinuous Galerkin
