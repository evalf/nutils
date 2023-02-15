# Deformation of an infinite plate with a circular hole
#
# In this script we solve the linear plane strain elasticity problem for an
# infinite plate with a circular hole under tension. We do this by placing the
# circle in the origin of a unit square, imposing symmetry conditions on the
# left and bottom (named `sym`), and Dirichlet conditions constraining the
# displacements to the analytical solution to the right and top (named `far`).
#
# The script can be run in two modes: by specifying `mode=FCM`, the circular
# hole is cut out of a regular finite element mesh by means of the Finite Cell
# Method; by specifying `mode=NURBS` a Non-Uniform Rational BSpline geometry is
# created to map a regular domain onto the desired shape. Either mode supports
# sub-parameters which can be specified from the command-line by attaching them
# in curly braces (e.g. `FCM{nelems=20,degree=1}`).

from nutils import mesh, function, solver, export, cli, testing
from nutils.expression_v2 import Namespace
from dataclasses import dataclass
from typing import Union
import treelog as log
import numpy


@dataclass
class FCM:
    nelems: int = 9
    etype: str = 'square'
    btype: str = 'std'
    degree: int = 2
    maxrefine: int = 2
    def generate(self, radius):
        topo0, geom = mesh.unitsquare(self.nelems, self.etype)
        topo = topo0.trim(function.norm2(geom) - radius, maxrefine=self.maxrefine, name='hole')
        basis = topo.basis(self.btype, degree=self.degree)
        return topo.withboundary(sym='left,bottom', far='top,right'), geom, basis, self.degree


@dataclass
class NURBS:
    nrefine: int = 2
    def generate(self, radius):
        topo, geom0 = mesh.rectilinear([1, 2])
        bsplinebasis = topo.basis('spline', degree=2)
        controlweights = numpy.ones(12)
        controlweights[1:3] = .5 + .25 * numpy.sqrt(2)
        weightfunc = bsplinebasis @ controlweights
        nurbsbasis = bsplinebasis * controlweights / weightfunc
        # create geometry function
        A = 0, 0, 0
        B = (2**.5-1) * radius, .3 * (radius+1) / 2, 1
        C = radius, (radius+1) / 2, 1
        controlpoints = numpy.array([[A, B, C, C], [C, C, B, A]]).T.reshape(-1, 2)
        geom = nurbsbasis @ controlpoints
        # refine topology
        if self.nrefine:
            topo = topo.refine(self.nrefine)
            bsplinebasis = topo.basis('spline', degree=2)
            sqr = topo.integral((function.dotarg('w', bsplinebasis) - weightfunc)**2, degree=9)
            controlweights = solver.optimize('w', sqr)
            nurbsbasis = bsplinebasis * controlweights / weightfunc
        return topo.withboundary(hole='left', sym='top,bottom', far='right'), geom, nurbsbasis, 5


def main(mode: Union[FCM, NURBS], radius: float, traction: float, poisson: float):
    '''
    Horizontally loaded linear elastic plate with circular hole.

    .. arguments::

       mode [NURBS]
         Discretization strategy: FCM (Finite Cell Method) or NURBS.
       radius [.5]
         Cut-out radius.
       traction [.1]
         Far field traction (relative to Young's modulus).
       poisson [.3]
         Poisson's ratio, nonnegative and strictly smaller than 1/2.
    '''

    topo, geom, basis, degree = mode.generate(radius)

    ns = Namespace()
    ns.δ = function.eye(topo.ndims)
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.λ = 2 * poisson
    ns.μ = 1 - poisson
    ns.ubasis = basis.vector(2)
    ns.u = function.dotarg('u', ns.ubasis)
    ns.X_i = 'x_i + u_i'
    ns.ε_ij = '(∇_j(u_i) + ∇_i(u_j)) / 2'
    ns.σ_ij = 'λ ε_kk δ_ij + 2 μ ε_ij'
    ns.r2 = 'x_k x_k'
    ns.R2 = radius**2 / ns.r2
    ns.k = (3-poisson) / (1+poisson) # plane stress parameter
    ns.scale = traction * (1+poisson) / 2
    ns.uexact_i = 'scale (x_i ((k + 1) (.5 + R2) + (1 - R2) R2 (x_0^2 - 3 x_1^2) / r2) - 2 δ_i1 x_1 (1 + (k - 1 + R2) R2))'
    ns.du_i = 'u_i - uexact_i'
    ns.dr = numpy.sqrt(ns.r2) - radius

    radiuserr = topo.boundary['hole'].integrate('dr^2 dS' @ ns, degree=9)**.5
    log.info('hole radius exact up to L2 error {:.2e}'.format(radiuserr))

    sqr = topo.boundary['sym'].integral('(u_i n_i)^2 dS' @ ns, degree=degree*2)
    cons = solver.optimize(('u',), sqr, droptol=1e-15)

    sqr = topo.boundary['far'].integral('du_k du_k dS' @ ns, degree=20)
    cons = solver.optimize(('u',), sqr, droptol=1e-15, constrain=cons)

    res = topo.integral('∇_j(ubasis_ni) σ_ij dV' @ ns, degree=degree*2)
    args = solver.solve_linear(('u',), (res,), constrain=cons)

    bezier = topo.sample('bezier', 5)
    X, σxx = bezier.eval(['X_i', 'σ_00'] @ ns, **args)
    export.triplot('stressxx.png', X, σxx, tri=bezier.tri, hull=bezier.hull, clim=(numpy.nanmin(σxx), numpy.nanmax(σxx)), cmap='jet')

    err = numpy.sqrt(topo.integrate(['du_k du_k dV', '∇_j(du_i) ∇_j(du_i) dV'] @ ns, degree=max(degree, 3)*2, arguments=args))
    log.user('errors: L2={:.2e}, H1={:.2e}'.format(*err))

    return err, cons, args

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to keep with the default arguments simply run :sh:`python3
# platewithhole.py`. To select mixed elements and quadratic basis functions add
# :sh:`python3 platewithhole.py etype=mixed degree=2`.


if __name__ == '__main__':
    cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.


class test(testing.TestCase):

    def test_spline(self):
        err, cons, args = main(mode=FCM(nelems=4, etype='square', btype='spline', degree=2, maxrefine=2), traction=.1, radius=.5, poisson=.3)
        with self.subTest('l2-error'):
            self.assertAlmostEqual(err[0], .00033, places=5)
        with self.subTest('h1-error'):
            self.assertAlmostEqual(err[1], .00672, places=5)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjaGBoYGBAxvrnGBow4X89g3NQFSjQwLAGq7i10Wus4k+NfM8fNWZgOGL89upc47WX0ozvXjAzPn1e
                1TjnPACrACoJ''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(args['u'], '''
                eNpbZHbajIHhxzkGBhMgtgdi/XPypyRPvjFxO/PccPq5Vn2vcxr6luf+6xmcm2LMwLDQePf5c0bTzx8x
                5D7vaTjnnIFhzbmlQPH5xhV39Y3vXlxtJHoh2EjvvLXR63MbgOIbjRdfrTXeecnUeO+Fn0Yrzj818j1/
                FCh+xPjt1bnGay+lGd+9YGZ8+ryqcc55AK+AP/0=''')

    def test_mixed(self):
        err, cons, args = main(mode=FCM(nelems=4, etype='mixed', btype='std', degree=2, maxrefine=2), traction=.1, radius=.5, poisson=.3)
        with self.subTest('l2-error'):
            self.assertAlmostEqual(err[0], .00024, places=5)
        with self.subTest('h1-error'):
            self.assertAlmostEqual(err[1], .00739, places=5)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjaGDADhlwiOEU1z8HZusbgukkg5BzRJqKFRoa1oD1HzfceA5NH9FmgKC10SuwOdONpM7DxDYa77gM
                MueoMQPDEePzV2Hic42XXmoynnQRxvc3dryQbnz3Aoj91Mj3vJnx6fOqxjnnAQzkV94=''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(args['u'], '''
                eNoNzE8og3EcBvC3uUo5rNUOnBSK9/19n0Ic0Eo5oJBmRxcaB04kUnPgoETmT2w7LVrtMBy4auMw+35/
                7/vaykFSFEopKTnIe/jU01PPU6FNWcQIn+Or5CBfSqCGD1uDYhi7/KbW+dma5aK65gX6Y8Po8HSzZQ7y
                vBniHyvFV9aq17V7TK42O9kwFS9YUzxhjXIcZxLCnIzjTsfxah/BMFJotjUlZYz6xYeoPqEPKaigbKhb
                9lOj9NGa9KgtVmqJH9UT36gcp71dEr6HaVS5GS8f46AcQ9itx739SQXdBL8dRqeTo1odox35poh2yJVh
                apEueucsRWWPgpJFoLKPNzeHC/fU+yl48pDyMi6dCFbsBNJODNu2iawOoE4PoVdP4kH/UkZeaEDaUJQG
                zMg/DouRUg==''')

    def test_nurbs0(self):
        err, cons, args = main(mode=NURBS(nrefine=0), traction=.1, radius=.5, poisson=.3)
        with self.subTest('l2-error'):
            self.assertAlmostEqual(err[0], .00200, places=5)
        with self.subTest('h1-error'):
            self.assertAlmostEqual(err[1], .02271, places=5)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjYGBoQIIggMZXOKdmnHRe3vjh+cvGDAwA6w0LgQ==''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(args['u'], '''
                eNpjYJh07qLhhnOTjb0vTDdmAAKVcy/1u85lGYforQDzFc6pGSedlzd+eP4ykA8AvkQRaA==''')

    def test_nurbs2(self):
        err, cons, args = main(mode=NURBS(nrefine=2), traction=.1, radius=.5, poisson=.3)
        with self.subTest('l2-error'):
            self.assertAlmostEqual(err[0], .00009, places=5)
        with self.subTest('h1-error'):
            self.assertAlmostEqual(err[1], .00286, places=5)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjYGBoIAKCwCBXp3kuysDjnLXR+3NPjTzPqxrnAnHeeQvjk+dTjZ9d2GG85soJYwYGAPkhPtE=''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(args['u'], '''
                eNpjYOg890mv85yM4axz0kYHz+00Yj6vZJxzPtWY+0KPMffFucaml+caMwBB5LlCvYhzCw0qzu0wPHyu
                0sjlPIsx14VoY/6LvcaxlxYZz7myCKzO+dwWPZdzBwzqz20z/Hguxmj2+TtGHRdsjHdfbDB2v7zUeMXV
                pWB1VucC9B3OORmuOCdhZHR+ktGu87eNbC6oGstfLDA+eWm1seG19WB1Buf+6ruce2p469wco9Dzb4wm
                n2c23nZe3djqQqpx88XNxrOv7gOr0zwXZeBxztro/bmnRp7nVY1zgTjvvIXxSaBfnl3YYbzmygmgOgDU
                Imlr''')

# example:tags=elasticity,FCM,NURBS
