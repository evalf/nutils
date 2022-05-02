#!/usr/bin/env python3
#
# In this script we solve the same infinite plane strain problem as in
# :ref:`examples/platewithhole.py`, but instead of using FCM to create the hole
# we use a NURBS-based mapping. A detailed description of the testcase can be
# found in Hughes et al., `Isogeometric analysis: CAD, finite elements, NURBS,
# exact geometry and mesh refinement`, Computer Methods in Applied Mechanics
# and Engineering, Elsevier, 2005, 194, 4135-4195.

from nutils import mesh, function, solver, export, cli, testing
from nutils.expression_v2 import Namespace
import numpy
import treelog


def main(nrefine: int, traction: float, radius: float, poisson: float):
    '''
    Horizontally loaded linear elastic plate with IGA hole.

    .. arguments::

       nrefine [2]
         Number of uniform refinements starting from 1x2 base mesh.
       traction [.1]
         Far field traction (relative to Young's modulus).
       radius [.5]
         Cut-out radius.
       poisson [.3]
         Poisson's ratio, nonnegative and strictly smaller than 1/2.
    '''

    # create the coarsest level parameter domain
    domain, geom0 = mesh.rectilinear([1, 2])
    bsplinebasis = domain.basis('spline', degree=2)
    controlweights = numpy.ones(12)
    controlweights[1:3] = .5 + .25 * numpy.sqrt(2)
    weightfunc = bsplinebasis.dot(controlweights)
    nurbsbasis = bsplinebasis * controlweights / weightfunc

    # create geometry function
    indices = [0, 2], [1, 2], [2, 1], [2, 0]
    controlpoints = numpy.concatenate([
        numpy.take([0, 2**.5-1, 1], indices) * radius,
        numpy.take([0, .3, 1], indices) * (radius+1) / 2,
        numpy.take([0, 1, 1], indices)])
    geom = (nurbsbasis[:, numpy.newaxis] * controlpoints).sum(0)

    radiuserr = domain.boundary['left'].integral((function.norm2(geom) - radius)**2 * function.J(geom0), degree=9).eval()**.5
    treelog.info('hole radius exact up to L2 error {:.2e}'.format(radiuserr))

    # refine domain
    if nrefine:
        domain = domain.refine(nrefine)
        bsplinebasis = domain.basis('spline', degree=2)
        controlweights = domain.project(weightfunc, onto=bsplinebasis, geometry=geom0, ischeme='gauss9')
        nurbsbasis = bsplinebasis * controlweights / weightfunc

    ns = Namespace()
    ns.δ = function.eye(domain.ndims)
    ns.X = geom
    ns.define_for('X', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.λ = 2 * poisson
    ns.μ = 1 - poisson
    ns.add_field(('u', 'v'), nurbsbasis, shape=[2])
    ns.x_i = 'X_i + u_i'
    ns.ε_ij = '(∇_j(u_i) + ∇_i(u_j)) / 2'
    ns.σ_ij = 'λ ε_kk δ_ij + 2 μ ε_ij'
    ns.r2 = 'X_k X_k'
    ns.R2 = radius**2 / ns.r2
    ns.k = (3-poisson) / (1+poisson)  # plane stress parameter
    ns.scale = traction * (1+poisson) / 2
    ns.uexact_i = 'scale (X_i ((k + 1) (0.5 + R2) + (1 - R2) R2 (X_0^2 - 3 X_1^2) / r2) - 2 δ_i1 X_1 (1 + (k - 1 + R2) R2))'
    ns.du_i = 'u_i - uexact_i'

    sqr = domain.boundary['top,bottom'].integral('(u_i n_i)^2 dS' @ ns, degree=9)
    cons = solver.optimize('u', sqr, droptol=1e-15)
    sqr = domain.boundary['right'].integral('du_k du_k dS' @ ns, degree=20)
    cons = solver.optimize('u', sqr, droptol=1e-15, constrain=cons)

    # construct residual
    res = domain.integral('∇_j(v_i) σ_ij dV' @ ns, degree=9)

    # solve system
    u = solver.solve_linear('u:v', res, constrain=cons)

    # vizualize result
    bezier = domain.sample('bezier', 9)
    x, σxx = bezier.eval(['x_i', 'σ_00'] @ ns, u=u)
    export.triplot('stressxx.png', x, σxx, tri=bezier.tri, hull=bezier.hull, clim=(numpy.nanmin(σxx), numpy.nanmax(σxx)))

    # evaluate error
    err = function.sqrt(domain.integral(['du_k du_k dV', '∇_j(du_i) ∇_j(du_i) dV'] @ ns, degree=9)).eval(u=u)
    treelog.user('errors: L2={:.2e}, H1={:.2e}'.format(*err))

    return err, cons, u

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to keep with the default arguments simply run :sh:`python3
# platewithhole-nurbs.py`.


if __name__ == '__main__':
    cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.


class test(testing.TestCase):

    def test0(self):
        err, cons, u = main(nrefine=0, traction=.1, radius=.5, poisson=.3)
        with self.subTest('l2-error'):
            self.assertAlmostEqual(err[0], .00199, places=5)
        with self.subTest('h1-error'):
            self.assertAlmostEqual(err[1], .02269, places=5)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons, '''
                eNpjYGBoQIIggMZXOKdmnHRe3vjh+cvGDAwA6w0LgQ==''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNpjYJh07qLhhnOTjb0vTDdmAAKVcy/1u85lGYforQDzFc6pGSedlzd+eP4ykA8AvkQRaA==''')

    def test2(self):
        err, cons, u = main(nrefine=2, traction=.1, radius=.5, poisson=.3)
        with self.subTest('l2-error'):
            self.assertAlmostEqual(err[0], .00009, places=5)
        with self.subTest('h1-error'):
            self.assertAlmostEqual(err[1], .00286, places=5)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons, '''
                eNpjYGBoIAKCwCBXp3kuysDjnLXR+3NPjTzPqxrnAnHeeQvjk+dTjZ9d2GG85soJYwYGAPkhPtE=''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNpjYOg890mv85yM4axz0kYHz+00Yj6vZJxzPtWY+0KPMffFucaml+caMwBB5LlCvYhzCw0qzu0wPHyu
                0sjlPIsx14VoY/6LvcaxlxYZz7myCKzO+dwWPZdzBwzqz20z/Hguxmj2+TtGHRdsjHdfbDB2v7zUeMXV
                pWB1VucC9B3OORmuOCdhZHR+ktGu87eNbC6oGstfLDA+eWm1seG19WB1Buf+6ruce2p469wco9Dzb4wm
                n2c23nZe3djqQqpx88XNxrOv7gOr0zwXZeBxztro/bmnRp7nVY1zgTjvvIXxSaBfnl3YYbzmygmgOgDU
                Imlr''')
