#! /usr/bin/env python3
#
# In this script we solve the linear plane strain elasticity problem for an
# infinite plate with a circular hole under tension. We do this by placing the
# circle in the origin of a unit square, imposing symmetry conditions on the
# left and bottom, and Dirichlet conditions constraining the displacements to
# the analytical solution to the right and top. The traction-free circle is
# removed by means of the Finite Cell Method (FCM).

from nutils import mesh, function, solver, export, cli, testing
from nutils.expression_v2 import Namespace
import numpy
import treelog


def main(nelems: int, etype: str, btype: str, degree: int, traction: float, maxrefine: int, radius: float, poisson: float):
    '''
    Horizontally loaded linear elastic plate with FCM hole.

    .. arguments::

       nelems [9]
         Number of elements along edge.
       etype [square]
         Type of elements (square/triangle/mixed).
       btype [std]
         Type of basis function (std/spline), with availability depending on the
         selected element type.
       degree [2]
         Polynomial degree.
       traction [.1]
         Far field traction (relative to Young's modulus).
       maxrefine [2]
         Number or refinement levels used for the finite cell method.
       radius [.5]
         Cut-out radius.
       poisson [.3]
         Poisson's ratio, nonnegative and strictly smaller than 1/2.
    '''

    domain0, geom = mesh.unitsquare(nelems, etype)
    domain = domain0.trim(function.norm2(geom) - radius, maxrefine=maxrefine)

    ns = Namespace()
    ns.δ = function.eye(domain.ndims)
    ns.X = geom
    ns.define_for('X', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.λ = 2 * poisson
    ns.μ = 1 - poisson
    ns.add_field(('u', 'v'), domain.basis(btype, degree=degree), shape=[2])
    ns.x_i = 'X_i + u_i'
    ns.ε_ij = '(∇_j(u_i) + ∇_i(u_j)) / 2'
    ns.σ_ij = 'λ ε_kk δ_ij + 2 μ ε_ij'
    ns.r2 = 'X_k X_k'
    ns.R2 = radius**2 / ns.r2
    ns.k = (3-poisson) / (1+poisson)  # plane stress parameter
    ns.scale = traction * (1+poisson) / 2
    ns.uexact_i = 'scale (X_i ((k + 1) (0.5 + R2) + (1 - R2) R2 (X_0^2 - 3 X_1^2) / r2) - 2 δ_i1 X_1 (1 + (k - 1 + R2) R2))'
    ns.du_i = 'u_i - uexact_i'

    sqr = domain.boundary['left,bottom'].integral('(u_i n_i)^2 dS' @ ns, degree=degree*2)
    cons = solver.optimize('u,', sqr, droptol=1e-15)
    sqr = domain.boundary['top,right'].integral('du_k du_k dS' @ ns, degree=20)
    cons = solver.optimize('u,', sqr, droptol=1e-15, constrain=cons)

    res = domain.integral('∇_j(v_i) σ_ij dV' @ ns, degree=degree*2)
    args = solver.solve_linear('u:v', res, constrain=cons)

    bezier = domain.sample('bezier', 5)
    x, σxx = bezier.eval(['x_i', 'σ_00'] @ ns, **args)
    export.triplot('stressxx.png', x, σxx, tri=bezier.tri, hull=bezier.hull)

    err = domain.integral(numpy.stack(['du_k du_k dV', '∇_j(du_i) ∇_j(du_i) dV'] @ ns), degree=max(degree, 3)*2).eval(**args)**.5
    treelog.user('errors: L2={:.2e}, H1={:.2e}'.format(*err))

    return err, cons['u'], args['u']

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
        err, cons, u = main(nelems=4, etype='square', btype='spline', degree=2, traction=.1, maxrefine=2, radius=.5, poisson=.3)
        with self.subTest('l2-error'):
            self.assertAlmostEqual(err[0], .00033, places=5)
        with self.subTest('h1-error'):
            self.assertAlmostEqual(err[1], .00674, places=5)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons, '''
                eNpjaGBoYGBAxvrnGBow4X89g3NQFSjQwLAGq7i10Wus4k+NfM8fNWZgOGL89upc47WX0ozvXjAzPn1e
                1TjnPACrACoJ''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNpbb3bMjIHhxzkGBhMgtgdi/XMqp8RPvjLxOPPCcNq5Fn3Pcxr6luf+6xmcm2LMwLDQePf5c0bTzx8x
                5DnvaTjnnIFhzbmlQPH5xgvu6hvfvbjaSPRCsJHeeWuj1+c2AMU3Gi++Wmu885Kp8d4LP41WnH9q5Hv+
                KFD8iPHbq3ON115KM757wcz49HlV45zzAL8gQC8=''')

    def test_mixed(self):
        err, cons, u = main(nelems=4, etype='mixed', btype='std', degree=2, traction=.1, maxrefine=2, radius=.5, poisson=.3)
        with self.subTest('l2-error'):
            self.assertAlmostEqual(err[0], .00024, places=5)
        with self.subTest('h1-error'):
            self.assertAlmostEqual(err[1], .00740, places=5)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons, '''
                eNpjaGDADhlwiOEU1z8HZusbgukkg5BzRJqKFRoa1oD1HzfceA5NH9FmgKC10SuwOdONpM7DxDYa77gM
                MueoMQPDEePzV2Hic42XXmoynnQRxvc3dryQbnz3Aoj91Mj3vJnx6fOqxjnnAQzkV94=''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNoNzEEoQ3EcB/AXVymHtdqBkyLx3v/3LTQHtHJQKKHZ0YXMQS6sSM2BcrKMqbHTotUOw4GrthzWfr//
                e6+nHJYUyUopKSnlHT717Vvfr0cpSWCWr/FVs1GuZdHKmb6QGMYRN9Qev1irXFUVTtAfG8agb5gtc5LX
                zQj/WDm+s3b8bsBncosZZsNUvGEt8YI1w2lcSQRrMg9Pp/FmZ2EYOfTYmnIyR+PShLi+oA8pq5DsqxoH
                qEvGaFdG1AErtclP6pnvVYnz/u4MVj2OZrfg53OceElE3Q482p9U0d0I2FGEnRK16SQdyjfFtEOuTFOv
                DFGDi7QsxxSSIoIPGby7Jdy4l/5PxVeGeFu4dWLYtk+Rd5JI2SaKOoh2PYVRvYi6/qWCvNKE9KMqnViR
                fyhZkYI=''')
