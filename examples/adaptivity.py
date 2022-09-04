#! /usr/bin/env python3
#
# In this script we solve the Laplace problem on a unit square that has the
# bottom-right quadrant removed (a.k.a. an L-shaped domain) with Dirichlet
# boundary conditions matching the harmonic function
#
#     u = (x^2+y^2)^(1/3) cos(arctan2(y+x,y-x)*2/3)
#
# shifted by 0.5 such that the origin coincides with the middle of the unit
# square. Note that the function evaluates to zero over the two boundaries
# bordering the removed quadrant.
#
# This benchmark problem is known to converge suboptimally under uniform
# refinement due to a singular gradient in the reentrant corner. This script
# demonstrates that optimal convergence can be restored by using adaptive
# refinement.

from nutils import mesh, function, solver, util, export, cli, testing
from nutils.expression_v2 import Namespace
import numpy
import treelog


def main(etype: str, btype: str, degree: int, nrefine: int):
    '''
    Adaptively refined Laplace problem on an L-shaped domain.

    .. arguments::

       etype [square]
         Type of elements (square/triangle/mixed).
       btype [h-std]
         Type of basis function (h/th-std/spline), with availability depending on
         the configured element type.
       degree [2]
         Polynomial degree
       nrefine [5]
         Number of refinement steps to perform.
    '''

    domain, geom = mesh.unitsquare(2, etype)
    geom -= .5 # shift domain center to origin

    x, y = geom
    exact = (x**2 + y**2)**(1/3) * numpy.cos(numpy.arctan2(y+x, y-x) * (2/3))
    selection = domain.select(exact, ischeme='gauss1')
    domain = domain.subset(selection, newboundary='corner')
    linreg = util.linear_regressor()

    for irefine in treelog.iter.fraction('level', range(nrefine+1)):

        if irefine:
            refdom = domain.refined
            refbasis = refdom.basis(btype, degree=degree)
            ns.add_field('vref', refbasis)
            res = refdom.integral('∇_k(vref) ∇_k(u) dV' @ ns, degree=degree*2)
            res -= refdom.boundary.integral('vref ∇_k(u) n_k dS' @ ns, degree=degree*2)
            indicator = res.derivative('vref').eval(**args)
            supp = refbasis.get_support(indicator**2 > numpy.mean(indicator**2))
            domain = domain.refined_by(refdom.transforms[supp])

        ns = Namespace()
        ns.x = geom
        ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
        ns.add_field(('u', 'v'), domain.basis(btype, degree=degree))
        ns.uexact = exact
        ns.du = 'u - uexact'

        sqr = domain.boundary['corner'].integral('u^2 dS' @ ns, degree=degree*2)
        cons = solver.optimize('u,', sqr, droptol=1e-15)

        sqr = domain.boundary.integral('du^2 dS' @ ns, degree=7)
        cons = solver.optimize('u,', sqr, droptol=1e-15, constrain=cons)

        res = domain.integral('∇_k(v) ∇_k(u) dV' @ ns, degree=degree*2)
        args = solver.solve_linear('u:v', res, constrain=cons)

        ndofs = len(args['u'])
        error = numpy.sqrt(domain.integral(['du du dV', '∇_k(du) ∇_k(du) dV'] @ ns, degree=7)).eval(**args)
        rate, offset = linreg.add(numpy.log(ndofs), numpy.log(error))
        treelog.user(f'ndofs: {ndofs}, L2 error: {error[0]:.2e} ({rate[0]:.2f}), H1 error: {error[1]:.2e} ({rate[1]:.2f})')

        bezier = domain.sample('bezier', 9)
        xsmp, usmp, dusmp = bezier.eval(['x_i', 'u', 'du'] @ ns, **args)
        export.triplot('sol.png', xsmp, usmp, tri=bezier.tri, hull=bezier.hull)
        export.triplot('err.png', xsmp, dusmp, tri=bezier.tri, hull=bezier.hull)

    return error, args['u']

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to perform four refinement steps with quadratic basis functions
# starting from a triangle mesh run :sh:`python3 adaptivity.py etype=triangle
# degree=2 nrefine=4`.


if __name__ == '__main__':
    cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.


class test(testing.TestCase):

    def test_square_quadratic(self):
        error, u = main(nrefine=2, btype='h-std', etype='square', degree=2)
        with self.subTest('degrees of freedom'):
            self.assertEqual(len(u), 149)
        with self.subTest('L2-error'):
            self.assertAlmostEqual(error[0], 0.00065, places=5)
        with self.subTest('H1-error'):
            self.assertAlmostEqual(error[1], 0.03461, places=5)
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNo1j6FrQmEUxT8RBi4KllVfMsl3z/nK4zEmLC6bhsKCw2gSw5IPFsymGbZiWnr+By8Ii7Yhsk3BMtC4
                Z9sJ223ncs85vzvmM9+Yhix8hDIjtnkdHqQSdDDDj1Qajr5qPXN/07MZ2vI4V7UOIvmdO/oEZY45xYDn
                oR7ikLHAHVpcs2A1TLhChDO+MOeWt5xjYzm6fOQrGxxiZPeoMGaf37hCyU72hB0u6PglPcQcKxRI/KUd
                7AYLvMPpsqGkCTPumzWf+qV92kKevjK36ozDP/FSnh1iteWiqWuf+oMaKuyKaC1i52rKPokiF2WLA/20
                bya+ZCPbWKRPpvgFaedebw==''')

    def test_triangle_quadratic(self):
        error, u = main(nrefine=2, btype='h-std', etype='triangle', degree=2)
        with self.subTest('degrees of freedom'):
            self.assertEqual(len(u), 98)
        with self.subTest('L2-error'):
            self.assertAlmostEqual(error[0], 0.00138, places=5)
        with self.subTest('H1-error'):
            self.assertAlmostEqual(error[1], 0.05324, places=5)
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNprMV1oesqU2VTO1Nbko6myWbhpq+kckwST90avjRgYzptYm+YYMwBBk3GQWavZb1NXs2+mm83um1WY
                bQbyXYEiQWbKZjNM7wJVzjBlYICoPW8CMiXH+LXRR9NwoPkg82xN5IB2MZu2mGabSBnnAbGscYEJj3GV
                YQAQg/TVGfaA7RI0BsErRjeNeowDgDQPmF9gkmciaJxtArGjzrAKCGWNpYAQAL0kOBE=''')

    def test_mixed_linear(self):
        error, u = main(nrefine=2, btype='h-std', etype='mixed', degree=1)
        with self.subTest('degrees of freedom'):
            self.assertEqual(len(u), 34)
        with self.subTest('L2-error'):
            self.assertAlmostEqual(error[0], 0.00450, places=5)
        with self.subTest('H1-error'):
            self.assertAlmostEqual(error[1], 0.11683, places=5)
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNprMT1u6mQyxUTRzMCUAQhazL6b3jNrMYPxp5iA5FtMD+lcMgDxHa4aXzS+6HDV+fKO85cMnC8zMBzS
                AQDBThbY''')
