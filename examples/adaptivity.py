from nutils import mesh, function, solver, export, testing
from nutils.expression_v2 import Namespace
import numpy
import treelog


def main(etype: str = 'square',
         btype: str = 'h-std',
         degree: int = 2,
         nrefine: int = 5):

    '''Adaptively refined Laplace problem on an L-shaped domain
    
    Solves the Laplace problem on a unit square that has the bottom-right
    quadrant removed (a.k.a. an L-shaped domain) with Dirichlet boundary
    conditions matching the harmonic function
    
        ³√(x² + y²) cos(⅔ arctan2(y+x, y-x))
    
    shifted by ½ such that the origin coincides with the center of the unit
    square. Note that the function evaluates to zero where the domain borders
    the removed quadrant.
    
    This benchmark problem is known to converge suboptimally under uniform
    refinement due to a singular gradient in the reentrant corner. This script
    demonstrates that optimal convergence rates of -(p+1)/2 for the L2 norm and
    -p/2 for the H1 norm can be restored by using adaptive refinement.

    Parameters
    ----------
    etype
        Type of elements (square/triangle/mixed).
    btype
        Type of basis function (h/th-std/spline), with availability depending
        on the configured element type.
    degree
        Polynomial degree.
    nrefine
        Number of refinement steps to perform.
    '''

    domain, geom = mesh.unitsquare(2, etype)
    geom -= .5 # shift domain center to origin

    x, y = geom
    exact = (x**2 + y**2)**(1/3) * numpy.cos(numpy.arctan2(y+x, y-x) * (2/3))
    selection = domain.select(exact, ischeme='gauss1')
    domain = domain.subset(selection, newboundary='corner')
    linreg = LinearRegressor(bias=1)

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
        error = numpy.sqrt(domain.integral(['du^2 dV', '(du^2 + ∇_k(du) ∇_k(du)) dV'] @ ns, degree=7)).eval(**args)
        treelog.user(f'errors at {ndofs} dofs: L2 {error[0]:.2e}, H1 {error[1]:.2e}')
        linreg[numpy.log(ndofs)] = numpy.log(error)
        if irefine:
            treelog.user(f'error convergence rates: L2 {linreg.rate[0]:.2f} (optimal {-(degree+1)/2}), H1 {linreg.rate[1]:.2f} (optimal {-degree/2})')

        bezier = domain.sample('bezier', 9)
        xsmp, usmp, dusmp = bezier.eval(['x_i', 'u', 'du'] @ ns, **args)
        export.triplot('sol.png', xsmp, usmp, tri=bezier.tri, hull=bezier.hull)
        export.triplot('err.png', xsmp, dusmp, tri=bezier.tri, hull=bezier.hull)

    return error, args['u']


class LinearRegressor:
    '''Linear regression facilitator.

    For a growing collection of (x, y) data points, this class continuously
    computes the linear trend of the form y = offset + rate * x that has the
    least square error. Data points are added using setitem:

    >>> linreg = LinearRegressor()
    >>> linreg[1] = 10
    >>> linreg[2] = 5
    >>> linreg.offset
    15
    >>> linreg.rate
    -5

    Rather than storing the sequence, this class continuously updates the sums
    and inner products required to compute the offset and rate. The optional
    `bias` argument can be used to weight every newly added point `2**bias`
    times that of the previous, so as to emphasize focus on the tail of the
    sequence.
    '''

    def __init__(self, bias=0):
        self.n = self.x = self.y = self.xx = self.xy = 0.
        self.w = .5**bias

    def __setitem__(self, x, y):
        self.n = self.n * self.w + 1
        self.x = self.x * self.w + x
        self.y = self.y * self.w + y
        self.xx = self.xx * self.w + x * x
        self.xy = self.xy * self.w + x * y

    @property
    def rate(self):
        return (self.n * self.xy - self.x * self.y) / (self.n * self.xx - self.x**2)

    @property
    def offset(self):
        return (self.xx * self.y - self.x * self.xy) / (self.n * self.xx - self.x**2)


class test(testing.TestCase):

    def test_square_quadratic(self):
        error, u = main(nrefine=2)
        with self.subTest('degrees of freedom'):
            self.assertEqual(len(u), 149)
        with self.subTest('L2-error'):
            self.assertAlmostEqual(error[0], 0.00065, places=5)
        with self.subTest('H1-error'):
            self.assertAlmostEqual(error[1], 0.03462, places=5)
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNo1j6FrQmEUxT8RBi4KllVfMsl3z/nK4zEmLC6bhsKCw2gSw5IPFsymGbZiWnr+By8Ii7Yhsk3BMtC4
                Z9sJ223ncs85vzvmM9+Yhix8hDIjtnkdHqQSdDDDj1Qajr5qPXN/07MZ2vI4V7UOIvmdO/oEZY45xYDn
                oR7ikLHAHVpcs2A1TLhChDO+MOeWt5xjYzm6fOQrGxxiZPeoMGaf37hCyU72hB0u6PglPcQcKxRI/KUd
                7AYLvMPpsqGkCTPumzWf+qV92kKevjK36ozDP/FSnh1iteWiqWuf+oMaKuyKaC1i52rKPokiF2WLA/20
                bya+ZCPbWKRPpvgFaedebw==''')

    def test_triangle_quadratic(self):
        error, u = main(nrefine=2, etype='triangle')
        with self.subTest('degrees of freedom'):
            self.assertEqual(len(u), 98)
        with self.subTest('L2-error'):
            self.assertAlmostEqual(error[0], 0.00138, places=5)
        with self.subTest('H1-error'):
            self.assertAlmostEqual(error[1], 0.05326, places=5)
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNprMV1oesqU2VTO1Nbko6myWbhpq+kckwST90avjRgYzptYm+YYMwBBk3GQWavZb1NXs2+mm83um1WY
                bQbyXYEiQWbKZjNM7wJVzjBlYICoPW8CMiXH+LXRR9NwoPkg82xN5IB2MZu2mGabSBnnAbGscYEJj3GV
                YQAQg/TVGfaA7RI0BsErRjeNeowDgDQPmF9gkmciaJxtArGjzrAKCGWNpYAQAL0kOBE=''')

    def test_mixed_linear(self):
        error, u = main(nrefine=2, etype='mixed', degree=1)
        with self.subTest('degrees of freedom'):
            self.assertEqual(len(u), 34)
        with self.subTest('L2-error'):
            self.assertAlmostEqual(error[0], 0.00450, places=5)
        with self.subTest('H1-error'):
            self.assertAlmostEqual(error[1], 0.11692, places=5)
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNprMT1u6mQyxUTRzMCUAQhazL6b3jNrMYPxp5iA5FtMD+lcMgDxHa4aXzS+6HDV+fKO85cMnC8zMBzS
                AQDBThbY''')


if __name__ == '__main__':
    from nutils import cli
    cli.run(main)


# example:tags=hierarchical refinements,Laplace:thumbnail=0
