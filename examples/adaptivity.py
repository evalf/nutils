#! /usr/bin/env python3
#
# In this script we solve the Laplace problem on a unit square that has the
# bottom-right quadrant removed (a.k.a. an L-shaped domain) with Dirichlet
# boundary conditions matching the harmonic function
#
# .. math:: \sqrt[3]{x^2 + y^2} \cos\left(\tfrac23 \arctan\frac{y+x}{y-x}\right),
#
# shifted by 0.5 such that the origin coincides with the middle of the unit
# square. This variation of a well known benchmark problem is known to converge
# suboptimally under uniform refinement due to a singular gradient in the
# reentrant corner. This script demonstrates that optimal convergence can be
# restored by using adaptive refinement.

import nutils, numpy

# The main function defines the parameter space for the script. Configurable
# parameters are the element type (square, triangle, or mixed), type of basis
# function (std or spline, with availability depending on element type),
# polynomial degree, and the number of refinement steps to perform before
# quitting (by default the script will run forever).

def main(etype: 'type of elements (square/triangle/mixed)' = 'square',
         btype: 'type of basis function (h/th-std/spline)' = 'h-std',
         degree: 'polynomial degree' = 2,
         nrefine: 'number of refinement steps (-1 for unlimited)' = -1):

  domain, geom = nutils.mesh.unitsquare(2, etype)

  x, y = geom - .5
  exact = (x**2 + y**2)**(1/3) * nutils.function.cos(nutils.function.arctan2(y+x, y-x) * (2/3))
  domain = domain.trim(exact-1e-15, maxrefine=0)
  linreg = nutils.util.linear_regressor()

  for irefine in nutils.log.count('level'):

    ns = nutils.function.Namespace()
    ns.x = geom
    ns.basis = domain.basis(btype, degree=degree)
    ns.u = 'basis_n ?lhs_n'
    ns.du = ns.u - exact

    sqr = domain.boundary['trimmed'].integral('u^2 d:x' @ ns, degree=degree*2)
    cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15)

    sqr = domain.boundary.integral('du^2 d:x' @ ns, degree=7)
    cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15, constrain=cons)

    res = domain.integral('basis_n,k u_,k d:x' @ ns, degree=degree*2)
    lhs = nutils.solver.solve_linear('lhs', res, constrain=cons)

    ndofs = len(ns.basis)
    error = domain.integral('<du^2, du_,k du_,k>_i d:x' @ ns, degree=7).eval(lhs=lhs)**.5
    rate, offset = linreg.add(numpy.log(len(ns.basis)), numpy.log(error))
    nutils.log.user('ndofs: {ndofs}, L2 error: {error[0]:.2e} ({rate[0]:.2f}), H1 error: {error[1]:.2e} ({rate[1]:.2f})'.format(ndofs=len(ns.basis), error=error, rate=rate))

    bezier = domain.sample('bezier', 9)
    x, u, du = bezier.eval(['x_i', 'u', 'du'] @ ns, lhs=lhs)
    nutils.export.triplot('sol.jpg', x, u, tri=bezier.tri, hull=bezier.hull)
    nutils.export.triplot('err.jpg', x, du, tri=bezier.tri, hull=bezier.hull)

    if irefine == nrefine:
      break

    refdom = domain.refined
    ns.refbasis = refdom.basis(btype, degree=degree)
    indicator = refdom.integral('refbasis_n,k u_,k d:x' @ ns, degree=degree*2).eval(lhs=lhs)
    indicator -= refdom.boundary.integral('refbasis_n u_,k n_k d:x' @ ns, degree=degree*2).eval(lhs=lhs)
    mask = indicator**2 > numpy.mean(indicator**2)

    domain = domain.refined_by(elem.transform[:-1] for elem in domain.refined.supp(ns.refbasis, mask))

  return ndofs, error, rate, lhs

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to perform four refinement steps with quadratic basis functions
# starting from a triangle mesh run :sh:`python3 adaptivity.py etype=triangle
# degree=2 nrefine=4`.

if __name__ == '__main__':
  nutils.cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategicly chosen return values for routine regression testing. Here we use
# the standard :mod:`unittest` framework, with
# :func:`nutils.numeric.assert_allclose64` facilitating the embedding of
# desired results as compressed base64 data.

class test(nutils.testing.TestCase):

  def test_square_quadratic(self):
    ndofs, error, rate, lhs = main(nrefine=2, etype='square', degree=2)
    self.assertEqual(ndofs, 149)
    numpy.testing.assert_almost_equal(error, [0.00065, 0.03461], decimal=5)
    numpy.testing.assert_almost_equal(rate, [-1.066, -0.478], decimal=3)
    nutils.numeric.assert_allclose64(lhs, 'eNo1j6FrQmEUxT8RBi4KllVfMsl3z/nK4zEmLC'
      '6bhsKCw2gSw5IPFsymGbZiWnr+By8Ii7Yhsk3BMtC4Z9sJ223ncs85vzvmM9+Yhix8hDIjtnkd'
      'HqQSdDDDj1Qajr5qPXN/07MZ2vI4V7UOIvmdO/oEZY45xYDnoR7ikLHAHVpcs2A1TLhChDO+MO'
      'eWt5xjYzm6fOQrGxxiZPeoMGaf37hCyU72hB0u6PglPcQcKxRI/KUd7AYLvMPpsqGkCTPumzWf'
      '+qV92kKevjK36ozDP/FSnh1iteWiqWuf+oMaKuyKaC1i52rKPokiF2WLA/20bya+ZCPbWKRPpv'
      'gFaedebw==')

  def test_triangle_quadratic(self):
    ndofs, error, rate, lhs = main(nrefine=2, etype='triangle', degree=2)
    self.assertEqual(ndofs, 98)
    numpy.testing.assert_almost_equal(error, [0.00138, 0.05324], decimal=5)
    numpy.testing.assert_almost_equal(rate, [-1.111, -0.548], decimal=3)
    nutils.numeric.assert_allclose64(lhs, 'eNprMV1oesqU2VTO1Nbko6myWbhpq+kckwST90'
      'avjRgYzptYm+YYMwBBk3GQWavZb1NXs2+mm83um1WYbQbyXYEiQWbKZjNM7wJVzjBlYICoPW8C'
      'MiXH+LXRR9NwoPkg82xN5IB2MZu2mGabSBnnAbGscYEJj3GVYQAQg/TVGfaA7RI0BsErRjeNeo'
      'wDgDQPmF9gkmciaJxtArGjzrAKCGWNpYAQAL0kOBE=')

  def test_mixed_linear(self):
    ndofs, error, rate, lhs = main(nrefine=2, etype='mixed', degree=1)
    self.assertEqual(ndofs, 34)
    numpy.testing.assert_almost_equal(error, [0.00450, 0.11683], decimal=5)
    numpy.testing.assert_almost_equal(rate, [-1.143, -0.545], decimal=3)
    nutils.numeric.assert_allclose64(lhs, 'eNprMT1u6mQyxUTRzMCUAQhazL6b3jNrMYPxp5'
      'iA5FtMD+lcMgDxHa4aXzS+6HDV+fKO85cMnC8zMBzSAQDBThbY')
