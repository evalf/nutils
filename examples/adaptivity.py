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

from nutils import mesh, function, solver, util, export, cli, testing
import numpy, treelog

def main(etype:str, btype:str, degree:int, nrefine:int):
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

  x, y = geom - .5
  exact = (x**2 + y**2)**(1/3) * function.cos(function.arctan2(y+x, y-x) * (2/3))
  domain = domain.trim(exact-1e-15, maxrefine=0)
  linreg = util.linear_regressor()

  with treelog.iter.fraction('level', range(nrefine+1)) as lrange:
    for irefine in lrange:

      if irefine:
        refdom = domain.refined
        ns.refbasis = refdom.basis(btype, degree=degree)
        indicator = refdom.integral('refbasis_n,k u_,k d:x' @ ns, degree=degree*2).eval(lhs=lhs)
        indicator -= refdom.boundary.integral('refbasis_n u_,k n_k d:x' @ ns, degree=degree*2).eval(lhs=lhs)
        supp = ns.refbasis.get_support(indicator**2 > numpy.mean(indicator**2))
        domain = domain.refined_by(ns.refbasis.transforms[supp])

      ns = function.Namespace()
      ns.x = geom
      ns.basis = domain.basis(btype, degree=degree)
      ns.u = 'basis_n ?lhs_n'
      ns.du = ns.u - exact

      sqr = domain.boundary['trimmed'].integral('u^2 d:x' @ ns, degree=degree*2)
      cons = solver.optimize('lhs', sqr, droptol=1e-15)

      sqr = domain.boundary.integral('du^2 d:x' @ ns, degree=7)
      cons = solver.optimize('lhs', sqr, droptol=1e-15, constrain=cons)

      res = domain.integral('basis_n,k u_,k d:x' @ ns, degree=degree*2)
      lhs = solver.solve_linear('lhs', res, constrain=cons)

      ndofs = len(ns.basis)
      error = domain.integral('<du^2, du_,k du_,k>_i d:x' @ ns, degree=7).eval(lhs=lhs)**.5
      rate, offset = linreg.add(numpy.log(len(ns.basis)), numpy.log(error))
      treelog.user('ndofs: {ndofs}, L2 error: {error[0]:.2e} ({rate[0]:.2f}), H1 error: {error[1]:.2e} ({rate[1]:.2f})'.format(ndofs=len(ns.basis), error=error, rate=rate))

      bezier = domain.sample('bezier', 9)
      x, u, du = bezier.eval(['x_i', 'u', 'du'] @ ns, lhs=lhs)
      export.triplot('sol.png', x, u, tri=bezier.tri, hull=bezier.hull)
      export.triplot('err.png', x, du, tri=bezier.tri, hull=bezier.hull)

  return ndofs, error, lhs

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

  @testing.requires('matplotlib')
  def test_square_quadratic(self):
    ndofs, error, lhs = main(nrefine=2, btype='h-std', etype='square', degree=2)
    with self.subTest('degrees of freedom'):
      self.assertEqual(ndofs, 149)
    with self.subTest('L2-error'):
      self.assertAlmostEqual(error[0], 0.00065, places=5)
    with self.subTest('H1-error'):
      self.assertAlmostEqual(error[1], 0.03461, places=5)
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNo1j6FrQmEUxT8RBi4KllVfMsl3z/nK4zEmLC6bhsKCw2gSw5IPFsymGbZiWnr+By8Ii7Yhsk3BMtC4
      Z9sJ223ncs85vzvmM9+Yhix8hDIjtnkdHqQSdDDDj1Qajr5qPXN/07MZ2vI4V7UOIvmdO/oEZY45xYDn
      oR7ikLHAHVpcs2A1TLhChDO+MOeWt5xjYzm6fOQrGxxiZPeoMGaf37hCyU72hB0u6PglPcQcKxRI/KUd
      7AYLvMPpsqGkCTPumzWf+qV92kKevjK36ozDP/FSnh1iteWiqWuf+oMaKuyKaC1i52rKPokiF2WLA/20
      bya+ZCPbWKRPpvgFaedebw==''')

  @testing.requires('matplotlib')
  def test_triangle_quadratic(self):
    ndofs, error, lhs = main(nrefine=2, btype='h-std', etype='triangle', degree=2)
    with self.subTest('degrees of freedom'):
      self.assertEqual(ndofs, 98)
    with self.subTest('L2-error'):
      self.assertAlmostEqual(error[0], 0.00138, places=5)
    with self.subTest('H1-error'):
      self.assertAlmostEqual(error[1], 0.05324, places=5)
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNprMV1oesqU2VTO1Nbko6myWbhpq+kckwST90avjRgYzptYm+YYMwBBk3GQWavZb1NXs2+mm83um1WY
      bQbyXYEiQWbKZjNM7wJVzjBlYICoPW8CMiXH+LXRR9NwoPkg82xN5IB2MZu2mGabSBnnAbGscYEJj3GV
      YQAQg/TVGfaA7RI0BsErRjeNeowDgDQPmF9gkmciaJxtArGjzrAKCGWNpYAQAL0kOBE=''')

  @testing.requires('matplotlib')
  def test_mixed_linear(self):
    ndofs, error, lhs = main(nrefine=2, btype='h-std', etype='mixed', degree=1)
    with self.subTest('degrees of freedom'):
      self.assertEqual(ndofs, 34)
    with self.subTest('L2-error'):
      self.assertAlmostEqual(error[0], 0.00450, places=5)
    with self.subTest('H1-error'):
      self.assertAlmostEqual(error[1], 0.11683, places=5)
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNprMT1u6mQyxUTRzMCUAQhazL6b3jNrMYPxp5iA5FtMD+lcMgDxHa4aXzS+6HDV+fKO85cMnC8zMBzS
      AQDBThbY''')
