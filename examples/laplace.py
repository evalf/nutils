#! /usr/bin/env python3
#
# In this script we solve the Laplace equation :math:`u_{,kk} = 0` on a unit
# square domain :math:`Ω` with boundary :math:`Γ`, subject to boundary
# conditions:
#
# .. math:: u &= 0                     && Γ_{\rm left}
#
#       ∂_n u &= 0                     && Γ_{\rm bottom}
#
#       ∂_n u &= \cos(1) \cosh(x_1)    && Γ_{\rm right}
#
#           u &= \cosh(1) \sin(x_0)    && Γ_{\rm top}
#
# This case is constructed to contain all combinations of homogenous and
# heterogeneous, Dirichlet and Neumann type boundary conditions, as well as to
# have a known exact solution:
#
# .. math:: u_{\rm exact} = \sin(x_0) \cosh(x_1).
#
# We start by importing the necessary modules.

import nutils, numpy

# The main function defines the parameter space for the script. Configurable
# parameters are the mesh density (in number of elements along an edge),
# element type (square, triangle, or mixed), type of basis function (std or
# spline, with availability depending on element type), and polynomial degree.

def main(nelems: 'number of elements along edge' = 10,
         etype: 'type of elements (square/triangle/mixed)' = 'square',
         btype: 'type of basis function (std/spline)' = 'std',
         degree: 'polynomial degree' = 1):

  # A unit square domain is created by calling the
  # :func:`nutils.mesh.unitsquare` mesh generator, with the number of elements
  # along an edge as the first argument, and the type of elements ("square",
  # "triangle", or "mixed") as the second. The result is a topology object
  # ``domain`` and a vectored valued geometry function ``geom``.

  domain, geom = nutils.mesh.unitsquare(nelems, etype)

  # To be able to write index based tensor contractions, we need to bundle all
  # relevant functions together in a namespace. Here we add the geometry ``x``,
  # a scalar ``basis``, and the solution ``u``. The latter is formed by
  # contracting the basis with a to-be-determined solution vector ``?lhs``.

  ns = nutils.function.Namespace()
  ns.x = geom
  ns.basis = domain.basis(btype, degree=degree)
  ns.u = 'basis_n ?lhs_n'

  # We are now ready to implement the Laplace equation. In weak form, the
  # solution is a scalar field :math:`u` for which:
  #
  # .. math:: ∀ v: ∫_Ω v_{,k} u_{,k} - ∫_{Γ_n} v f = 0.
  #
  # By linearity the test function :math:`v` can be replaced by the basis that
  # spans its space. The result is an integral ``res`` that evaluates to a
  # vector matching the size of the function space.

  res = domain.integral('basis_n,i u_,i' @ ns, geometry=ns.x, degree=degree*2)
  res -= domain.boundary['right'].integral('basis_n cos(1) cosh(x_1)' @ ns, geometry=ns.x, degree=degree*2)

  # The Dirichlet constraints are set by finding the coefficients that minimize
  # the error:
  #
  # .. math:: \min_u ∫_{\Gamma_d} (u - u_d)^2
  #
  # The resulting ``cons`` array holds numerical values for all the entries of
  # ``?lhs`` that contribute (up to ``droptol``) to the minimization problem.
  # All remaining entries are set to ``NaN``, signifying that these degrees of
  # freedom are unconstrained.

  sqr = domain.boundary['left'].integral('u^2' @ ns, geometry=ns.x, degree=degree*2)
  sqr += domain.boundary['top'].integral('(u - cosh(1) sin(x_0))^2' @ ns, geometry=ns.x, degree=degree*2)
  cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15)

  # The unconstrained entries of ``?lhs`` are to be determined such that the
  # residual vector evaluates to zero in the corresponding entries. This step
  # involves a linearization of ``res``, resulting in a jacobian matrix and
  # right hand side vector that are subsequently assembled and solved. The
  # resulting ``lhs`` array matches ``cons`` in the constrained entries.

  lhs = nutils.solver.solve_linear('lhs', res, constrain=cons)

  # Once all entries of ``?lhs`` are establised, the corresponding solution can
  # be vizualised by sampling values of ``ns.u`` along with physical
  # coordinates ``ns.x``, with the solution vector provided via the
  # ``arguments`` dictionary. The sample members ``tri`` and ``hull`` provide
  # additional inter-point information required for drawing the mesh and
  # element outlines.

  bezier = domain.sample('bezier', 9)
  x, u = bezier.eval([ns.x, ns.u], arguments=dict(lhs=lhs))
  nutils.export.triplot('solution.jpg', x, u, tri=bezier.tri, hull=bezier.hull)

  # To confirm that our computation is correct, we use our knowledge of the
  # analytical solution to evaluate the L2-error of the discrete result.

  err = domain.integrate('(u - sin(x_0) cosh(x_1))^2' @ ns, geometry=ns.x, degree=degree*2, arguments=dict(lhs=lhs))**.5
  nutils.log.user('L2 error: {:.2e}'.format(err))

  return cons, lhs, err

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to keep with the default arguments simply run :sh:`python3
# laplace.py`. To select mixed elements and quadratic basis functions add
# :sh:`python3 laplace.py etype=mixed degree=2`.

if __name__ == '__main__':
  nutils.cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategicly chosen return values for routine regression testing. Here we use
# the standard :mod:`unittest` framework, with
# :func:`nutils.numeric.assert_allclose64` facilitating the embedding of
# desired results as compressed base64 data.

import unittest

class test(unittest.TestCase):

  def test_default(self):
    cons, lhs, err = main(nelems=4, etype='square', btype='std', degree=1)
    nutils.numeric.assert_allclose64(cons, 'eNrbKPv1QZ3ip9sL1BgaILDYFMbaZwZj5ZnDW'
      'NfNAeWPESU=')
    nutils.numeric.assert_allclose64(lhs, 'eNoBMgDN/7Ed9eB+IfLboCaXNKc01DQaNXM14j'
      'XyNR82ZTa+NpI2oTbPNhU3bjf7Ngo3ODd+N9c3SNEU1g==')
    numpy.testing.assert_almost_equal(err, 1.63e-3, decimal=5)

  def test_spline(self):
    cons, lhs, err = main(nelems=4, etype='square', btype='spline', degree=2)
    nutils.numeric.assert_allclose64(cons, 'eNqrkmN+sEfhzF0xleRbDA0wKGeCYFuaIdjK5'
      'gj2aiT2VXMAJB0VAQ==')
    nutils.numeric.assert_allclose64(lhs, 'eNqrkmN+sEfhzF0xleRbrsauxsnGc43fGMuZJJ'
      'gmmNaZ7jBlN7M08wLCDLNFZh/NlM0vmV0y+2CmZV5pvtr8j9kfMynzEPPF5lfNAcuhGvs=')
    numpy.testing.assert_almost_equal(err, 8.04e-5, decimal=7)

  def test_mixed(self):
    cons, lhs, err = main(nelems=4, etype='mixed', btype='std', degree=2)
    nutils.numeric.assert_allclose64(cons, 'eNorfLZF2ucJQwMC3pR7+QDG9lCquAtj71Rlu'
      '8XQIGfC0FBoiqweE1qaMTTsNsOvRtmcoSHbHL+a1UD5q+YAxhcu1g==')
    nutils.numeric.assert_allclose64(lhs, 'eNorfLZF2ueJq7GrcYjxDJPpJstNbsq9fOBr3G'
      'h8xWS7iYdSxd19xseMP5hImu5UZbv1xljOxM600DTWNN/0k2mC6SPTx6Z1pnNMGc3kzdaaPjRN'
      'MbMyEzWzNOsy223mBYRRZpPNJpktMks1azM7Z7bRbIXZabNXZiLmH82UzS3Ns80vmj004za/ZP'
      'YHCD+Y8ZlLmVuYq5kHm9eahwDxavPF5lfNAWFyPdk=')
    numpy.testing.assert_almost_equal(err, 1.25e-4, decimal=6)
