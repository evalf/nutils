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

from nutils import mesh, function, solver, export, cli, testing
import treelog

def main(nelems:int, etype:str, btype:str, degree:int):
  '''
  Laplace problem on a unit square.

  .. arguments::

     nelems [10]
       Number of elements along edge.
     etype [square]
       Type of elements (square/triangle/mixed).
     btype [std]
       Type of basis function (std/spline), availability depending on the
       selected element type.
     degree [1]
       Polynomial degree.
  '''

  # A unit square domain is created by calling the
  # :func:`nutils.mesh.unitsquare` mesh generator, with the number of elements
  # along an edge as the first argument, and the type of elements ("square",
  # "triangle", or "mixed") as the second. The result is a topology object
  # ``domain`` and a vectored valued geometry function ``geom``.

  domain, geom = mesh.unitsquare(nelems, etype)

  # To be able to write index based tensor contractions, we need to bundle all
  # relevant functions together in a namespace. Here we add the geometry ``x``,
  # a scalar ``basis``, and the solution ``u``. The latter is formed by
  # contracting the basis with a to-be-determined solution vector ``?lhs``.

  ns = function.Namespace()
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

  res = domain.integral('basis_n,i u_,i d:x' @ ns, degree=degree*2)
  res -= domain.boundary['right'].integral('basis_n cos(1) cosh(x_1) d:x' @ ns, degree=degree*2)

  # The Dirichlet constraints are set by finding the coefficients that minimize
  # the error:
  #
  # .. math:: \min_u ∫_{\Gamma_d} (u - u_d)^2
  #
  # The resulting ``cons`` array holds numerical values for all the entries of
  # ``?lhs`` that contribute (up to ``droptol``) to the minimization problem.
  # All remaining entries are set to ``NaN``, signifying that these degrees of
  # freedom are unconstrained.

  sqr = domain.boundary['left'].integral('u^2 d:x' @ ns, degree=degree*2)
  sqr += domain.boundary['top'].integral('(u - cosh(1) sin(x_0))^2 d:x' @ ns, degree=degree*2)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)

  # The unconstrained entries of ``?lhs`` are to be determined such that the
  # residual vector evaluates to zero in the corresponding entries. This step
  # involves a linearization of ``res``, resulting in a jacobian matrix and
  # right hand side vector that are subsequently assembled and solved. The
  # resulting ``lhs`` array matches ``cons`` in the constrained entries.

  lhs = solver.solve_linear('lhs', res, constrain=cons)

  # Once all entries of ``?lhs`` are establised, the corresponding solution can
  # be vizualised by sampling values of ``ns.u`` along with physical
  # coordinates ``ns.x``, with the solution vector provided via the
  # ``arguments`` dictionary. The sample members ``tri`` and ``hull`` provide
  # additional inter-point information required for drawing the mesh and
  # element outlines.

  bezier = domain.sample('bezier', 9)
  x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)
  export.triplot('solution.png', x, u, tri=bezier.tri, hull=bezier.hull)

  # To confirm that our computation is correct, we use our knowledge of the
  # analytical solution to evaluate the L2-error of the discrete result.

  err = domain.integral('(u - sin(x_0) cosh(x_1))^2 d:x' @ ns, degree=degree*2).eval(lhs=lhs)**.5
  treelog.user('L2 error: {:.2e}'.format(err))

  return cons, lhs, err

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to keep with the default arguments simply run :sh:`python3
# laplace.py`. To select mixed elements and quadratic basis functions add
# :sh:`python3 laplace.py etype=mixed degree=2`.

if __name__ == '__main__':
  cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(testing.TestCase):

  @testing.requires('matplotlib')
  def test_default(self):
    cons, lhs, err = main(nelems=4, etype='square', btype='std', degree=1)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons, '''
      eNrbKPv1QZ3ip9sL1BgaILDYFMbaZwZj5ZnDWNfNAeWPESU=''')
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNoBMgDN/7Ed9eB+IfLboCaXNKc01DQaNXM14jXyNR82ZTa+NpI2oTbPNhU3bjf7Ngo3ODd+N9c3SNEU
      1g==''')
    with self.subTest('L2-error'):
      self.assertAlmostEqual(err, 1.63e-3, places=5)

  @testing.requires('matplotlib')
  def test_spline(self):
    cons, lhs, err = main(nelems=4, etype='square', btype='spline', degree=2)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons, '''
      eNqrkmN+sEfhzF0xleRbDA0wKGeCYFuaIdjK5gj2aiT2VXMAJB0VAQ==''')
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNqrkmN+sEfhzF0xleRbrsauxsnGc43fGMuZJJgmmNaZ7jBlN7M08wLCDLNFZh/NlM0vmV0y+2CmZV5p
      vtr8j9kfMynzEPPF5lfNAcuhGvs=''')
    with self.subTest('L2-error'):
      self.assertAlmostEqual(err, 8.04e-5, places=7)

  @testing.requires('matplotlib')
  def test_mixed(self):
    cons, lhs, err = main(nelems=4, etype='mixed', btype='std', degree=2)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons, '''
      eNorfLZF2ucJQwMC3pR7+QDG9lCquAtj71Rlu8XQIGfC0FBoiqweE1qaMTTsNsOvRtmcoSHbHL+a1UD5
      q+YAxhcu1g==''')
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNorfLZF2ueJq7GrcYjxDJPpJstNbsq9fOBr3Gh8xWS7iYdSxd19xseMP5hImu5UZbv1xljOxM600DTW
      NN/0k2mC6SPTx6Z1pnNMGc3kzdaaPjRNMbMyEzWzNOsy223mBYRRZpPNJpktMks1azM7Z7bRbIXZabNX
      ZiLmH82UzS3Ns80vmj004za/ZPYHCD+Y8ZlLmVuYq5kHm9eahwDxavPF5lfNAWFyPdk=''')
    with self.subTest('L2-error'):
      self.assertAlmostEqual(err, 1.25e-4, places=6)
