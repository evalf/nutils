#! /usr/bin/env python3
#
# In this script we solve the linear elasticity problem on a unit square
# domain, clamped at the left boundary, and stretched at the right boundary
# while keeping vertical displacements free.

import nutils

# The main function defines the parameter space for the script. Configurable
# parameters are the mesh density (in number of elements along an edge),
# element type (square, triangle, or mixed), type of basis function (std or
# spline, with availability depending on element type), polynomial degree, and
# Poisson's ratio.

def main(nelems: 'number of elements along edge' = 10,
         etype: 'type of elements (square/triangle/mixed)' = 'square',
         btype: 'type of basis function (std/spline)' = 'std',
         degree: 'polynomial degree' = 1,
         poisson: 'poisson ratio < 0.5' = .25):

  domain, geom = nutils.mesh.unitsquare(nelems, etype)

  ns = nutils.function.Namespace()
  ns.x = geom
  ns.basis = domain.basis(btype, degree=degree).vector(2)
  ns.u_i = 'basis_ni ?lhs_n'
  ns.X_i = 'x_i + u_i'
  ns.lmbda = 2 * poisson
  ns.mu = 1 - 2 * poisson
  ns.strain_ij = '(u_i,j + u_j,i) / 2'
  ns.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'

  sqr = domain.boundary['left'].integral('u_k u_k' @ ns, geometry=ns.x, degree=degree*2)
  sqr += domain.boundary['right'].integral('(u_0 - .5)^2' @ ns, geometry=ns.x, degree=degree*2)
  cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15)

  res = domain.integral('basis_ni,j stress_ij' @ ns, geometry=ns.x, degree=degree*2)
  lhs = nutils.solver.solve_linear('lhs', res, constrain=cons)

  bezier = domain.sample('bezier', 5)
  X, sxy = bezier.eval([ns.X, ns.stress[0,1]], arguments=dict(lhs=lhs))
  nutils.export.triplot('shear.jpg', X, sxy, tri=bezier.tri, hull=bezier.hull)

  return cons, lhs

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to keep with the default arguments simply run :sh:`python3
# elasticity.py`. To select mixed elements and quadratic basis functions add
# :sh:`python3 elasticity.py etype=mixed degree=2`.

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
    cons, lhs = main(nelems=4)
    nutils.numeric.assert_allclose64(cons, 'eNpjYICDBnzwhykMMhCpAwEBQ08XYg==')
    nutils.numeric.assert_allclose64(lhs, 'eNpjYICBFGMxYyEgTjFebDLBpB2IF5tkmKaYJg'
      'JxhukPOIRrYBA1CjJgYFh3/vXZMiMVQwaGO+e6zvYY2QBZR86VnO2FsorPAgAXLB7S')

  def test_mixed(self):
    cons, lhs = main(nelems=4, etype='mixed')
    nutils.numeric.assert_allclose64(cons, 'eNpjYACCBiBkQMJY4A9TGGQgUgcCAgBVTxdi')
    nutils.numeric.assert_allclose64(lhs, 'eNpjYGBgSDKWNwZSQKwExAnGfSbLTdpNek2WmW'
      'SYppgmAHGG6Q84BKpk4DASN2Bg2K/JwHDrPAPDj7MqhnlGRddenpt+ts/I0nChyrlzJWcdDbuN'
      'YjUOnSs/CwB0uyJb')

  def test_quadratic(self):
    cons, lhs = main(nelems=4, degree=2)
    nutils.numeric.assert_allclose64(cons, 'eNpjYMAADQMJf5iiQ4ZB5kJMCAAkxE4W')
    nutils.numeric.assert_allclose64(lhs, 'eNpjYEAHlUauhssMuw2nAvEyQ1fDSqMsY1NjJW'
      'NxYzEgVgKys4xlTThNfhu/NX4HxL+NOU1kTRabzDaZbNJj0g3Ek4HsxSa8ptym7KZMYMgOZPOa'
      'Zpimm6aYJoFhCpCdYboFCDfDIYj3AwNiOJDhviGPQbf+RV0GBv1LpRe+nFc8x22UY5hv8F6PgU'
      'Hw4sTzU859PZtldNGQ3XCCPgNDwYWf5/TPTTtbYvTKUNpwP1DE8cLTc2Lnes62Gf01NDW8BxRR'
      'unD6HPO5KqjIA6CIAlSkw+ifobnhI6CI3IWT55jOVQBF/hqaGT4EishfOAVUU3EWAA5lcd0=')

  def test_poisson(self):
    cons, lhs = main(nelems=4, poisson=.4)
    nutils.numeric.assert_allclose64(cons, 'eNpjYICDBnzwhykMMhCpAwEBQ08XYg==')
    nutils.numeric.assert_allclose64(lhs, 'eNpjYIABC+M1RkuN1hhZGE8xyTKJAOIpJomm4a'
      'aBQJxo+gMO4RoYJhu/MWRgEDmXe+a18QKj//8TzoqeYTLZCmR5n/13msVkG5DldfbPaQC28iVf')
