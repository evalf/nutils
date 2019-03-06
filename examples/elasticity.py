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

  sqr = domain.boundary['left'].integral('u_k u_k d:x' @ ns, degree=degree*2)
  sqr += domain.boundary['right'].integral('(u_0 - .5)^2 d:x' @ ns, degree=degree*2)
  cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15)

  res = domain.integral('basis_ni,j stress_ij d:x' @ ns, degree=degree*2)
  lhs = nutils.solver.solve_linear('lhs', res, constrain=cons)

  bezier = domain.sample('bezier', 5)
  X, sxy = bezier.eval(['X_i', 'stress_01'] @ ns, lhs=lhs)
  nutils.export.triplot('shear.png', X, sxy, tri=bezier.tri, hull=bezier.hull)

  return cons, lhs

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to keep with the default arguments simply run :sh:`python3
# elasticity.py`. To select mixed elements and quadratic basis functions add
# :sh:`python3 elasticity.py etype=mixed degree=2`.

if __name__ == '__main__':
  nutils.cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(nutils.testing.TestCase):

  @nutils.testing.requires('matplotlib')
  def test_default(self):
    cons, lhs = main(nelems=4)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons, '''
      eNpjYICDBnzwhykMMhCpAwEBQ08XYg==''')
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNpjYICBFGMxYyEgTjFebDLBpB2IF5tkmKaYJgJxhukPOIRrYBA1CjJgYFh3/vXZMiMVQwaGO+e6zvYY
      2QBZR86VnO2FsorPAgAXLB7S''')

  @nutils.testing.requires('matplotlib')
  def test_mixed(self):
    cons, lhs = main(nelems=4, etype='mixed')
    with self.subTest('constraints'): self.assertAlmostEqual64(cons, '''
      eNpjYACCBiBkQMJY4A9TGGQgUgcCAgBVTxdi''')
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNpjYGBgSDKWNwZSQKwExAnGfSbLTdpNek2WmWSYppgmAHGG6Q84BKpk4DASN2Bg2K/JwHDrPAPDj7Mq
      hnlGRddenpt+ts/I0nChyrlzJWcdDbuNYjUOnSs/CwB0uyJb''')

  @nutils.testing.requires('matplotlib')
  def test_quadratic(self):
    cons, lhs = main(nelems=4, degree=2)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons, '''
      eNpjYMAADQMJf5iiQ4ZB5kJMCAAkxE4W''')
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNpjYEAHlUauhssMuw2nAvEyQ1fDSqMsY1NjJWNxYzEgVgKys4xlTThNfhu/NX4HxL+NOU1kTRabzDaZ
      bNJj0g3Ek4HsxSa8ptym7KZMYMgOZPOaZpimm6aYJoFhCpCdYboFCDfDIYj3AwNiOJDhviGPQbf+RV0G
      Bv1LpRe+nFc8x22UY5hv8F6PgUHw4sTzU859PZtldNGQ3XCCPgNDwYWf5/TPTTtbYvTKUNpwP1DE8cLT
      c2Lnes62Gf01NDW8BxRRunD6HPO5KqjIA6CIAlSkw+ifobnhI6CI3IWT55jOVQBF/hqaGT4EishfOAVU
      U3EWAA5lcd0=''')

  @nutils.testing.requires('matplotlib')
  def test_poisson(self):
    cons, lhs = main(nelems=4, poisson=.4)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons, '''
      eNpjYICDBnzwhykMMhCpAwEBQ08XYg==''')
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNpjYIABC+M1RkuN1hhZGE8xyTKJAOIpJomm4aaBQJxo+gMO4RoYJhu/MWRgEDmXe+a18QKj//8Tzoqe
      YTLZCmR5n/13msVkG5DldfbPaQC28iVf''')
