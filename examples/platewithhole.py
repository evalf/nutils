#! /usr/bin/env python3
#
# In this script we solve the linear plane strain elasticity problem for an
# infinite plate with a circular hole under tension. We do this by placing the
# circle in the origin of a unit square, imposing symmetry conditions on the
# left and bottom, and Dirichlet conditions constraining the displacements to
# the analytical solution to the right and top. The traction-free circle is
# removed by means of the Finite Cell Method (FCM).

from nutils import mesh, function, solver, export, cli, testing
import numpy, treelog

def main(nelems:int, etype:str, btype:str, degree:int, traction:float, maxrefine:int, radius:float, poisson:float):
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

  ns = function.Namespace()
  ns.x = geom
  ns.lmbda = 2 * poisson
  ns.mu = 1 - poisson
  ns.ubasis = domain.basis(btype, degree=degree).vector(2)
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.X_i = 'x_i + u_i'
  ns.strain_ij = '(u_i,j + u_j,i) / 2'
  ns.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'
  ns.r2 = 'x_k x_k'
  ns.R2 = radius**2 / ns.r2
  ns.k = (3-poisson) / (1+poisson) # plane stress parameter
  ns.scale = traction * (1+poisson) / 2
  ns.uexact_i = 'scale (x_i ((k + 1) (0.5 + R2) + (1 - R2) R2 (x_0^2 - 3 x_1^2) / r2) - 2 δ_i1 x_1 (1 + (k - 1 + R2) R2))'
  ns.du_i = 'u_i - uexact_i'

  sqr = domain.boundary['left,bottom'].integral('(u_i n_i)^2 d:x' @ ns, degree=degree*2)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)
  sqr = domain.boundary['top,right'].integral('du_k du_k d:x' @ ns, degree=20)
  cons = solver.optimize('lhs', sqr, droptol=1e-15, constrain=cons)

  res = domain.integral('ubasis_ni,j stress_ij d:x' @ ns, degree=degree*2)
  lhs = solver.solve_linear('lhs', res, constrain=cons)

  bezier = domain.sample('bezier', 5)
  X, stressxx = bezier.eval(['X_i', 'stress_00'] @ ns, lhs=lhs)
  export.triplot('stressxx.png', X, stressxx, tri=bezier.tri, hull=bezier.hull)

  err = domain.integral('<du_k du_k, du_i,j du_i,j>_n d:x' @ ns, degree=max(degree,3)*2).eval(lhs=lhs)**.5
  treelog.user('errors: L2={:.2e}, H1={:.2e}'.format(*err))

  return err, cons, lhs

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

  @testing.requires('matplotlib')
  def test_spline(self):
    err, cons, lhs = main(nelems=4, etype='square', btype='spline', degree=2, traction=.1, maxrefine=2, radius=.5, poisson=.3)
    with self.subTest('l2-error'):
      self.assertAlmostEqual(err[0], .00033, places=5)
    with self.subTest('h1-error'):
      self.assertAlmostEqual(err[1], .00672, places=5)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons, '''
      eNpjaGCAgQYY/K+HYBsYItjWRgj2U6OjxkeM5xqnGZsZqxrDRPXPIek8hzCzBon9Gonte56B4e3VtZfu
      Xjh9Puc8ALOgKgk=''')
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNpbZMYABfKn3pg8N2zV19D/rzfFeKHxOaMjhp6GBoZLjecb6xuvNgo2sjbaYLzRuNbY1Pin0VOjo8ZH
      jOcapxmbGasanzb7cc7knP05/XOSJ93OTD/ndc7ynME5Bobd56ef5z4/51wNkF1x9+5F0Qt6518D2Yuv
      7ry098KK877nGRjeXl176e6F0+dzzgMA63Y//Q==''')

  @testing.requires('matplotlib')
  def test_mixed(self):
    err, cons, lhs = main(nelems=4, etype='mixed', btype='std', degree=2, traction=.1, maxrefine=2, radius=.5, poisson=.3)
    with self.subTest('l2-error'):
      self.assertAlmostEqual(err[0], .00024, places=5)
    with self.subTest('h1-error'):
      self.assertAlmostEqual(err[1], .00739, places=5)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons, '''
      eNpjaGBAQAYkFjpbn6EhyYChAR80NGRoOG6IX421EUPDdCMQa6MxQ8NR4yPGIPZc4yYw7W+cDqSfGpkZ
      qxrjNwcC9c8BbQXikHNY/IAEa4DyG89B5Riwm/UKqEbqPIi14zLIy+evgthLL026CKIdL9y9wNDge/70
      +ZzzANABV94=''')
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNoVj08og3EYx3/NVcpBqx04KRTv+3u+hTggpRxQSLOjC40DJxKpOXBQIvMnbKdFqx2GA1dtHGbP83vf
      11YOkqJQSknJQeZ7+lw+ffoWaR5ncHGI/SalXvWTndNzpEqzrVnr2760/7ncUtacPWZH0Y1RRHGAOAyN
      kA/HlNVruop6qFNr/aCvdQIxTCKJI0RQgw+qRxBpitAXuTRI7ZSiHUphF2mcIIsMFhEq1SOw4McAxvFD
      z9SMWqzLEH/mM/kKDsg2r/I0X/Evt3IH93M3x3mZW9jiNtY8wcN8KjNya14cpRqcmPRK2LxLQG64TlZk
      gxf4kdOslO++zFNqrxD07pysqXLa3EqzJSHjSaO8cbhk+Iuv3rn3/1kKF+6Sk3A3nZSpNl3m3iSlT3Iy
      JX8zb5FS''')
