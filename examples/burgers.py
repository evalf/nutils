#! /usr/bin/env python3
#
# In this script we solve the Burgers equation on a 1D or 2D periodic domain,
# starting from a centered Gaussian and convecting in the positive direction of
# the first coordinate.

from nutils import mesh, function, solver, export, cli, testing
import numpy, treelog

def main(nelems:int, ndims:int, btype:str, degree:int, timescale:float, newtontol:float, endtime:float):
  '''
  Burgers equation on a 1D or 2D periodic domain.

  .. arguments::

     nelems [20]
       Number of elements along a single dimension.
     ndims [1]
       Number of spatial dimensions.
     btype [discont]
       Type of basis function (discont/legendre).
     degree [1]
       Polynomial degree for discontinuous basis functions.
     timescale [.5]
       Fraction of timestep and element size: timestep=timescale/nelems.
     newtontol [1e-5]
       Newton tolerance.
     endtime [inf]
       Stopping time.
  '''

  domain, geom = mesh.rectilinear([numpy.linspace(0,1,nelems+1)]*ndims, periodic=range(ndims))

  ns = function.Namespace()
  ns.x = geom
  ns.basis = domain.basis(btype, degree=degree)
  ns.u = 'basis_n ?lhs_n'
  ns.f = '.5 u^2'
  ns.C = 1

  res = domain.integral('-d(basis_n, x_i) δ_i0 f J(x)' @ ns, degree=5)
  res += domain.interfaces.integral('-[basis_n] n(x_i) δ_i0 ({f} - .5 C [u] n(x_j) δ_j0) J(x)' @ ns, degree=degree*2)
  inertia = domain.integral('basis_n u J(x)' @ ns, degree=5)

  sqr = domain.integral('(u - exp(-sum:i((5 (x_i - 0.5_i))^2)))^2 J(x)' @ ns, degree=5)
  lhs0 = solver.optimize('lhs', sqr)

  timestep = timescale/nelems
  bezier = domain.sample('bezier', 7)
  with treelog.iter.plain('timestep', solver.impliciteuler('lhs', res, inertia, timestep=timestep, arguments=dict(lhs=lhs0), newtontol=newtontol)) as steps:
    for itime, lhs in enumerate(steps):
      x, u = bezier.eval(['x', 'u'] @ ns, lhs=lhs)
      export.triplot('solution.png', x, u, tri=bezier.tri, hull=bezier.hull, clim=(0,1))
      if itime * timestep >= endtime:
        break

  return lhs

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to simulate until 0.5 seconds run :sh:`python3 burgers.py
# endtime=0.5`.

if __name__ == '__main__':
  cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(testing.TestCase):

  @testing.requires('matplotlib')
  def test_1d_p0(self):
    lhs = main(ndims=1, nelems=10, timescale=.1, btype='discont', degree=0, endtime=.01, newtontol=1e-5)
    self.assertAlmostEqual64(lhs, '''
      eNrz1ttqGGOiZSZlrmbuZdZgcsEwUg8AOqwFug==''')

  @testing.requires('matplotlib')
  def test_1d_p1(self):
    lhs = main(ndims=1, nelems=10, timescale=.1, btype='discont', degree=1, endtime=.01, newtontol=1e-5)
    self.assertAlmostEqual64(lhs, '''
      eNrbocann6u3yqjTyMLUwfSw2TWzKPNM8+9mH8wyTMNNZxptMirW49ffpwYAI6cOVA==''')

  @testing.requires('matplotlib')
  def test_1d_p2(self):
    lhs = main(ndims=1, nelems=10, timescale=.1, btype='discont', degree=2, endtime=.01, newtontol=1e-5)
    self.assertAlmostEqual64(lhs, '''
      eNrr0c7SrtWfrD/d4JHRE6Ofxj6mnqaKZofNDpjZmQeYB5pHmL8we23mb5ZvWmjKY/LV6KPRFIMZ+o36
      8dp92gCxZxZG''')

  @testing.requires('matplotlib')
  def test_1d_p1_legendre(self):
    lhs = main(ndims=1, nelems=10, timescale=.1, btype='legendre', degree=1, endtime=.01, newtontol=1e-5)
    self.assertAlmostEqual64(lhs, '''
      eNrbpbtGt9VQyNDfxMdYzczERNZczdjYnOdsoNmc01kmE870Gj49t0c36BIAAhsO1g==''')

  @testing.requires('matplotlib')
  def test_1d_p2_legendre(self):
    lhs = main(ndims=1, nelems=10, timescale=.1, btype='legendre', degree=2, endtime=.01, newtontol=1e-5)
    self.assertAlmostEqual64(lhs, '''
      eNoBPADD/8ot2y2/K4UxITFFLk00RTNNLyY2KzTTKx43QjOOzzM3Ss0pz1A2qsvhKGk0jsyXL48xzc5j
      LswtIdLIK5SlF78=''')

  @testing.requires('matplotlib')
  def test_2d_p1(self):
    lhs = main(ndims=2, nelems=4, timescale=.1, btype='discont', degree=1, endtime=.01, newtontol=1e-5)
    import os
    if os.environ.get('NUTILS_TENSORIAL'):
      lhs = lhs.reshape(4,2,4,2).transpose(0,2,1,3).ravel()
    self.assertAlmostEqual64(lhs, '''
      eNoNyKENhEAQRuGEQsCv2SEzyQZHDbRACdsDJNsBjqBxSBxBHIgJ9xsqQJ1Drro1L1/eYBZceGz8njrR
      yacm8UQLBvPYCw1airpyUVYSJLhKijK4IC01WDnqqxvX8OTl427aU73sctPGr3qqceBnRzOjo0xy9JpJ
      R73m6R6YMZo/Q+FCLQ==''')
