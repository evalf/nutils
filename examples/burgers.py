#! /usr/bin/python3
#
# In this script we solve the Burgers equation on a 1D or 2D periodic dommain,
# starting from a centered Gaussian and convecting in the positive direction of
# the first coordinate.

import nutils, numpy

# The main function defines the parameter space for the script. Configurable
# parameters are the mesh density (in number of elements along an edge), number
# of dimensions, polynomial degree, time scale, Newton tolerance and the
# stopping time.

def main(nelems: 'number of elements' = 20,
         ndims: 'spatial dimension' = 1,
         degree: 'polynomial degree' = 1,
         timescale: 'time scale (timestep=timescale/nelems)' = .5,
         newtontol: 'solver tolerance' = 1e-5,
         endtime: 'end time' = numpy.inf):

  domain, geom = nutils.mesh.rectilinear([numpy.linspace(0,1,nelems+1)]*ndims, periodic=range(ndims))

  ns = nutils.function.Namespace()
  ns.x = geom
  ns.basis = domain.basis('discont', degree=degree)
  ns.u = 'basis_n ?lhs_n'
  ns.f = '.5 u^2'
  ns.C = 1

  res = domain.integral('-basis_n,0 f' @ ns, geometry=ns.x, degree=5)
  res += domain.interfaces.integral('-[basis_n] n_0 ({f} - .5 C [u] n_0)' @ ns, geometry=ns.x, degree=degree*2)
  inertia = domain.integral('basis_n u' @ ns, geometry=ns.x, degree=5)

  sqr = domain.integral('(u - exp(-?y_i ?y_i)(y_i = 5 (x_i - 0.5_i)))^2' @ ns, geometry=ns.x, degree=5)
  lhs0 = nutils.solver.optimize('lhs', sqr)

  timestep = timescale/nelems
  bezier = domain.sample('bezier', 7)
  for itime, lhs in nutils.log.enumerate('timestep', nutils.solver.impliciteuler('lhs', res, inertia, timestep=timestep, lhs0=lhs0, newtontol=newtontol)):
    x, u = bezier.eval([ns.x, ns.u], arguments=dict(lhs=lhs))
    nutils.export.triplot('solution.jpg', x, u, tri=bezier.tri, hull=bezier.hull, clim=(0,1))
    if itime * timestep >= endtime:
      break

  return lhs

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to simulate until 0.5 seconds run :sh:`python3 burgers.py
# endtime=0.5`.

if __name__ == '__main__':
  nutils.cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategicly chosen return values for routine regression testing. Here we use
# the standard :mod:`unittest` framework, with
# :func:`nutils.numeric.assert_allclose64` facilitating the embedding of
# desired results as compressed base64 data.

import unittest

class test(unittest.TestCase):

  def test_1d_p1(self):
    lhs = main(ndims=1, nelems=10, timescale=.1, degree=1, endtime=.01)
    nutils.numeric.assert_allclose64(lhs, 'eNrbocann6u3yqjTyMLUwfSw2TWzKPNM8+9mH8wyTMNNZxptMir'
      'W49ffpwYAI6cOVA==')

  def test_1d_p2(self):
    lhs = main(ndims=1, nelems=10, timescale=.1, degree=2, endtime=.01)
    nutils.numeric.assert_allclose64(lhs, 'eNrr0c7SrtWfrD/d4JHRE6Ofxj6mnqaKZofNDpjZmQeYB5pHmL8'
      'we23mb5ZvWmjKY/LV6KPRFIMZ+o368dp92gCxZxZG')

  def test_2d_p1(self):
    lhs = main(ndims=2, nelems=4, timescale=.1, degree=1, endtime=.01)
    nutils.numeric.assert_allclose64(lhs, 'eNoNyKENhEAQRuGEQsCv2SEzyQZHDbRACdsDJNsBjqBxSBxBHIg'
      'J9xsqQJ1Drro1L1/eYBZceGz8njrRyacm8UQLBvPYCw1airpyUVYSJLhKijK4IC01WDnqqxvX8OTl427'
      'aU73sctPGr3qqceBnRzOjo0xy9JpJR73m6R6YMZo/Q+FCLQ==')
