#! /usr/bin/env python3
#
# In this script we solve the lid driven cavity problem for stationary Stokes
# and Navier-Stokes flow. That is, a unit square domain, with no-slip left,
# bottom and right boundaries and a top boundary that is moving at unit
# velocity in positive x-direction.
#
# The script is identical to :ref:`examples/drivencavity.py` except that it
# uses the Raviart-Thomas discretization providing compatible velocity and
# pressure spaces resulting in a pointwise divergence-free velocity field.

from nutils import mesh, function, solver, export, cli, testing
import numpy, treelog

def main(nelems:int, degree:int, reynolds:float):
  '''
  Driven cavity benchmark problem using compatible spaces.

  .. arguments::

     nelems [12]
       Number of elements along edge.
     degree [2]
       Polynomial degree for velocity; the pressure space is one degree less.
     reynolds [1000]
       Reynolds number, taking the domain size as characteristic length.
  '''

  verts = numpy.linspace(0, 1, nelems+1)
  domain, geom = mesh.rectilinear([verts, verts])

  uxbasis = domain.basis('spline', degree=(degree,degree-1), removedofs=((0,-1),None))
  uybasis = domain.basis('spline', degree=(degree-1,degree), removedofs=(None,(0,-1)))

  Ubasis = function.stack([function.concatenate([uxbasis, function.zeros_like(uybasis)]), function.concatenate([function.zeros_like(uxbasis), uybasis])], axis=1) # OK
  ubasis = function.kronecker(function.concatenate([uxbasis, function.zeros_like(uybasis)]), 1, 2, 0) + function.kronecker(function.concatenate([function.zeros_like(uxbasis), uybasis]), 1, 2, 1) # FAILS

  g = numpy.asarray([[1,2],[3,4]])
  f = ((ubasis.grad(geom) - Ubasis.grad(geom)) * g).sum([1,2])
  print('f.simplified:')
  print(f.simplified.asciitree())
  treelog.info('cmp:', numpy.linalg.norm(domain.integrate(f, degree=2)))

  f = (ubasis.grad(geom) * g).sum([1,2])
  gauss = domain.sample('gauss', 2)
  v1 = 0
  for ielem in range(gauss.nelems):
    v1 += gauss.points[ielem].weights @ f.eval(_transforms=(gauss.transforms[0][ielem],), _points=gauss.points[ielem].coords)
  print(v1)
  v2 = 0
  for ielem in range(gauss.nelems):
    v2 += gauss.points[ielem].weights @ f.simplified.eval(_transforms=(gauss.transforms[0][ielem],), _points=gauss.points[ielem].coords)
  print(v2)
  w = gauss.integrate(f)
  print(w)
  print(v1-w)
  print(v2-w)

if __name__ == '__main__':
  cli.run(main)

class test(testing.TestCase):

  @testing.requires('matplotlib')
  def test_p1(self):
    main(nelems=3, reynolds=100, degree=2)
