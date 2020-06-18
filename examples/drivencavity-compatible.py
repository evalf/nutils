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

  ns = function.Namespace()
  ns.x = geom
  ns.Ubasis = function.stack([function.concatenate([uxbasis, function.zeros_like(uybasis)]), function.concatenate([function.zeros_like(uxbasis), uybasis])], axis=1) # OK
  ns.ubasis = function.kronecker(function.concatenate([uxbasis, function.zeros_like(uybasis)]), 1, 2, 0) + function.kronecker(function.concatenate([function.zeros_like(uxbasis), uybasis]), 1, 2, 1) # FAILS

  treelog.info('cmp1:', numpy.linalg.norm(domain.integrate('(ubasis_ni,j - Ubasis_ni,j) δ_ij' @ ns, degree=2*degree)))
  ns.g = function.asarray([[1,2],[3,4]])
  treelog.info('cmp2:', numpy.linalg.norm(domain.integrate('(ubasis_ni,j - Ubasis_ni,j) g_ij' @ ns, degree=2*degree)))
  ns.G = function.Guard([[1,2],[3,4]])
  treelog.info('cmp3:', numpy.linalg.norm(domain.integrate('(ubasis_ni,j - Ubasis_ni,j) G_ij' @ ns, degree=2*degree)))

if __name__ == '__main__':
  cli.run(main)

class test(testing.TestCase):

  @testing.requires('matplotlib')
  def test_p1(self):
    main(nelems=3, reynolds=100, degree=2)
