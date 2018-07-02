#!/usr/bin/env python3

"""
This example demonstrates how to perform NURBS-based isogeometric analysis
for the elastic analysis of an infinite plate with a circular hole under
tension. A detailed description of the testcase can be found in:
Hughes et al. 'Isogeometric analysis: CAD, finite elements, NURBS, exact 
geometry and mesh refinement', Computer Methods in Applied Mechanics and 
Engineering, Elsevier, 2005, 194, 4135-4195.
"""

from nutils import *
import numpy, unittest
from matplotlib import collections


def main(
    L: 'domain size' = 4.,
    R: 'hole radius' = 1.,
    E: "young's modulus" = 1e5,
    nu: "poisson's ratio" = 0.3,
    T: 'far field traction' = 10,
    nr: 'number of h-refinements' = 2,
  ):

  ns = function.Namespace()
  ns.lmbda = E*nu/(1-nu**2)
  ns.mu = .5*E/(1+nu)

  # create the coarsest level parameter domain
  domain0, geometry = mesh.rectilinear([1, 2])

  # create the second-order B-spline basis over the coarsest domain
  ns.bsplinebasis = domain0.basis('spline', degree=2)
  ns.controlweights = [1] + [.5+.5/2**.5]*2 + [1]*9
  ns.weightfunc = 'bsplinebasis_n controlweights_n'
  ns.nurbsbasis = ns.bsplinebasis * ns.controlweights / ns.weightfunc

  # create the isogeometric map
  ns.controlpoints = [0,R], [R*(1-2**.5),R], [-R,R*(2**.5-1)], [-R,0], [0,.5*(R+L)], [-.15*(R+L),.5*(R+L)], [-.5*(R+L),.15*(R*L)], [-.5*(R+L),0], [0,L], [-L,L], [-L,L], [-L,0]
  ns.x_i = 'nurbsbasis_n controlpoints_ni'

  convergence = []

  for irefine in log.range('refinement', nr+1):

    # create the computational domain by h-refinement
    domain = domain0.refine(irefine)

    # create the second-order B-spline basis over the refined domain
    ns.bsplinebasis = domain.basis('spline', degree=2)
    sqr = domain.integral('(bsplinebasis_n ?lhs_n - weightfunc)^2' @ ns, geometry=ns.x, degree=9)
    ns.controlweights = solver.optimize('lhs', sqr)
    ns.nurbsbasis = ns.bsplinebasis * ns.controlweights / ns.weightfunc

    # prepare the displacement field
    ns.ubasis = ns.nurbsbasis.vector(2)
    ns.u_i = 'ubasis_ni ?lhs_n'
    ns.strain_ij = '(u_i,j + u_j,i) / 2'
    ns.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'

    # construct the exact solution
    ns.r2 = 'x_k x_k'
    ns.R2 = R**2
    ns.T = T
    ns.k = (3-nu) / (1+nu) # plane stress parameter
    ns.uexact_i = '''(T / 4 mu) <x_0 ((k + 1) / 2 + (1 + k) R2 / r2 + (1 - R2 / r2) (x_0^2 - 3 x_1^2) R2 / r2^2),
                                 x_1 ((k - 3) / 2 + (1 - k) R2 / r2 + (1 - R2 / r2) (3 x_0^2 - x_1^2) R2 / r2^2)>_i'''
    ns.strainexact_ij = '(uexact_i,j + uexact_j,i) / 2'
    ns.stressexact_ij = 'lmbda strainexact_kk δ_ij + 2 mu strainexact_ij'

    # define the linear and bilinear forms
    res = domain.integral('ubasis_ni,j stress_ij' @ ns, geometry=ns.x, degree=9)
    res -= domain.boundary['right'].integral('ubasis_ni stressexact_ij n_j' @ ns, geometry=ns.x, degree=9)

    # compute the constraints vector for the symmetry conditions
    sqr = domain.boundary['top,bottom'].integral('(u_i n_i)^2' @ ns, degree=9)
    cons = solver.optimize('lhs', sqr, droptol=1e-15)

    # solve the system of equations
    lhs = solver.solve_linear('lhs', res, constrain=cons)
    ns = ns(lhs=lhs)

    # post-processing
    bezier = domain.sample('bezier', 8)
    x, stressxx = bezier.eval([ns.x, ns.stress[0,0]])
    with export.mplfigure('solution{}.png'.format(irefine)) as fig:
      ax = fig.add_subplot(111, aspect='equal')
      im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, stressxx, shading='gouraud', cmap='jet')
      im.set_clim(numpy.nanmin(stressxx),numpy.nanmax(stressxx))
      ax.add_collection(collections.LineCollection(x[bezier.hull], colors='k', linewidths=.1))
      ax.autoscale(enable=True, axis='both', tight=True)
      fig.colorbar(im)

    # compute the L2-norm of the error in the stress
    err = domain.integrate('(?dstress_ij ?dstress_ij)(dstress_ij = stress_ij - stressexact_ij)' @ ns, geometry=ns.x, ischeme='gauss9')**.5

    # approximate mesh parameter (maximum physical distance between knots) from area
    hmax = max(domain.integrate_elementwise(1, degree=2, geometry=ns.x))**.5
    convergence.append((hmax, err))

  with export.mplfigure('convergence.png') as fig:
    ax = fig.add_subplot(111, xlabel='mesh parameter', title='L2 error of stress')
    ax.loglog(*numpy.array(convergence).T, 'k*--')
    ax.grid(True)

  return convergence


class test(unittest.TestCase):

  def test1(self):
    conv = main(nr=2)
    numpy.testing.assert_almost_equal(conv, [[2.77449, 3.917807], [1.793076, 2.930154], [1.041247, 1.476470]], decimal=6)

  def test2(self):
    conv = main(L=3, R=1.5, E=1e6, nu=0.4, T=15, nr=3)
    numpy.testing.assert_almost_equal(conv, [[1.902403, 6.005827], [1.140464, 3.374952], [0.645049, 1.060993], [0.346868, 0.254238]], decimal=6)


if __name__ == '__main__':
  cli.run(main)
