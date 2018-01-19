#!/usr/bin/env python3

"""
This example demonstrates how to perform NURBS-based isogeometric analysis
for the elastic analysis of an infinite plate with a circular hole under
tension. A detailed description of the testcase can be found in:
Hughes et al. 'Isogeometric analysis: CAD, finite elements, NURBS, exact 
geometry and mesh refinement', Computer Methods in Applied Mechanics and 
Engineering, Elsevier, 2005, 194, 4135-4195.
"""

from nutils import cli, mesh, function, plot, log, solver
import numpy, unittest


def main(
    L: 'domain size' = 4.,
    R: 'hole radius' = 1.,
    E: "young's modulus" = 1e5,
    nu: "poisson's ratio" = 0.3,
    T: 'far field traction' = 10,
    nr: 'number of h-refinements' = 2,
    figures: 'create figures' = True,
  ):

  ns = function.Namespace()
  ns.lmbda = E*nu/(1-nu**2)
  ns.mu = .5*E/(1+nu)

  # create the coarsest level parameter domain
  domain0, geometry = mesh.rectilinear( [1,2] )

  # create the second-order B-spline basis over the coarsest domain
  ns.bsplinebasis = domain0.basis('spline', degree=2)
  ns.controlweights = 1,.5+.5/2**.5,.5+.5/2**.5,1,1,1,1,1,1,1,1,1
  ns.weightfunc = 'bsplinebasis_n controlweights_n'
  ns.nurbsbasis = ns.bsplinebasis * ns.controlweights / ns.weightfunc

  # create the isogeometric map
  ns.controlpoints = [0,R],[R*(1-2**.5),R],[-R,R*(2**.5-1)],[-R,0],[0,.5*(R+L)],[-.15*(R+L),.5*(R+L)],[-.5*(R+L),.15*(R*L)],[-.5*(R+L),0],[0,L],[-L,L],[-L,L],[-L,0]
  ns.x_i = 'nurbsbasis_n controlpoints_ni'

  # create the computational domain by h-refinement
  domain = domain0.refine(nr)

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
  if figures:
    geom, stressxx = domain.simplex.elem_eval([ns.x, 'stress_00' @ ns], ischeme='bezier8', separate=True)
    with plot.PyPlot( 'solution', index=nr ) as plt:
      plt.mesh( geom, stressxx )
      plt.colorbar()

  # compute the L2-norm of the error in the stress
  err = domain.integrate('(?dstress_ij ?dstress_ij)(dstress_ij = stress_ij - stressexact_ij)' @ ns, geometry=ns.x, ischeme='gauss9')**.5

  # compute the mesh parameter (maximum physical distance between knots)
  hmax = max(numpy.linalg.norm(v[:,numpy.newaxis]-v, axis=2).max() for v in domain.elem_eval(ns.x, ischeme='bezier2', separate=True))

  return err, hmax


def convergence(nrefine=5):
  err, h = numpy.array([main(nr=irefine) for irefine in log.range('refine', nrefine)]).T
  with plot.PyPlot( 'convergence' ) as plt:
    plt.loglog(h, err, 'k*--')
    plt.slope_triangle(h, err)
    plt.ylabel('L2 error of stress')
    plt.grid(True)


class test(unittest.TestCase):

  def test0(self):
    err, hmax = main(nr=0, figures=False)
    numpy.testing.assert_almost_equal(err, 3.917807, decimal=6)
    numpy.testing.assert_almost_equal(hmax, 5.0, decimal=6)

  def test1(self):
    err, hmax = main(nr=2, figures=False)
    numpy.testing.assert_almost_equal(err, 1.476470, decimal=6)
    numpy.testing.assert_almost_equal(hmax, 2.028562, decimal=6)

  def test2(self):
    err, hmax = main(L=3, R=1.5, E=1e6, nu=0.4, T=15, nr=3, figures=False)
    numpy.testing.assert_almost_equal(err, 0.254238, decimal=6)
    numpy.testing.assert_almost_equal(hmax, 0.768562, decimal=6)


if __name__ == '__main__':
  cli.choose(main, convergence)
